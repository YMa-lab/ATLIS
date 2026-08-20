"""Joint deconvolution across time slices — torch / GPU version.

Entry point: `solve_V_all_slices_torch`. Everything else here is a dependency.
Structured to mirror the NumPy module section-for-section, so the two can be
diffed against each other.

Everything after data loading stays on device: Laplacians, cost matrices,
couplings, block assembly, and the V solve. The `fgw` and `sinkhorn` branches
are the exceptions — POT and ott are host-side, so those pay a transfer.

Optional backends (FRLC, POT, ott) are imported lazily inside the branch that
needs them, so the module loads without them unless that solver is used.
"""

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Device and format helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_device(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def to_torch_csr(A, device, dtype=torch.float32):
    """scipy sparse -> torch sparse CSR on device."""
    A = A.tocsr()
    return torch.sparse_csr_tensor(
        torch.as_tensor(A.indptr, dtype=torch.int64, device=device),
        torch.as_tensor(A.indices, dtype=torch.int64, device=device),
        torch.as_tensor(A.data, dtype=dtype, device=device),
        size=A.shape,
    )


def csr_to_scipy(A):
    """torch sparse CSR -> scipy CSR, for saving / interop."""
    A = A.cpu()
    return sp.csr_matrix(
        (A.values().numpy(), A.col_indices().numpy(), A.crow_indices().numpy()),
        shape=tuple(A.shape),
    )


def coo_from(idx, val, shape, device):
    """Coalesced sparse CSR from explicit indices and values."""
    return (
        torch.sparse_coo_tensor(idx, val, size=shape, device=device)
        .coalesce()
        .to_sparse_csr()
    )


def scale_csr(A, s, device):
    """Multiply a torch sparse CSR by a scalar."""
    coo = A.to_sparse_coo()
    return coo_from(coo.indices(), s * coo.values(), tuple(A.shape), device)


def add_csr(A, B):
    """Add two torch sparse CSR matrices of the same shape."""
    return (A.to_sparse_coo() + B.to_sparse_coo()).coalesce().to_sparse_csr()


def diag_csr(v, device):
    """Sparse CSR diagonal matrix from a 1D tensor."""
    n = v.shape[0]
    idx = torch.arange(n, device=device)
    return coo_from(torch.stack([idx, idx]), v, (n, n), device)


def csr_diagonal(A, n, device):
    """Extract the diagonal of a torch sparse CSR as a dense vector."""
    crow, cols, vals = A.crow_indices(), A.col_indices(), A.values()
    rows = torch.repeat_interleave(
        torch.arange(n, device=device), crow[1:] - crow[:-1]
    )
    out = torch.zeros(n, dtype=vals.dtype, device=device)
    on_diag = rows == cols
    out.index_add_(0, rows[on_diag], vals[on_diag])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Coupling sparsification
# ─────────────────────────────────────────────────────────────────────────────
def topk_mask(A, k, dim):
    """Boolean mask keeping the k largest entries along `dim` of dense A."""
    k = min(k, A.shape[1 - dim])
    idx = torch.topk(A, k, dim=dim).indices
    mask = torch.zeros_like(A, dtype=torch.bool)
    mask.scatter_(dim, idx, True)
    return mask


def sinkhorn_scale_dense(A, row_target, col_target, n_iter=50, eps=1e-12):
    """Rescale a nonnegative dense matrix toward the given marginals.

    A is expected to be already pruned, so it is mostly zeros despite the
    dense storage.
    """
    c = torch.ones(A.shape[1], dtype=A.dtype, device=A.device)
    for _ in range(n_iter):
        r = row_target / torch.clamp(A @ c, min=eps)
        c = col_target / torch.clamp(A.T @ r, min=eps)
    return r[:, None] * A * c[None, :]


def sparsify_pi_torch(pi, k_row=None, k_col=None, mode="doubly", n_iter=50):
    """Prune a dense coupling to its top-k entries per row/column, then rescale.

    The pruned support is the union of the row-wise and column-wise top-k, so
    no row or column is emptied unless k is smaller than the support allows.
    `mode` controls which marginals are restored afterwards.
    """
    if (pi < 0).any():
        raise ValueError("pi must be nonnegative.")

    row_target = pi.sum(dim=1)
    col_target = pi.sum(dim=0)

    mask_row = topk_mask(pi, k_row, dim=1) if k_row is not None else pi > 0
    mask_col = topk_mask(pi, k_col, dim=0) if k_col is not None else pi > 0
    A = pi * (mask_row | mask_col)

    if (A.sum(dim=1) == 0).any():
        raise ValueError("Empty row after sparsification. Increase k_row/k_col.")
    if (A.sum(dim=0) == 0).any():
        raise ValueError("Empty col after sparsification. Increase k_row/k_col.")

    mode = mode.lower()
    if mode == "row":
        return (row_target / torch.clamp(A.sum(dim=1), min=1e-12))[:, None] * A
    if mode == "col":
        return A * (col_target / torch.clamp(A.sum(dim=0), min=1e-12))[None, :]
    if mode == "doubly":
        return sinkhorn_scale_dense(A, row_target, col_target, n_iter=n_iter)
    raise ValueError("mode must be 'row', 'col', or 'doubly'.")


# ─────────────────────────────────────────────────────────────────────────────
# Graph structure
# ─────────────────────────────────────────────────────────────────────────────
def assemble_blocks_torch(
    diag_blocks, offdiag_blocks=(), device=None, dtype=torch.float32
):
    """Build a block matrix from diagonal blocks plus (i, j, B_ij) entries.

    Blocks may be sparse CSR or dense. Torch has no LIL equivalent, so this
    collects triplets and builds one COO rather than assigning into slices.
    """
    device = get_device(device)

    r_sizes = [b.shape[0] for b in diag_blocks]
    c_sizes = [b.shape[1] for b in diag_blocks]
    r_off = np.r_[0, np.cumsum(r_sizes[:-1])]
    c_off = np.r_[0, np.cumsum(c_sizes[:-1])]

    rows_all, cols_all, vals_all = [], [], []

    def add(block, r0, c0):
        if block.layout == torch.sparse_csr:
            crow = block.crow_indices()
            rr = torch.repeat_interleave(
                torch.arange(block.shape[0], device=device), crow[1:] - crow[:-1]
            )
            cc, vv = block.col_indices(), block.values()
        else:
            rr, cc = block.nonzero(as_tuple=True)
            vv = block[rr, cc]
        rows_all.append(rr + int(r0))
        cols_all.append(cc + int(c0))
        vals_all.append(vv)

    for i, Bii in enumerate(diag_blocks):
        add(Bii, r_off[i], c_off[i])

    for i, j, Bij in offdiag_blocks:
        ri, cj = r_sizes[i], c_sizes[j]
        if tuple(Bij.shape) != (ri, cj):
            raise ValueError(
                f"Offdiag block ({i},{j}) must be shape {(ri, cj)}, "
                f"got {tuple(Bij.shape)}"
            )
        add(Bij, r_off[i], c_off[j])

    return coo_from(
        torch.stack([torch.cat(rows_all), torch.cat(cols_all)]),
        torch.cat(vals_all).to(dtype),
        (sum(r_sizes), sum(c_sizes)),
        device,
    )


def laplacian_from_coords_torch(
    coords, k=20, mutual=False, normed=False, device=None,
    dtype=torch.float32, block=4096,
):
    """Unweighted k-NN graph Laplacian from 2D coordinates, on device.

    `coords` is an (n, 2) tensor or array. The neighbour search is blocked so
    the full n x n distance matrix is never materialized, and the Laplacian is
    built by hand since scipy's csgraph is host-side.

    `mutual=True` keeps edge i–j only when each is in the other's kNN;
    otherwise the directed graph is symmetrized by union.

    Returns (L, W) as sparse CSR.
    """
    device = get_device(device)
    coords = torch.as_tensor(coords, dtype=dtype, device=device)
    n = coords.shape[0]
    k = min(k, n - 1)

    chunks = []
    for s in range(0, n, block):
        D = torch.cdist(coords[s:s + block], coords)
        chunks.append(torch.topk(D, k + 1, dim=1, largest=False).indices[:, 1:])
        del D
    nbrs = torch.cat(chunks, dim=0)

    row = torch.repeat_interleave(torch.arange(n, device=device), k)
    col = nbrs.reshape(-1)

    if mutual:
        keep = torch.isin(row * n + col, col * n + row)
        row, col = row[keep], col[keep]
        idx = torch.stack([row, col])
    else:
        idx = torch.stack([torch.cat([row, col]), torch.cat([col, row])])

    val = torch.ones(idx.shape[1], dtype=dtype, device=device)
    W = torch.sparse_coo_tensor(idx, val, size=(n, n), device=device).coalesce()
    # coalesce sums duplicate edges to 2; the graph is unweighted.
    W = torch.sparse_coo_tensor(
        W.indices(), W.values().clamp_max(1.0), size=(n, n), device=device
    ).coalesce()

    deg = torch.sparse.sum(W, dim=1).to_dense()
    wi, wj = W.indices()
    wv = W.values()

    if normed:
        dinv = torch.where(deg > 0, deg.rsqrt(), torch.zeros_like(deg))
        off_val = -dinv[wi] * wv * dinv[wj]
        diag_val = torch.where(deg > 0, torch.ones_like(deg), torch.zeros_like(deg))
    else:
        off_val = -wv
        diag_val = deg

    dia = torch.arange(n, device=device)
    L = coo_from(
        torch.cat([torch.stack([wi, wj]), torch.stack([dia, dia])], dim=1),
        torch.cat([off_val, diag_val]),
        (n, n),
        device,
    )
    return L, W.to_sparse_csr()


def proj_simplex_rows_torch(V, z=1.0, proj=False):
    """Project each row onto {x >= 0, sum(x) = z}; with proj=False, clamp at 0."""
    n, m = V.shape

    if proj:
        U, _ = torch.sort(V, dim=1, descending=True)
        cssv = torch.cumsum(U, dim=1)
        j = torch.arange(1, m + 1, device=V.device, dtype=V.dtype)
        rho = ((U * j) > (cssv - z)).sum(dim=1) - 1
        theta = (
            cssv[torch.arange(n, device=V.device), rho] - z
        ) / (rho.to(V.dtype) + 1.0)
        X = V - theta[:, None]
    else:
        X = V

    return X.clamp_min(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Inner solve: min_V ||X - B V^T||_F^2 + tr(V^T L V)
# ─────────────────────────────────────────────────────────────────────────────
def solve_V_universal_torch(
    X, B, L,
    niter_max=1e3, tol=1e-5, eta=1e-3, verbose=True,
    V0=None, return_obj=False, proj_simplex=False, step_rate="backtrack",
    device=None, dtype=torch.float32, check_every=1,
):
    """Projected gradient descent on V, never forming the residual X - B V^T.

    The objective is expanded as
        ||X||_F^2 - 2 tr(V^T G) + tr(V^T V H) + tr(V^T L V),
    with G = X^T B and H = B^T B precomputed, so the sparse X is touched once.

    X is (d, n) sparse CSR or dense, B is (d, r) dense, L is (n, n) sparse CSR.

    `check_every > 1` evaluates the objective only every few iterations under
    the 'automatic' step rule, removing most host syncs. Note this also makes
    rel_drop span several steps, so the stopping point differs from
    check_every=1 — leave it at 1 to match the NumPy version.

    Returns V, or (V, reg_term, data_term) when `return_obj`.
    """
    device = get_device(device)

    d, n = X.shape
    dB, r = B.shape
    assert d == dB, f"X is {tuple(X.shape)}, B is {tuple(B.shape)} (dim mismatch)"

    if X.layout == torch.sparse_csr:
        G = torch.sparse.mm(X.t(), B)
        xnorm2 = (X.values() ** 2).sum()
    else:
        G = X.T @ B
        xnorm2 = (X * X).sum()
    H = B.T @ B

    dL = csr_diagonal(L, n, device)   # only used by the 'automatic' step

    if V0 is None:
        gen = torch.Generator(device="cpu").manual_seed(0)
        V = torch.rand((n, r), generator=gen, dtype=dtype).to(device)
    else:
        V = V0.to(device=device, dtype=dtype)
        assert tuple(V.shape) == (n, r)

    def objective(V):
        data = xnorm2 - 2.0 * (V * G).sum() + ((V @ H) * V).sum()
        return data + ((L @ V) * V).sum()

    def gradient(V):
        return 2.0 * (V @ H - G) + 2.0 * (L @ V)

    f_prev = float(objective(V))

    for it in range(int(niter_max)):
        g = gradient(V)
        V_old = V

        if step_rate == "backtrack":
            step = float(eta)
            gnorm2 = float((g * g).sum())
            while True:
                V_trial = proj_simplex_rows_torch(
                    V_old - step * g, z=1.0, proj=proj_simplex
                )
                f_trial = float(objective(V_trial))
                # Armijo condition, or give up once the step is negligible.
                if f_trial <= f_prev - 1e-4 * step * gnorm2 or step < 1e-8:
                    V, f_curr = V_trial, f_trial
                    break
                step *= 0.5

        elif step_rate == "automatic":
            # Multiplicative-style step; assumes L is diagonally dominant.
            step_mat = V_old / (2.0 * (V_old @ H + dL[:, None] * V_old + 1e-15))
            V = V_old - step_mat * g
            if it % check_every:
                continue
            f_curr = float(objective(V))
            step = float(step_mat.mean())

        else:
            raise ValueError("step_rate must be 'backtrack' or 'automatic'")

        rel_drop = (f_prev - f_curr) / max(1.0, abs(f_prev))

        if verbose and (it % 50 == 0 or rel_drop < tol):
            label = "step" if step_rate == "backtrack" else "mean step"
            print(
                f"iter {it:4d}  f={f_curr:.6e}  "
                f"rel_drop={rel_drop:.3e}  {label}={step:.2e}"
            )

        if f_curr <= f_prev and rel_drop < tol:
            if verbose:
                print("stopped after iteration #", it)
            break

        f_prev = f_curr

    if not return_obj:
        return V
    data_term = float(xnorm2 - 2.0 * (V * G).sum() + ((V @ H) * V).sum())
    return V, float(((L @ V) * V).sum()), data_term


def _X_to_torch_csr(adata_sub, device, dtype):
    """(genes x spots) counts as torch sparse CSR on device."""
    Xi = adata_sub.layers["counts"].T if "counts" in adata_sub.layers else adata_sub.X.T
    if not sp.issparse(Xi):
        Xi = sp.csr_matrix(np.asarray(Xi))
    return to_torch_csr(Xi.tocsr(), device, dtype=dtype)


def _coords_of(adata_slice, device, dtype):
    return torch.as_tensor(
        np.column_stack([
            adata_slice.obs["x"].to_numpy(),
            adata_slice.obs["y"].to_numpy(),
        ]),
        dtype=dtype,
        device=device,
    )


def solve_V_per_slice_torch(
    adata, pd_B, lam=10, k=20, niter_max=1e3, tol=1e-5, eta=1e-3,
    verbose=True, return_obj=False, proj_simplex=False, step_rate="backtrack",
    L=None, device=None, dtype=torch.float32,
):
    """Deconvolve a single slice with only its own spatial Laplacian.

    Pass `L` to reuse an already-scaled Laplacian instead of rebuilding it.
    """
    device = get_device(device)

    common = adata.var_names.intersection(pd_B.columns)
    adata_sub = adata[:, common].copy()

    if L is None:
        L, _ = laplacian_from_coords_torch(
            _coords_of(adata_sub, device, dtype),
            k=k, normed=True, device=device, dtype=dtype,
        )
        L = scale_csr(L, lam, device)

    B = torch.as_tensor(pd_B[common].to_numpy().T, dtype=dtype, device=device)
    X = _X_to_torch_csr(adata_sub, device, dtype)

    return solve_V_universal_torch(
        X, B, L,
        niter_max=niter_max, tol=tol, eta=eta, verbose=verbose,
        return_obj=return_obj, proj_simplex=proj_simplex, step_rate=step_rate,
        device=device, dtype=dtype,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Couplings between consecutive slices
# ─────────────────────────────────────────────────────────────────────────────
def _spatial_cost(coords):
    """Max-normalized squared-distance cost within one slice."""
    C = torch.cdist(coords, coords).pow_(2)
    return C / (C.max() + 1e-15)


def _frlc_coupling(C, C1=None, C2=None, alpha=0.9, inner_iters=300,
                   device=None, **kwargs):
    """Low-rank OT via FRLC. C1/C2 present -> fused GW; absent -> Wasserstein."""
    import sys
    sys.path.insert(0, '../../FRLC/src/')
    import FRLC

    device = get_device(device)
    r = min(C.shape[0], C.shape[1], 100)

    args = dict(
        device=device, r=r,
        min_iter=100, max_iter=100,
        max_inneriters_balanced=inner_iters,
        max_inneriters_relaxed=inner_iters,
        min_iterGW=100,
        returnFull=True, printCost=False,
    )
    if C1 is not None:
        args.update(A=C1, B=C2, alpha=alpha, Wasserstein=False, FGW=True)
    else:
        args.update(Wasserstein=True, FGW=False)
    args.update(kwargs)
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)

    pi, _ = FRLC.FRLC_opt(C, **args)
    return pi.detach()


def _compute_coupling(
    ot_solver, C, coords_i, coords_j, V_i, V_j, a, b,
    alpha, verbose, device, dtype,
):
    """Dispatch to the requested OT backend and return a dense coupling."""
    if ot_solver == "fgw":
        from ot.gromov import fused_gromov_wasserstein

        C1, C2 = _spatial_cost(coords_i), _spatial_cost(coords_j)

        # POT is host-side only, so this branch pays a transfer.
        to_host = lambda T: T.cpu().numpy().astype(np.float64)
        pi = fused_gromov_wasserstein(
            to_host(C), to_host(C1), to_host(C2), to_host(a), to_host(b),
            loss_fun="square_loss", alpha=alpha,
        )
        pi = torch.as_tensor(pi, dtype=dtype, device=device)

        # A near-constant plan means the solver stalled; retry low-rank.
        if float(pi.std()) < 1e-10:
            if verbose:
                print("constant coupling... fall back", tuple(C.shape))
            pi = _frlc_coupling(C, C1, C2, alpha=alpha, device=device)
        return pi.to(dtype)

    if ot_solver == "sinkhorn":
        from ott.geometry.pointcloud import PointCloud
        from ott.problems.linear.linear_problem import LinearProblem
        from ott.solvers.linear.sinkhorn import Sinkhorn

        denom = max(float(V_i.abs().max()), float(V_j.abs().max()), 1e-15)
        geom = PointCloud(
            V_i.cpu().numpy() / denom, V_j.cpu().numpy() / denom, epsilon=0.01
        )
        out = Sinkhorn()(LinearProblem(geom))
        return torch.as_tensor(np.array(out.matrix), dtype=dtype, device=device)

    if ot_solver == "sinkhorn_lr":
        return _frlc_coupling(C, device=device).to(dtype)

    if ot_solver in {"low_rank", "low_rank_SR"}:
        sr = ot_solver == "low_rank_SR"
        pi = _frlc_coupling(
            C.double(),
            _spatial_cost(coords_i).double(),
            _spatial_cost(coords_j).double(),
            alpha=alpha, inner_iters=100, device=device,
            semiRelaxedLeft=sr, semiRelaxedRight=sr,
        )
        return pi.to(dtype)

    raise ValueError(f"Unknown ot_solver: {ot_solver}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def solve_V_all_slices_torch(
    adata_group, pd_B,
    lam=10, mu=1, time=None,
    outer_max=5, niter_max=1e3, tol=1e-5, eta=1e-3,
    verbose=True, coupling=None, k=10, init="per_slice",
    V_init=None, ot_solver="fgw", one_step=True,
    step_rate="automatic", alpha=0.9,
    dtype=torch.float32, device=None,
    k_graph=None, return_numpy=True,
):
    """Deconvolve all slices jointly, coupling consecutive time points via OT.

    Alternates between (i) computing couplings pi_t between adjacent slices and
    (ii) solving for V under a block Laplacian that combines each slice's
    spatial graph with the coupling-induced cross-slice penalty, weighted by mu.
    `mu < 0` skips the V update entirely and only returns the couplings.

    V and the couplings are dense by design; the couplings are pruned after
    solving, but the OT solvers still build dense cost matrices.

    `k_graph` sets the spatial kNN degree separately from `k`, which drives the
    coupling sparsification via int(k/4). `return_numpy` converts the results
    back to numpy / scipy on the way out.

    Returns (V_per_slice, pi_list, L).
    """
    device = get_device(device)
    if k_graph is None:
        k_graph = k

    n_times = len(adata_group)

    n_spots = [0]
    for a_i in adata_group:
        n_spots.append(n_spots[-1] + a_i.shape[0])

    adata = ad.concat(adata_group, axis=0, join="outer")
    common = adata.var_names.intersection(pd_B.columns)

    adata_sub = adata[:, common].copy()
    adata_group = [a_i[:, common].copy() for a_i in adata_group]
    pd_B_sub = pd_B[common]

    if time is None:
        time = list(range(n_times))
    if time[-1] == time[0]:
        rescaled_time = [0.0] * len(time)
    else:
        span = time[-1] - time[0]
        rescaled_time = [(t - time[0]) / span for t in time]

    # Coords stay on device, reused for the spatial graphs and the GW costs.
    coords = [_coords_of(a_i, device, dtype) for a_i in adata_group]

    base_diag_blocks = []
    for c_i in coords:
        Li, _ = laplacian_from_coords_torch(
            c_i, k=k_graph, normed=True, device=device, dtype=dtype
        )
        base_diag_blocks.append(scale_csr(Li, lam, device))

    B = torch.as_tensor(pd_B_sub.to_numpy().T, dtype=dtype, device=device)
    X = _X_to_torch_csr(adata_sub, device, dtype)

    G = torch.sparse.mm(X.t(), B)
    H = B.T @ B
    xnorm2 = (X.values() ** 2).sum()

    def objective(V, L):
        data = xnorm2 - 2.0 * (V * G).sum() + ((V @ H) * V).sum()
        return float(data + ((L @ V) * V).sum())

    if V_init is None:
        V_init = []
        for a_i, L_i in zip(adata_group, base_diag_blocks):
            V_i = solve_V_per_slice_torch(
                a_i, pd_B_sub, lam=lam, k=k_graph, eta=eta,
                step_rate=step_rate, L=L_i, device=device,
                dtype=dtype, verbose=verbose,
            )
            if init == "random":
                gen = torch.Generator(device="cpu").manual_seed(0)
                V_i = torch.rand(V_i.shape, generator=gen, dtype=dtype).to(device)
                V_i = V_i / (V_i.sum(dim=1, keepdim=True) + 1e-15)
            V_init.append(V_i)
    else:
        V_init = [
            torch.as_tensor(np.asarray(V_i), dtype=dtype, device=device)
            for V_i in V_init
        ]

    V = torch.cat(V_init, dim=0)

    if one_step:
        outer_max = 1
    if mu < 0:
        outer_max = -1

    L = assemble_blocks_torch(base_diag_blocks, device=device, dtype=dtype)
    f_prev = objective(V, L)
    pi = [None] * (n_times - 1)

    for iter_outer in range(outer_max):
        diag_blocks = list(base_diag_blocks)
        off_diag_blocks = []

        for i in range(n_times - 1):
            n_i, n_ip = adata_group[i].n_obs, adata_group[i + 1].n_obs
            a = torch.full((n_i,), 1.0 / n_i, dtype=dtype, device=device)
            b = torch.full((n_ip,), 1.0 / n_ip, dtype=dtype, device=device)

            if coupling is None:
                C = torch.cdist(V_init[i], V_init[i + 1]).pow_(2)
                C = C / (C.max() + 1e-15)
                pi_i = _compute_coupling(
                    ot_solver, C,
                    coords[i], coords[i + 1],
                    V_init[i], V_init[i + 1],
                    a, b, alpha, verbose, device, dtype,
                )
                del C
            else:
                pi_i = torch.as_tensor(
                    np.asarray(coupling[i]), dtype=dtype, device=device
                )

            # FRLC can return tiny negatives; sparsify rejects them.
            pi_i = pi_i.clamp_min(0.0)

            pi_i = sparsify_pi_torch(pi_i, k_row=int(k / 4), k_col=int(k / 4))
            pi_i = pi_i / (pi_i.sum() + 1e-15)
            pi_i = pi_i * float(np.sqrt(pi_i.shape[0] * pi_i.shape[1]))

            dt = rescaled_time[i + 1] - rescaled_time[i]
            if dt != 0:
                pi_i = pi_i / dt
            pi[i] = pi_i

            a_marg = pi_i.sum(dim=1)
            b_marg = pi_i.sum(dim=0)

            diag_blocks[i] = add_csr(
                diag_blocks[i], diag_csr(mu * a_marg, device)
            )
            diag_blocks[i + 1] = add_csr(
                diag_blocks[i + 1], diag_csr(mu * b_marg, device)
            )

            off_diag_blocks.append((i, i + 1, -mu * pi_i))
            off_diag_blocks.append((i + 1, i, -mu * pi_i.T))

        L = assemble_blocks_torch(
            diag_blocks, offdiag_blocks=off_diag_blocks, device=device, dtype=dtype
        )

        if mu >= 0:
            V = solve_V_universal_torch(
                X, B, L,
                niter_max=niter_max, tol=tol, eta=eta,
                verbose=verbose, V0=V, step_rate=step_rate,
                device=device, dtype=dtype,
            )

            for i in range(n_times):
                V_init[i] = V[n_spots[i]:n_spots[i + 1], :]

        f_curr = objective(V, L)
        rel_drop = (f_prev - f_curr) / max(1.0, abs(f_prev))

        if verbose and (iter_outer % 50 == 0 or rel_drop < tol):
            print(f"iter {iter_outer:4d}  f={f_curr:.6e}  rel_drop={rel_drop:.3e}")

        if f_curr <= f_prev and rel_drop < tol:
            if verbose:
                print("stopped after iteration #", iter_outer)
            break

        f_prev = f_curr

    if return_numpy:
        return (
            [V_i.cpu().numpy() for V_i in V_init],
            [None if p is None else sp.csr_matrix(p.cpu().numpy()) for p in pi],
            csr_to_scipy(L),
        )
    return V_init, pi, L