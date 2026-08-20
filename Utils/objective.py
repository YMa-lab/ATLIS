import anndata as ad
import numpy as np
import scipy.sparse as sp
import scipy.spatial
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from sklearn.neighbors import NearestNeighbors
import re
_MONTHS = "(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
_DATE_LIKE_RE = re.compile(rf"^\d{{1,2}}-{_MONTHS}$")



# ─────────────────────────────────────────────────────────────────────────────
# Coupling sparsification (for large datasets)
# ─────────────────────────────────────────────────────────────────────────────
def _topk_along_indptr(A, k, n_slices, matrix_cls, shape):
    """Keep the k largest values in each slice delimited by A.indptr.

    Shared body for row-wise (CSR) and column-wise (CSC) top-k, since the two
    are the same operation on different orientations.
    """
    indptr, indices, data = A.indptr, A.indices, A.data

    new_indptr = np.empty(n_slices + 1, dtype=indptr.dtype)
    new_indptr[0] = 0
    idx_parts, data_parts = [], []
    nnz_out = 0

    for s in range(n_slices):
        lo, hi = indptr[s], indptr[s + 1]
        nnz = hi - lo
        if nnz == 0:
            new_indptr[s + 1] = nnz_out
            continue

        if nnz <= k:
            idx_parts.append(indices[lo:hi])
            data_parts.append(data[lo:hi])
            nnz_out += nnz
        else:
            seg = data[lo:hi]
            sel = np.argpartition(seg, -k)[-k:]
            sel.sort()
            idx_parts.append(indices[lo:hi][sel])
            data_parts.append(seg[sel])
            nnz_out += k

        new_indptr[s + 1] = nnz_out

    new_indices = (
        np.concatenate(idx_parts) if idx_parts
        else np.array([], dtype=indices.dtype)
    )
    new_data = (
        np.concatenate(data_parts) if data_parts
        else np.array([], dtype=data.dtype)
    )

    out = matrix_cls((new_data, new_indices, new_indptr), shape=shape)
    out.eliminate_zeros()
    return out


def topk_per_row_csr(A_csr, k):
    A = A_csr.tocsr(copy=True)
    return _topk_along_indptr(A, k, A.shape[0], sp.csr_matrix, A.shape)


def topk_per_col_csc(A_csc, k):
    A = A_csc.tocsc(copy=True)
    return _topk_along_indptr(A, k, A.shape[1], sp.csc_matrix, A.shape)


def sinkhorn_scale_sparse(A_csr, row_target, col_target, n_iter=50, eps=1e-12):
    """Rescale a nonnegative sparse matrix toward the given marginals."""
    A = A_csr.tocsr(copy=True)
    AT = A.T.tocsr()
    row_target = np.asarray(row_target, dtype=np.float64)
    col_target = np.asarray(col_target, dtype=np.float64)

    c = np.ones(A.shape[1], dtype=np.float64)
    for _ in range(n_iter):
        r = row_target / np.maximum(np.asarray(A.dot(c)).ravel(), eps)
        c = col_target / np.maximum(np.asarray(AT.dot(r)).ravel(), eps)

    A = sp.diags(r).dot(A).dot(sp.diags(c))
    A.eliminate_zeros()
    return A


def sparsify_pi_sparse(pi, k_row=None, k_col=None, mode="doubly", n_iter=50):
    """Prune a coupling to its top-k entries per row/column, then rescale.

    The pruned support is the union of the row-wise and column-wise top-k, so
    no row or column is emptied unless k is smaller than the support allows.
    `mode` controls which marginals are restored afterwards.
    """
    A = pi.tocsr() if sp.issparse(pi) else sp.csr_matrix(np.asarray(pi))
    if A.nnz == 0:
        return A
    if (A.data < 0).any():
        raise ValueError("pi must be nonnegative.")

    A.sum_duplicates()
    A.eliminate_zeros()
    A.sort_indices()

    row_target = np.asarray(A.sum(axis=1)).ravel()
    col_target = np.asarray(A.sum(axis=0)).ravel()

    A_row = topk_per_row_csr(A, k_row) if k_row is not None else A.tocsr(copy=True)
    A_col = (
        topk_per_col_csc(A.tocsc(), k_col).tocsr() if k_col is not None
        else A.tocsr(copy=True)
    )
    A_pruned = A_row.maximum(A_col)

    if np.any(np.diff(A_pruned.indptr) == 0):
        raise ValueError("Empty row after sparsification. Increase k_row/k_col.")
    if np.any(np.diff(A_pruned.tocsc().indptr) == 0):
        raise ValueError("Empty col after sparsification. Increase k_row/k_col.")

    mode = mode.lower()
    if mode == "row":
        rs = np.asarray(A_pruned.sum(axis=1)).ravel()
        return sp.diags(row_target / np.maximum(rs, 1e-12)).dot(A_pruned)
    if mode == "col":
        cs = np.asarray(A_pruned.sum(axis=0)).ravel()
        return A_pruned.dot(sp.diags(col_target / np.maximum(cs, 1e-12)))
    if mode == "doubly":
        return sinkhorn_scale_sparse(A_pruned, row_target, col_target, n_iter=n_iter)
    raise ValueError("mode must be 'row', 'col', or 'doubly'.")


# ─────────────────────────────────────────────────────────────────────────────
# Graph structure
# ─────────────────────────────────────────────────────────────────────────────
def assemble_blocks(diag_blocks, offdiag_blocks=()):
    """Build a block matrix from diagonal blocks plus (i, j, B_ij) entries."""
    r_sizes = [b.shape[0] for b in diag_blocks]
    c_sizes = [b.shape[1] for b in diag_blocks]

    A = lil_matrix((sum(r_sizes), sum(c_sizes)))
    r_off = np.r_[0, np.cumsum(r_sizes[:-1])]
    c_off = np.r_[0, np.cumsum(c_sizes[:-1])]

    for i, Bii in enumerate(diag_blocks):
        ri, ci = Bii.shape
        A[r_off[i]:r_off[i] + ri, c_off[i]:c_off[i] + ci] = Bii

    for i, j, Bij in offdiag_blocks:
        ri, cj = r_sizes[i], c_sizes[j]
        if Bij.shape != (ri, cj):
            raise ValueError(
                f"Offdiag block ({i},{j}) must be shape {(ri, cj)}, got {Bij.shape}"
            )
        A[r_off[i]:r_off[i] + ri, c_off[j]:c_off[j] + cj] = Bij

    return A.tocsr()


def laplacian_from_coords(x, y, k=20, mutual=False, normed=False, return_dense=False):
    """Unweighted k-NN graph Laplacian from 2D coordinates.

    `mutual=True` keeps edge i–j only when each is in the other's kNN;
    otherwise the directed graph is symmetrized by union.

    Returns (L, W).
    """
    coords = np.column_stack([x, y])
    n = coords.shape[0]
    k = min(k, n - 1)

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(coords)
    nbrs = nn.kneighbors(coords)[1][:, 1:]          # drop self

    row = np.repeat(np.arange(n), k)
    col = nbrs.ravel()
    W_dir = csr_matrix((np.ones_like(col, dtype=float), (row, col)), shape=(n, n))
    W = W_dir.minimum(W_dir.T) if mutual else W_dir.maximum(W_dir.T)

    L = csgraph_laplacian(W, normed=normed)
    return (L.toarray(), W.toarray()) if return_dense else (L, W)


def proj_simplex_rows(V, z=1.0, proj=False):
    """Project each row onto {x >= 0, sum(x) = z}; with proj=False, clip at 0."""
    V = np.asarray(V, dtype=float)
    n, m = V.shape

    if proj:
        U = -np.sort(-V, axis=1)
        cssv = np.cumsum(U, axis=1)
        j = np.arange(1, m + 1)
        rho = ((U * j) > (cssv - z)).sum(axis=1) - 1
        theta = (cssv[np.arange(n), rho] - z) / (rho + 1.0)
        X = V - theta[:, None]
    else:
        X = V

    np.maximum(X, 0.0, out=X)
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Inner solve: min_V ||X - B V^T||_F^2 + tr(V^T L V)
# ─────────────────────────────────────────────────────────────────────────────
def solve_V_universal(
    X, B, L,
    niter_max=1e3, tol=1e-5, eta=1e-3, verbose=True,
    V0=None, return_obj=False, proj_simplex=False, step_rate="backtrack",
):
    """Projected gradient descent on V, never forming the residual X - B V^T.

    The objective is expanded as
        ||X||_F^2 - 2 tr(V^T G) + tr(V^T V H) + tr(V^T L V),
    with G = X^T B and H = B^T B precomputed, so the sparse X is touched once.

    Returns V, or (V, reg_term, data_term) when `return_obj`.
    """
    X = X.tocsr() if sp.issparse(X) else np.asarray(X)
    B = B.tocsr() if sp.issparse(B) else np.asarray(B)
    L = L.tocsr() if sp.issparse(L) else np.asarray(L)

    d, n = X.shape
    dB, r = B.shape
    assert d == dB, f"X is {X.shape}, B is {B.shape} (dim mismatch)"

    G = X.T @ B
    H = B.T @ B
    if sp.issparse(G):
        G = G.toarray()
    if sp.issparse(H):
        H = H.toarray()

    xnorm2 = float(X.multiply(X).sum()) if sp.issparse(X) else float(np.sum(X * X))

    def objective(V):
        data = xnorm2 - 2.0 * np.sum(V * G) + np.sum((V @ H) * V)
        return data + np.sum((L @ V) * V)

    def gradient(V):
        return 2.0 * (V @ H - G) + 2.0 * (L @ V)

    if V0 is None:
        V = np.random.default_rng(0).random((n, r), dtype=float)
    else:
        V = np.asarray(V0, dtype=float)
        assert V.shape == (n, r)

    f_prev = objective(V)

    for it in range(int(niter_max)):
        g = gradient(V)
        V_old = V

        if step_rate == "backtrack":
            step = float(eta)
            gnorm2 = float(np.sum(g * g))
            while True:
                V_trial = proj_simplex_rows(V_old - step * g, z=1.0, proj=proj_simplex)
                f_trial = objective(V_trial)
                # Armijo condition, or give up once the step is negligible.
                if f_trial <= f_prev - 1e-4 * step * gnorm2 or step < 1e-8:
                    V, f_curr = V_trial, f_trial
                    break
                step *= 0.5

        elif step_rate == "automatic":
            # Multiplicative-style step; assumes L is diagonally dominant.
            dL = L.diagonal() if sp.issparse(L) else np.diag(L)
            step_mat = V_old / (2.0 * (V_old @ H + dL[:, None] * V_old + 1e-15))
            V = V_old - step_mat * g
            f_curr = objective(V)
            step = float(np.mean(step_mat))

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
    data_term = xnorm2 - 2.0 * np.sum(V * G) + np.sum((V @ H) * V)
    return V, float(np.sum((L @ V) * V)), data_term


def solve_V_per_slice(
    adata, pd_B, lam=10, k=20, niter_max=1e3, tol=1e-5, eta=1e-3,
    verbose=True, return_obj=False, proj_simplex=False, step_rate="backtrack",
):
    """Deconvolve a single slice with only its own spatial Laplacian."""
    common = adata.var_names.intersection(pd_B.columns)
    adata_sub = adata[:, common].copy()

    L, _ = laplacian_from_coords(
        adata_sub.obs["x"].to_numpy(),
        adata_sub.obs["y"].to_numpy(),
        k=k, normed=True,
    )
    X = adata_sub.layers["counts"].T if "counts" in adata_sub.layers else adata_sub.X.T

    return solve_V_universal(
        X, pd_B[common].to_numpy().T, lam * L,
        niter_max=niter_max, tol=tol, eta=eta, verbose=verbose,
        return_obj=return_obj, proj_simplex=proj_simplex, step_rate=step_rate,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Couplings between consecutive slices
# ─────────────────────────────────────────────────────────────────────────────
def _spatial_cost(adata_slice):
    import ot
    coords = np.vstack((
        adata_slice.obs["x"].to_numpy(),
        adata_slice.obs["y"].to_numpy(),
    )).T
    C = ot.dist(coords)
    return C / (C.max() + 1e-15)


def _frlc_coupling(C, C1=None, C2=None, alpha=0.9, inner_iters=300, **kwargs):
    """Low-rank OT via FRLC. C1/C2 present -> fused GW; absent -> Wasserstein."""
    import torch
    import sys
    sys.path.insert(0, '../../FRLC/src/')
    import FRLC


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        args.update(
            A=torch.tensor(C1, device=device),
            B=torch.tensor(C2, device=device),
            alpha=alpha, Wasserstein=False, FGW=True,
        )
    else:
        args.update(Wasserstein=True, FGW=False)
    args.update(kwargs)

    pi, _ = FRLC.FRLC_opt(torch.tensor(C, device=device), **args)
    return pi.detach().cpu().numpy()


def _compute_coupling(
    ot_solver, C, adata_i, adata_j, V_i, V_j, a, b, alpha, verbose,
):
    """Dispatch to the requested OT backend and return a dense coupling."""
    if ot_solver == "fgw":
        import ot
        from ot.gromov import fused_gromov_wasserstein

        C1, C2 = _spatial_cost(adata_i), _spatial_cost(adata_j)
        pi = fused_gromov_wasserstein(
            C, C1, C2, a, b, loss_fun="square_loss", alpha=alpha, log=True,
        )[0]

        # A near-constant plan means the solver stalled; retry low-rank.
        if np.std(pi.ravel()) < 1e-10:
            if verbose:
                print("constant coupling... fall back", C.shape)
            pi = _frlc_coupling(C, C1, C2, alpha=alpha)
        return pi

    if ot_solver == "sinkhorn":
        from ott.geometry.pointcloud import PointCloud
        from ott.problems.linear.linear_problem import LinearProblem
        from ott.solvers.linear.sinkhorn import Sinkhorn

        denom = max(np.abs(V_i).max(), np.abs(V_j).max(), 1e-15)
        problem = LinearProblem(PointCloud(V_i / denom, V_j / denom, epsilon=0.01))
        return np.array(Sinkhorn()(problem).matrix)

    if ot_solver == "sinkhorn_lr":
        return _frlc_coupling(C)

    if ot_solver in {"low_rank", "low_rank_SR"}:
        extra = (
            dict(semiRelaxedLeft=True, semiRelaxedRight=True)
            if ot_solver == "low_rank_SR" else {}
        )
        return _frlc_coupling(
            C, _spatial_cost(adata_i), _spatial_cost(adata_j),
            alpha=alpha, inner_iters=100, **extra,
        )

    raise ValueError(f"Unknown ot_solver: {ot_solver}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def solve_V_all_slices(
    adata_group, pd_B,
    lam=10, mu=1, time=None,
    outer_max=5, niter_max=1e3, tol=1e-5, eta=1e-3,
    verbose=True, coupling=None, k=10, init="per_slice",
    V_init=None, ot_solver="fgw", one_step=True,
    step_rate="automatic", alpha=0.5,
    dtype=np.float32,
    sparsify=False
):
    """Learns cell type program scores across all slices jointly, coupling consecutive time points via OT.

    Alternates between (i) computing couplings pi_t between adjacent slices and
    (ii) solving for V under a block Laplacian that combines each slice's
    spatial graph with the coupling-induced cross-slice penalty, weighted by mu.
    `mu < 0` skips the V update entirely and only returns the couplings.

    V is dense by design; the couplings are sparsified after solving, but the
    OT solvers themselves still build dense cost matrices.

    Returns (V_per_slice, pi_list, L).
    """
    rng = np.random.default_rng(0)
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

    base_diag_blocks = []
    for a_i in adata_group:
        Li, _ = laplacian_from_coords(
            a_i.obs["x"].to_numpy(), a_i.obs["y"].to_numpy(), k=k, normed=True
        )
        if not sp.issparse(Li):
            Li = sp.csr_matrix(Li)
        base_diag_blocks.append((lam * Li).tocsr())

    B = np.asarray(pd_B_sub.to_numpy().T, dtype=dtype)

    X = adata_sub.X.T
    X = (
        X.tocsr().astype(dtype) if sp.issparse(X)
        else sp.csr_matrix(np.asarray(X, dtype=dtype))
    )

    G = X.T @ B
    if sp.issparse(G):
        G = G.toarray()
    G = np.asarray(G, dtype=dtype)

    H = (B.T @ B).astype(dtype, copy=False)
    xnorm2 = float(X.multiply(X).sum())

    def objective(V, L):
        data = xnorm2 - 2.0 * float(np.sum(V * G)) + float(np.sum((V @ H) * V))
        return data + float(np.sum((L @ V) * V))

    if V_init is None:
        V_init = []
        for a_i in adata_group:
            V_i = np.asarray(
                solve_V_per_slice(
                    a_i, pd_B_sub, lam=lam, k=k, eta=eta, step_rate=step_rate
                ),
                dtype=dtype,
            )
            if init == "random":
                V_i = rng.random(V_i.shape, dtype=float).astype(dtype)
                V_i = V_i / (V_i.sum(axis=1, keepdims=True) + 1e-15)
            V_init.append(V_i)
    else:
        V_init = [np.asarray(V_i, dtype=dtype) for V_i in V_init]

    V = np.vstack(V_init).astype(dtype, copy=False)

    if one_step:
        outer_max = 1
    if mu < 0:
        outer_max = -1

    L = assemble_blocks([blk.copy() for blk in base_diag_blocks]).tocsr()
    f_prev = objective(V, L)
    pi = [None] * (n_times - 1)

    for iter_outer in range(outer_max):
        diag_blocks = [blk.copy() for blk in base_diag_blocks]
        off_diag_blocks = []

        for i in range(n_times - 1):
            n_i, n_ip = adata_group[i].n_obs, adata_group[i + 1].n_obs
            a = np.full(n_i, 1.0 / n_i, dtype=dtype)
            b = np.full(n_ip, 1.0 / n_ip, dtype=dtype)

            if coupling is None:
                C = scipy.spatial.distance.cdist(
                    V_init[i].astype(np.float64, copy=False),
                    V_init[i + 1].astype(np.float64, copy=False),
                    metric="sqeuclidean",
                )
                C = C / (C.max() + 1e-15)
                pi_i = _compute_coupling(
                    ot_solver, C,
                    adata_group[i], adata_group[i + 1],
                    V_init[i], V_init[i + 1],
                    a, b, alpha, verbose,
                )
            else:
                pi_i = np.asarray(coupling[i], dtype=dtype)

            if sparsify:
                pi_i = sparsify_pi_sparse(pi_i, k_row=int(k / 4), k_col=int(k / 4))
            pi_i = pi_i / (pi_i.sum() + 1e-15)
            pi_i = pi_i * np.sqrt(pi_i.shape[0] * pi_i.shape[1])

            dt = rescaled_time[i + 1] - rescaled_time[i]
            if dt != 0:
                pi_i = pi_i / dt
            pi[i] = pi_i

            a_marg = np.asarray(pi_i.sum(axis=1)).ravel()
            b_marg = np.asarray(pi_i.sum(axis=0)).ravel()

            diag_blocks[i] = diag_blocks[i] + mu * sp.diags(a_marg, format="csr")
            diag_blocks[i + 1] = diag_blocks[i + 1] + mu * sp.diags(b_marg, format="csr")

            off_diag_blocks.append((i, i + 1, -mu * pi_i))
            off_diag_blocks.append((i + 1, i, -mu * pi_i.T))

        L = assemble_blocks(diag_blocks, offdiag_blocks=off_diag_blocks).tocsr()

        if mu >= 0:
            V = solve_V_universal(
                X, B, L,
                niter_max=niter_max, tol=tol, eta=eta,
                verbose=verbose, V0=V, step_rate=step_rate,
            ).astype(dtype, copy=False)

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

    return V_init, pi, L

