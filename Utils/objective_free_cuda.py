"""Alternating GNMF across time slices — torch / GPU version.

Entry point: `solve_UV_all_slices_torch`. Mirrors the NumPy module
section-for-section so the two can be diffed against each other.

    U : (d x r)  gene basis, column-normalized
    V : (N x r)  spatial coefficients, graph-regularized
    X : (d x N)  genes x spots

Everything after data loading stays on device except the U step: nonnegative
least squares has no GPU equivalent in torch, so `solve_U_nnls_torch` falls
back to a projected-gradient solve on device (default) or transfers to SciPy's
exact NNLS when `exact_nnls=True`.
"""

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from objective_cuda import (
    add_csr,
    assemble_blocks_torch,
    diag_csr,
    get_device,
    laplacian_from_coords_torch,
    scale_csr,
    solve_V_universal_torch,
    sparsify_pi_torch,
    to_torch_csr,
    _compute_coupling,
    _coords_of,
)


# ─────────────────────────────────────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────────────────────────────────────
def col_normalize(U, eps=1e-15):
    """Scale each nonnegative column to unit sum; returns (U_normalized, scales)."""
    scales = torch.clamp(U.sum(dim=0), min=eps)
    return U / scales[None, :], scales


def init_U_from_svd(X, r, device=None, dtype=torch.float32, verbose=True):
    """Warm-start U from the left singular vectors of X.

    SVD gives the best rank-r approximation, so it is the natural starting
    point before nonnegativity is imposed; signs are discarded since only the
    magnitudes carry over. Uses torch's randomized SVD on a densified X, which
    is fine for the gene dimension but note it materializes (d x N).
    """
    device = get_device(device)
    X_dense = X.to_dense() if X.layout == torch.sparse_csr else X
    P, sigma, _ = torch.svd_lowrank(X_dense, q=min(r + 10, min(X_dense.shape)))
    U, _ = col_normalize(P[:, :r].abs())

    if verbose:
        print(f"  SVD init: singular values "
              f"{np.round(sigma[:r].cpu().numpy(), 3)}")
    return U.to(device=device, dtype=dtype)


# ─────────────────────────────────────────────────────────────────────────────
# U step
# ─────────────────────────────────────────────────────────────────────────────
def solve_U_nnls_torch(
    X, V, dtype=torch.float32, verbose=True,
    activity_threshold=1e-3, min_active_frac=0.01,
    niter=200, tol=1e-6, exact_nnls=False, n_jobs=-1, device=None,
):
    """Nonnegative least squares for U given V, pruning inactive factors.

    With `exact_nnls=False` (default) this runs projected gradient descent on
    device with a Lipschitz step — U >= 0 enforced by clamping. The objective
    is quadratic and the r x r Gram is tiny, so it converges quickly. Set
    `exact_nnls=True` to transfer to SciPy's active-set NNLS instead, which is
    exact but pays a round trip and a Python loop over genes.

    A factor is dead if fewer than `min_active_frac` of spots have a
    V-coefficient above `activity_threshold` times that factor's maximum;
    dead factors are dropped and U refit on the survivors.

    Returns (U column-normalized, the norms divided out, kept column indices) —
    the caller absorbs `scale` into V to leave U V^T unchanged.
    """
    device = get_device(device)

    def fit(V_sub):
        XV = torch.sparse.mm(X, V_sub) if X.layout == torch.sparse_csr else X @ V_sub
        VtV = V_sub.T @ V_sub

        if exact_nnls:
            from scipy.optimize import nnls
            from joblib import Parallel, delayed

            X_host = np.asarray(
                (X.to_dense() if X.layout == torch.sparse_csr else X).cpu(),
                dtype=np.float64,
            )
            V_host = V_sub.cpu().numpy().astype(np.float64)
            rows = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(lambda i: nnls(V_host, X_host[i])[0])(i)
                for i in range(X_host.shape[0])
            )
            return torch.as_tensor(
                np.vstack(rows), device=device, dtype=dtype
            )

        # Projected gradient; step from the Lipschitz constant of the quadratic.
        lip = torch.linalg.eigvalsh(VtV.double())[-1].item()
        step = 1.0 / max(2.0 * lip, 1e-12)

        U = torch.clamp(XV / torch.clamp(VtV.diagonal()[None, :], min=1e-12), min=0)
        for _ in range(niter):
            U_new = torch.clamp(U - step * 2.0 * (U @ VtV - XV), min=0)
            if (U_new - U).abs().max() < tol * max(U.abs().max().item(), 1.0):
                U = U_new
                break
            U = U_new
        return U

    def prune(V_sub):
        col_max = torch.clamp(V_sub.max(dim=0).values, min=1e-15)
        active_frac = (V_sub > activity_threshold * col_max[None, :]).float().mean(0)
        return active_frac >= min_active_frac

    kept = torch.arange(V.shape[1], device=device)
    U = fit(V)
    alive = prune(V)

    # Refit on the surviving basis; the refit can collapse further.
    if not bool(alive.all()):
        if verbose:
            print(f"  pruning {int((~alive).sum())}/{V.shape[1]} dead factors")
        kept = kept[alive]
        U = fit(V[:, kept])
        alive2 = prune(V[:, kept])
        kept, U = kept[alive2], U[:, alive2]
    else:
        U = U[:, alive]

    U, scale = col_normalize(U)
    if verbose:
        print(f"  kept {U.shape[1]} factors")
    return U.to(dtype), scale.to(dtype), kept


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def solve_UV_all_slices_torch(
    adata_group, r,
    lam=10, mu=1, time=None,
    outer_max=5, niter_max=1e3, tol=1e-5, eta=1e-3,
    outer_tol=None, patience=5, normalize_terms=True,
    verbose=True, coupling=None, k=10, k_graph=None,
    V_init=None, U_init=None,
    ot_solver="fgw", one_step=True,
    step_rate="automatic", alpha=0.9,
    dtype=torch.float32, device=None, sparsify=True,
    activity_threshold=1e-2, min_active_frac=0.05,
    exact_nnls=False, n_jobs=-1, return_numpy=True,
):
    """Learn a shared gene basis and per-slice coefficients across time.

    Each outer iteration: compute OT couplings between adjacent slices from
    the current V, rebuild the block Laplacian (per-slice spatial graphs
    weighted by `lam`, cross-slice coupling by `mu`), update V under it, then
    refit U and drop factors no longer used anywhere.

    `r` is an upper bound — pruning can return fewer. `mu < 0` computes the
    couplings and skips both factor updates. `k_graph` sets the spatial kNN
    degree separately from `k`, which drives coupling sparsification.

    Early stopping: an iteration counts as an improvement only if the
    objective falls below the best seen so far by more than `outer_tol` in
    relative terms. After `patience` consecutive non-improving iterations the
    loop stops and the best iterate is restored, so a run that stalls or
    drifts upward never returns a worse state than it reached. An iteration
    that rises does not reset the counter. `outer_tol` defaults to `tol`,
    which is also the inner V-solve tolerance and usually wants a tighter
    value than the outer loop does.

    With `normalize_terms=True`, the objective is proportional to

        data / (N G) + lam * spatial / tr(L_spatial)
                     + mu  * temporal / tr(L_temporal),

    with RMS-scaled expression and unit-mass temporal couplings before 1/dt
    weighting. This removes dependence on the numbers of genes, spots, graph
    edges, coupling size, and the total time span. With `verbose`, each
    iteration reports the weighted spatial and coupling penalties as fractions
    of the data term.

    Returns (U, V_per_slice, pi_list, L).
    """
    device = get_device(device)
    if k_graph is None:
        k_graph = k
    if outer_tol is None:
        outer_tol = tol

    n_times = len(adata_group)
    n_spots = [0]
    for a_i in adata_group:
        n_spots.append(n_spots[-1] + a_i.shape[0])

    common = ad.concat(adata_group, axis=0, join="inner").var_names
    adata_group = [a_i[:, common].copy() for a_i in adata_group]
    genes = adata_group[0].var_names
    d = len(genes)

    if time is None:
        rescaled_time = list(range(n_times))
    else:
        time = np.asarray(time, dtype=float)
        if time[-1] == time[0]:
            rescaled_time = np.zeros(n_times)
        else:
            # Preserve relative time intervals, but make the average interval 1.
            dt_mean = np.mean(np.diff(time))
            rescaled_time = (time - time[0]) / dt_mean

    # Coords stay on device, reused for the spatial graphs and the GW costs.
    coords = [_coords_of(a_i, device, dtype) for a_i in adata_group]

    spatial_blocks = []
    for c_i in coords:
        Li, _ = laplacian_from_coords_torch(
            c_i, k=k_graph, normed=True, device=device, dtype=dtype
        )
        spatial_blocks.append(Li)

    # X: (d x N) on device
    parts = []
    for a_i in adata_group:
        Xi = a_i.layers["counts"].T if "counts" in a_i.layers else a_i.X.T
        parts.append(Xi if sp.issparse(Xi) else sp.csr_matrix(np.asarray(Xi)))
    X = to_torch_csr(sp.hstack(parts, format="csr"), device, dtype=dtype)

    # Put expression on a dimensionless O(1) scale. Since the data-fitting
    # loss is squared Frobenius error, RMS scaling is the natural choice.
    data_scale = float(X.shape[0] * X.shape[1])
    expr_scale = torch.sqrt((X.values().double() ** 2).sum() / data_scale).item()
    expr_scale = max(expr_scale, 1e-15)
    X = scale_csr(X, 1.0 / expr_scale, device)
    xnorm2 = (X.values() ** 2).sum()

    def _csr_trace(A):
        coo = A.to_sparse_coo().coalesce()
        idx = coo.indices()
        return float(coo.values()[idx[0] == idx[1]].double().sum().item())

    spatial_trace = sum(_csr_trace(blk) for blk in spatial_blocks)
    lam_eff = lam * data_scale / max(spatial_trace, 1e-15) if normalize_terms else lam
    base_diag_blocks = [scale_csr(blk, lam_eff, device) for blk in spatial_blocks]

    # Spatial-only operator, held aside so the two penalties can be split.
    # Already carries the effective spatial coefficient, so `spatial` below is
    # the weighted penalty as it enters the objective.
    L_base = assemble_blocks_torch(base_diag_blocks, device=device, dtype=dtype)

    if verbose:
        print(f"expression RMS scale = {expr_scale:.6g}")
        print(f"effective lam={lam_eff:.6g}")

    # -- Initialize U and V ------------------------------------------------
    if U_init is not None:
        U = torch.as_tensor(np.asarray(U_init), device=device, dtype=dtype)
        if tuple(U.shape) != (d, r):
            raise ValueError(f"U_init shape {tuple(U.shape)} != {(d, r)}")
    else:
        U = init_U_from_svd(X, r, device=device, dtype=dtype, verbose=verbose)
    U, _ = col_normalize(U)

    if V_init is not None:
        V_list = [
            torch.as_tensor(np.asarray(V_i), device=device, dtype=dtype)
            for V_i in V_init
        ]
    else:
        # One unregularized V solve against the initial U, per slice.
        X_coo = X.to_sparse_coo()
        V_list = []
        for i in range(n_times):
            cols = X_coo.indices()[1]
            sel = (cols >= n_spots[i]) & (cols < n_spots[i + 1])
            idx = X_coo.indices()[:, sel].clone()
            idx[1] -= n_spots[i]
            Xi = torch.sparse_coo_tensor(
                idx, X_coo.values()[sel],
                size=(d, n_spots[i + 1] - n_spots[i]), device=device,
            ).coalesce().to_sparse_csr()

            V_list.append(
                solve_V_universal_torch(
                    Xi, U, base_diag_blocks[i],
                    niter_max=niter_max, tol=tol, eta=eta,
                    verbose=False, step_rate=step_rate,
                    device=device, dtype=dtype,
                )
            )
    V = torch.cat(V_list, dim=0)

    def objective_terms(V, U, L):
        """(data, spatial, coupling). data = ||X - U V^T||^2, which retains
        the constant xnorm2, so the printed fractions are against the total
        rather than the explained part.

        Reductions run in float64. The three terms of the expansion are each
        ~xnorm2 in magnitude and nearly cancel, so accumulating them in
        float32 leaves an error comparable to the per-iteration change and
        the sign of `improvement` becomes noise. The sparse matmuls stay in
        float32 (float64 sparse on CUDA is slow); only the sums are widened.
        """
        Vd = V.double()
        G = torch.sparse.mm(X.t(), U).double()
        H = (U.T @ U).double()
        data = float(xnorm2.double() - 2.0 * (Vd * G).sum() + ((Vd @ H) * Vd).sum())
        total_pen = float(((L @ V).double() * Vd).sum())
        spatial = float(((L_base @ V).double() * Vd).sum())
        return data, spatial, total_pen - spatial

    def objective(V, U, L):
        data, spatial, coupling_pen = objective_terms(V, U, L)
        return data + spatial + coupling_pen

    L = L_base

    f_best = float("inf")
    n_stall = 0
    best = None
    pi = [None] * (n_times - 1)

    if one_step:
        outer_max = 1
    if mu < 0:
        # One pass: build the couplings, skip both factor updates.
        # (outer_max = -1 made range() empty, so nothing ran at all.)
        outer_max = 1

    # For a unit-mass coupling, each pairwise temporal Laplacian has trace 2.
    # The 1/dt weighting scales that trace by 1/dt.
    temporal_trace = 0.0
    for i in range(n_times - 1):
        dt = rescaled_time[i + 1] - rescaled_time[i]
        temporal_trace += 2.0 / dt if dt != 0 else 2.0
    mu_eff = mu * data_scale / max(temporal_trace, 1e-15) if normalize_terms else mu
    if verbose:
        print(f"effective mu={mu_eff:.6g}")

    for iter_outer in range(outer_max):
        # -- Couplings and block Laplacian ---------------------------------
        # Clone, not list(): a shallow copy protects the list bindings but
        # not the tensors, so an in-place add_csr would accumulate mu terms
        # into base_diag_blocks and silently corrupt L_base as well.
        diag_blocks = [blk.clone() for blk in base_diag_blocks]
        off_diag_blocks = []

        for i in range(n_times - 1):
            n_i, n_ip = adata_group[i].n_obs, adata_group[i + 1].n_obs
            a = torch.full((n_i,), 1.0 / n_i, dtype=dtype, device=device)
            b = torch.full((n_ip,), 1.0 / n_ip, dtype=dtype, device=device)

            if coupling is None:
                C = torch.cdist(V_list[i], V_list[i + 1]).pow_(2)
                C = C / (C.max() + 1e-15)
                pi_i = _compute_coupling(
                    ot_solver, C,
                    coords[i], coords[i + 1],
                    V_list[i], V_list[i + 1],
                    a, b, alpha, verbose, device, dtype,
                )
                del C
            else:
                pi_i = torch.as_tensor(
                    np.asarray(coupling[i]), device=device, dtype=dtype
                )

            # FRLC can return tiny negatives; sparsify rejects them.
            pi_i = pi_i.clamp_min(0.0)
            pi_i = pi_i / (pi_i.sum() + 1e-15)
            if not normalize_terms:
                pi_i = pi_i * float(np.sqrt(pi_i.shape[0] * pi_i.shape[1]))

            dt = rescaled_time[i + 1] - rescaled_time[i]
            if dt != 0:
                pi_i = pi_i / dt
            if sparsify:
                pi_i = sparsify_pi_torch(pi_i, k_row=k, k_col=k)
            pi[i] = pi_i

            diag_blocks[i] = add_csr(
                diag_blocks[i], diag_csr(mu_eff * pi_i.sum(dim=1), device)
            )
            diag_blocks[i + 1] = add_csr(
                diag_blocks[i + 1], diag_csr(mu_eff * pi_i.sum(dim=0), device)
            )

            block = -mu_eff * pi_i
            off_diag_blocks.append((i, i + 1, block))
            off_diag_blocks.append((i + 1, i, block.T))

        L = assemble_blocks_torch(
            diag_blocks, offdiag_blocks=off_diag_blocks, device=device, dtype=dtype
        )
        if mu < 0:
            continue

        # -- V step: graph-regularized, U fixed ----------------------------
        V = solve_V_universal_torch(
            X, U, L,
            niter_max=niter_max, tol=tol, eta=eta,
            verbose=verbose, V0=V, step_rate=step_rate,
            device=device, dtype=dtype,
        )

        # -- U step: NNLS, then prune and absorb the scale into V ----------
        U, scale, kept = solve_U_nnls_torch(
            X, V, dtype=dtype, verbose=verbose,
            activity_threshold=activity_threshold,
            min_active_frac=min_active_frac,
            exact_nnls=exact_nnls, n_jobs=n_jobs, device=device,
        )
        V = V[:, kept] * scale[None, :]
        r = U.shape[1]

        for i in range(n_times):
            V_list[i] = V[n_spots[i]:n_spots[i + 1], :]

        # -- Term balance and early stopping -------------------------------
        data, spatial, coupling_pen = objective_terms(V, U, L)
        f_curr = data + spatial + coupling_pen

        improvement = ((f_best - f_curr) / max(1.0, abs(f_best))
                       if np.isfinite(f_best) else float("inf"))

        if improvement > outer_tol:
            f_best = f_curr
            n_stall = 0
            best = (U.clone(), V.clone(),
                    [V_i.clone() for V_i in V_list],
                    [None if p_i is None else p_i.clone() for p_i in pi],
                    L)
        else:
            n_stall += 1

        if verbose:
            print(f"outer {iter_outer:4d}  f={f_curr:.6e}  "
                  f"improvement={improvement:.3e}/{outer_tol:.1e}  r={r}  "
                  f"stall={n_stall}/{patience}")
            print(f"           data={data:.4e}  "
                  f"spatial={spatial:.4e} ({spatial / data:.2%})  "
                  f"coupling={coupling_pen:.4e} ({coupling_pen / data:.2%})")

        if n_stall >= patience:
            if verbose:
                print(f"early stop at outer {iter_outer}: no improvement in "
                      f"{patience} iterations (best f={f_best:.6e})")
            break

    # The last iterate is not necessarily the best one: L moves every
    # iteration and pruning changes r discontinuously, so the objective can
    # rise. Return the best.
    if best is not None:
        U, V, V_list, pi, L = best

    if return_numpy:
        def _pi_to_scipy(p):
            if p is None:
                return None
            p = p.to_dense() if p.layout != torch.strided else p
            return sp.csr_matrix(p.cpu().numpy())

        return (
            U.cpu().numpy(),
            [V_i.cpu().numpy() for V_i in V_list],
            [_pi_to_scipy(p) for p in pi],
            L,
        )
    return U, V_list, pi, L


def U_to_dataframe(U, gene_names):
    """Wrap U (d x r) as a factors x genes frame, the pd_B layout."""
    U = U.cpu().numpy() if torch.is_tensor(U) else np.asarray(U)
    return pd.DataFrame(U.T, columns=gene_names)