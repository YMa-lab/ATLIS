"""Alternating GNMF across time slices: X ≈ U V^T with no reference profiles.

Entry point: `solve_UV_all_slices`. Unlike `solve_V_all_slices`, the gene basis
U is learned rather than supplied, so the factors are discovered programs
instead of known cell types.

    U : (d x r)  gene basis, each column normalized to mean 1
    V : (N x r)  spatial coefficients, graph-regularized
    X : (d x N)  genes x spots

The graph regularizer tr(V^T L V) touches V only, so the U step is a plain
nonnegative least squares.
"""

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.spatial

from objective import (
    assemble_blocks,
    laplacian_from_coords,
    solve_V_universal,
    sparsify_pi_sparse,
    _compute_coupling,
)


def init_U_from_svd(X, r, dtype=np.float32, verbose=True, random_state=0):
    """Warm-start U from the left singular vectors of X.

    SVD gives the best rank-r approximation, so it is the natural starting
    point before nonnegativity is imposed; signs are discarded since only the
    magnitudes carry over.
    """
    from sklearn.utils.extmath import randomized_svd

    P, sigma, _ = randomized_svd(X, n_components=r, random_state=random_state)
    U = np.abs(P)
    U = U / np.maximum(U.mean(axis=0), 1e-15)[None, :]

    if verbose:
        print(f"  SVD init: singular values {np.round(sigma, 3)}")
    return U.astype(dtype)


def solve_U_nnls(
    X, V, n_jobs=-1, dtype=np.float32, verbose=True,
    activity_threshold=1e-3, min_active_frac=0.01,
):
    """Nonnegative least squares for U given V, pruning inactive factors.

    A factor is dead if fewer than `min_active_frac` of spots have a
    V-coefficient above `activity_threshold` times that factor's maximum;
    dead factors are dropped and U refit on the survivors.

    Returns (U with each column normalized to mean 1, the scales that were
    divided out, kept column indices) — the caller absorbs `scale` into V to
    leave U V^T fixed.
    """
    from scipy.optimize import nnls
    from joblib import Parallel, delayed

    X = np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    d = X.shape[0]

    def fit_and_prune(V_sub, col_indices):
        if verbose:
            print(f"  NNLS: {d} rows, r={V_sub.shape[1]}")
        rows = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(lambda i: nnls(V_sub, X[i])[0])(i) for i in range(d)
        )
        U_sub = np.vstack(rows)

        col_max = np.maximum(V_sub.max(axis=0), 1e-15)
        active_frac = (V_sub > activity_threshold * col_max[None, :]).mean(axis=0)
        alive = active_frac >= min_active_frac

        if verbose and not alive.all():
            print(f"  pruning {(~alive).sum()}/{V_sub.shape[1]} dead factors")
        return U_sub, alive, col_indices

    U, alive, cols = fit_and_prune(V, np.arange(V.shape[1]))
    kept = cols[alive]

    # Refit on the surviving basis; the refit can collapse further.
    if not alive.all():
        U, alive2, kept = fit_and_prune(V[:, kept], kept)
        kept, U = kept[alive2], U[:, alive2]
    else:
        U = U[:, alive]

    scale = np.maximum(U.mean(axis=0), 1e-15)
    U = U / scale[None, :]

    if verbose:
        print(f"  kept {U.shape[1]} factors")
    return U.astype(dtype), scale.astype(dtype), kept


def solve_UV_all_slices(
    adata_group, r,
    lam=None, mu=None, time=None,
    outer_max=5, niter_max=1e3, tol=1e-2, eta=1e-2,
    outer_tol=None, patience=5,
    target_spatial_ratio=0.01, target_temporal_ratio=0.01,
    verbose=True, coupling=None, k=10,
    V_init=None, U_init=None,
    ot_solver="fgw", one_step=False,
    step_rate="automatic", alpha=0.5,
    dtype=np.float32, sparsify=False,
    n_jobs=-1, activity_threshold=1e-2, min_active_frac=0.05,
    random_state=0, normalize_terms=True, auto_balance=None,
    return_ratio=False,
):
    """Learn a shared gene basis and per-slice coefficients across time.

    ``lam`` and ``mu`` are each independently either fixed (a number, used
    directly) or ``None`` (auto-chosen once so the penalty is
    ``target_spatial_ratio`` / ``target_temporal_ratio`` of the data term).
    E.g. lam=<tuned value>, mu=None fixes lambda and auto-selects mu from the
    coupling built on the lambda-regularized V.

    mu < 0 (with lam given) is a spatial-only sentinel: no OT coupling is
    built, and U/V alternate on the spatial Laplacian alone.

    Returns (U, V_per_slice, pi_list, L), or with ``return_ratio=True``
    (U, V_per_slice, pi_list, L, data_fitting, spatial_ratio, temporal_ratio),
    all computed from the returned state.
    """
    # lam and mu are each independently auto (None -> target ratio) or fixed.
    if auto_balance is False and (lam is None or mu is None):
        raise ValueError("lam and mu must be provided when auto_balance=False.")
    auto_lam = lam is None
    auto_mu = mu is None

    spatial_only = (not auto_mu) and mu < 0

    if outer_tol is None:
        outer_tol = tol

    n_times = len(adata_group)
    n_spots = [0]
    for a_i in adata_group:
        n_spots.append(n_spots[-1] + a_i.shape[0])

    common_genes = ad.concat(adata_group, axis=0, join="inner").var_names
    adata_group = [a_i[:, common_genes].copy() for a_i in adata_group]
    genes = adata_group[0].var_names
    d = len(genes)

    if time is None:
        rescaled_time = list(range(n_times))
    else:
        time = np.asarray(time, dtype=float)
        if time[-1] == time[0]:
            rescaled_time = np.zeros(n_times)
        else:
            dt_mean = np.mean(np.diff(time))
            rescaled_time = (time - time[0]) / dt_mean

    spatial_blocks = []
    for a_i in adata_group:
        Li, _ = laplacian_from_coords(
            a_i.obs["x"].to_numpy(), a_i.obs["y"].to_numpy(), k=k, normed=True
        )
        if not sp.issparse(Li):
            Li = sp.csr_matrix(Li)
        spatial_blocks.append(Li.tocsr())

    parts = []
    for a_i in adata_group:
        Xi = a_i.X.T
        parts.append(
            Xi.tocsr().astype(dtype) if sp.issparse(Xi)
            else sp.csr_matrix(np.asarray(Xi, dtype=dtype))
        )
    X = sp.hstack(parts, format="csr")

    expr_scale = np.sqrt(float(X.multiply(X).sum()) / (X.shape[0] * X.shape[1]))
    expr_scale = max(expr_scale, 1e-15)
    X = X / expr_scale
    xnorm2 = float(X.multiply(X).sum())
    data_scale = float(X.shape[0] * X.shape[1])
    spatial_trace = float(sum(blk.diagonal().sum() for blk in spatial_blocks))

    if verbose:
        print(f"expression RMS scale = {expr_scale:.6g}")

    # -- Initialize U --------------------------------------------------------
    if U_init is not None:
        U = np.asarray(U_init, dtype=dtype).copy()
        if U.shape != (d, r):
            raise ValueError(f"U_init shape {U.shape} != {(d, r)}")
    else:
        U = init_U_from_svd(
            X, r, dtype=dtype, verbose=verbose, random_state=random_state
        )
    U = U / np.maximum(U.mean(axis=0), 1e-15)[None, :]

    def data_term(V, U):
        G = X.T @ U
        if sp.issparse(G):
            G = G.toarray()
        H = U.T @ U
        return xnorm2 - 2.0 * float(np.sum(V * G)) + float(np.sum((V @ H) * V))

    L_spatial_raw = assemble_blocks([blk.copy() for blk in spatial_blocks]).tocsr()

    # -- Initial V -----------------------------------------------------------
    if V_init is not None:
        V_list = [np.asarray(V_i, dtype=dtype) for V_i in V_init]
        V = np.vstack(V_list).astype(dtype, copy=False)
    else:
        V_list = []
        if auto_lam:
            init_blocks = [sp.csr_matrix(Li.shape, dtype=dtype) for Li in spatial_blocks]
        else:
            lam_eff = (
                lam * data_scale / max(spatial_trace, 1e-15)
                if normalize_terms else lam
            )
            init_blocks = [(lam_eff * Li).tocsr() for Li in spatial_blocks]

        for i in range(n_times):
            Xi = X[:, n_spots[i]:n_spots[i + 1]]
            V_list.append(
                solve_V_universal(
                    Xi, U, init_blocks[i],
                    niter_max=niter_max, tol=tol, eta=eta,
                    verbose=False, step_rate=step_rate,
                    random_state=random_state,
                ).astype(dtype)
            )
        V = np.vstack(V_list).astype(dtype, copy=False)

    # -- Choose lambda once, then spatially regularize V --------------------
    if auto_lam:
        D0 = data_term(V, U)
        S0 = float(np.sum((L_spatial_raw @ V) * V))
        lam_eff = target_spatial_ratio * D0 / max(S0, 1e-30)

        spatial_blocks_weighted = [(lam_eff * Li).tocsr() for Li in spatial_blocks]
        V_list = []
        for i in range(n_times):
            Xi = X[:, n_spots[i]:n_spots[i + 1]]
            V_i0 = V[n_spots[i]:n_spots[i + 1], :]
            V_list.append(
                solve_V_universal(
                    Xi, U, spatial_blocks_weighted[i],
                    niter_max=niter_max, tol=tol, eta=eta,
                    verbose=False, V0=V_i0, step_rate=step_rate,
                    random_state=random_state,
                ).astype(dtype)
            )
        V = np.vstack(V_list).astype(dtype, copy=False)
    else:
        lam_eff = (
            lam * data_scale / max(spatial_trace, 1e-15)
            if normalize_terms else lam
        )
        spatial_blocks_weighted = [(lam_eff * Li).tocsr() for Li in spatial_blocks]

    base_diag_blocks = spatial_blocks_weighted
    L_base = assemble_blocks(base_diag_blocks).tocsr()

    if verbose:
        print(f"lambda used = {lam_eff:.6e}")

    def objective_terms(V, U, L):
        data = data_term(V, U)
        total_pen = float(np.sum((L @ V) * V))
        spatial = float(np.sum((L_base @ V) * V))
        return data, spatial, total_pen - spatial

    L = L_base
    f_best = np.inf
    n_stall = 0
    best = None
    pi = [None] * (n_times - 1)

    if one_step:
        outer_max = 1

    temporal_trace = 0.0
    for i in range(n_times - 1):
        dt = rescaled_time[i + 1] - rescaled_time[i]
        temporal_trace += 2.0 / dt if dt != 0 else 2.0

    if auto_mu or spatial_only:
        mu_eff = None
    else:
        mu_eff = (
            mu * data_scale / max(temporal_trace, 1e-15)
            if normalize_terms else mu
        )

    for iter_outer in range(outer_max):
        if spatial_only:
            # No coupling at all: alternate U/V on the spatial Laplacian.
            L = L_base
        else:
            # -- Couplings ---------------------------------------------------
            pi_current = [None] * (n_times - 1)

            for i in range(n_times - 1):
                n_i, n_ip = adata_group[i].n_obs, adata_group[i + 1].n_obs
                a = np.full(n_i, 1.0 / n_i, dtype=dtype)
                b = np.full(n_ip, 1.0 / n_ip, dtype=dtype)

                if coupling is None:
                    C = scipy.spatial.distance.cdist(
                        V_list[i].astype(np.float64, copy=False),
                        V_list[i + 1].astype(np.float64, copy=False),
                        metric="sqeuclidean",
                    )
                    C = C / (C.max() + 1e-15)
                    pi_i = _compute_coupling(
                        ot_solver, C,
                        adata_group[i], adata_group[i + 1],
                        V_list[i], V_list[i + 1],
                        a, b, alpha, verbose,
                    )
                else:
                    # A reused coupling may already be sparse (sparsify=True);
                    # np.asarray on a sparse matrix would wrap it in a 0-d object.
                    pi_i = coupling[i]
                    pi_i = (pi_i.astype(dtype).tocsr() if sp.issparse(pi_i)
                            else np.asarray(pi_i, dtype=dtype))

                pi_i = pi_i / (pi_i.sum() + 1e-15)
                if not normalize_terms:
                    pi_i = pi_i * np.sqrt(pi_i.shape[0] * pi_i.shape[1])

                dt = rescaled_time[i + 1] - rescaled_time[i]
                if dt != 0:
                    pi_i = pi_i / dt

                if sparsify and not sp.issparse(pi_i):
                    pi_i = sparsify_pi_sparse(pi_i, k_row=k, k_col=k).astype(dtype)

                pi_current[i] = pi_i

            if auto_mu and mu_eff is None:
                T0 = 0.0
                for i, pi_i in enumerate(pi_current):
                    V_i = V_list[i]
                    V_ip = V_list[i + 1]
                    a_marg = np.asarray(pi_i.sum(axis=1)).ravel()
                    b_marg = np.asarray(pi_i.sum(axis=0)).ravel()
                    cross = pi_i.dot(V_ip) if sp.issparse(pi_i) else pi_i @ V_ip
                    T0 += (
                        float(np.sum(a_marg[:, None] * (V_i * V_i)))
                        + float(np.sum(b_marg[:, None] * (V_ip * V_ip)))
                        - 2.0 * float(np.sum(V_i * cross))
                    )

                D_spatial = data_term(V, U)
                mu_eff = target_temporal_ratio * D_spatial / max(T0, 1e-30)

                if verbose:
                    print(f"mu used = {mu_eff:.6e}")

            # -- Build block Laplacian --------------------------------------
            diag_blocks = [blk.copy() for blk in base_diag_blocks]
            off_diag_blocks = []

            for i, pi_i in enumerate(pi_current):
                a_marg = np.asarray(pi_i.sum(axis=1)).ravel()
                b_marg = np.asarray(pi_i.sum(axis=0)).ravel()

                diag_blocks[i] = (
                    diag_blocks[i] + mu_eff * sp.diags(a_marg, format="csr")
                )
                diag_blocks[i + 1] = (
                    diag_blocks[i + 1] + mu_eff * sp.diags(b_marg, format="csr")
                )

                block = -mu_eff * pi_i
                off_diag_blocks.append((i, i + 1, block))
                off_diag_blocks.append((i + 1, i, block.T))

            pi = pi_current
            L = assemble_blocks(diag_blocks, offdiag_blocks=off_diag_blocks).tocsr()

        # -- V step ----------------------------------------------------------
        V = solve_V_universal(
            X, U, L,
            niter_max=niter_max, tol=tol, eta=eta,
            verbose=verbose, V0=V, step_rate=step_rate,
            random_state=random_state,
        ).astype(dtype, copy=False)

        # -- U step ----------------------------------------------------------
        U, scale, kept = solve_U_nnls(
            X, V, n_jobs=n_jobs, dtype=dtype, verbose=verbose,
            activity_threshold=activity_threshold,
            min_active_frac=min_active_frac,
        )
        V = V[:, kept] * scale[None, :]
        r = U.shape[1]

        for i in range(n_times):
            V_list[i] = V[n_spots[i]:n_spots[i + 1], :]

        # -- Term balance and early stopping --------------------------------
        data, spatial, coupling_pen = objective_terms(V, U, L)
        f_curr = data + spatial + coupling_pen

        improvement = (
            (f_best - f_curr) / max(1.0, abs(f_best))
            if np.isfinite(f_best) else np.inf
        )

        if improvement > outer_tol:
            f_best = f_curr
            n_stall = 0
            best = (
                U.copy(), V.copy(), [V_i.copy() for V_i in V_list],
                [None if p_i is None else p_i.copy() for p_i in pi],
                L.copy(),
            )
        else:
            n_stall += 1

        if verbose:
            print(
                f"outer {iter_outer:4d}  f={f_curr:.6e}  "
                f"improvement={improvement:.3e}  r={r}  "
                f"stall={n_stall}/{patience}"
            )
            print(
                f"data={data:.4e}  "
                f"spatial={spatial:.4e} ({spatial / data:.2%})  "
                f"coupling={coupling_pen:.4e} ({coupling_pen / data:.2%})"
            )

        if n_stall >= patience:
            if verbose:
                print(
                    f"early stop at outer {iter_outer}: no improvement in "
                    f"{patience} iterations (best f={f_best:.6e})"
                )
            break

    if best is not None:
        U, V, V_list, pi, L = best

    if not return_ratio:
        return U, V_list, pi, L

    data, spatial, coupling_pen = objective_terms(V, U, L)
    denom = max(data, 1e-30)
    return U, V_list, pi, L, data, spatial / denom, coupling_pen / denom


def U_to_dataframe(U, gene_names):
    """Wrap U (d x r) as a factors x genes frame, the pd_B layout."""
    return pd.DataFrame(U.T, columns=gene_names)


# ─────────────────────────────────────────────────────────────────────────────
# Threshold-based tuning for the reference-free solver: lam and mu, as two
# separate steps (mirrors tune_lam / tune_mu for the reference-based solver)
# ─────────────────────────────────────────────────────────────────────────────
def tune_lam_UV(
    adata_group, r,
    lam_grid=None, target_spatial_ratio=0.01,
    time=None, outer_max=5, niter_max=1e3, tol=1e-2, eta=1e-2,
    outer_tol=None, patience=5,
    k=10, U_init=None, V_init=None, ot_solver="fgw", one_step=False,
    step_rate="automatic", alpha=0.5, dtype=np.float32, sparsify=False,
    n_jobs=-1, activity_threshold=1e-2, min_active_frac=0.05,
    random_state=0, normalize_terms=True, verbose=True,
):
    """Tune lam for the reference-free (U, V) solver, with mu=-1 so no OT
    coupling is built.

    Same rule as `tune_lam`: increase lam along `lam_grid`, check the
    *achieved* spatial_ratio at each step (from the actual solved (U, V) via
    `return_ratio=True`, not an estimate), and once the sweep crosses
    `target_spatial_ratio`, keep whichever of the two neighboring grid points
    is closest to it.

    Because U is learned here (and factors can be pruned as lam changes),
    each candidate lam is warm-started from the *previous* lam's (U, V) --
    `r` shrinks to match whatever rank survived pruning. The very first lam
    uses `U_init`/`V_init` if given, else the normal SVD-based init with the
    requested `r`.

    Returns (lam_star, U_star, V_star, trace) where V_star is the list of
    per-slice V's at lam_star, and trace is a list of
    (lam, data_fitting, spatial_ratio, r_effective) tuples.
    """
    if lam_grid is None:
        lam_grid = [0.0, 1e0, 2e0, 3e0, 4e0, 5e0,6e0,1e1,3e1,1e2]
    lam_grid = sorted(lam_grid)

    common_kwargs = dict(
        time=time, outer_max=outer_max, niter_max=niter_max, tol=tol, eta=eta,
        outer_tol=outer_tol, patience=patience, k=k, ot_solver=ot_solver,
        one_step=one_step, step_rate=step_rate, alpha=alpha, dtype=dtype,
        sparsify=sparsify, n_jobs=n_jobs, activity_threshold=activity_threshold,
        min_active_frac=min_active_frac, random_state=random_state,
        normalize_terms=normalize_terms,
    )

    if verbose:
        print(f"=== tune_lam_UV: sweeping lam with mu=-1, target spatial_ratio={target_spatial_ratio:g} ===")

    trace = []
    prev = None  # dict: lam, U, V, s_ratio
    lam_star = lam_grid[-1]
    U_star, V_star = U_init, V_init
    reached = False

    U_cur, V_cur, r_cur = U_init, V_init, r

    for lam_val in lam_grid:
        U_out, V_out, _, _, D, s_ratio, _ = solve_UV_all_slices(
            adata_group, r_cur, lam=lam_val, mu=-1,
            U_init=U_cur, V_init=V_cur,
            return_ratio=True, verbose=False, **common_kwargs,
        )
        r_eff = U_out.shape[1]
        trace.append((lam_val, D, s_ratio, r_eff))

        if verbose:
            print(f"  lam={lam_val:<10g} D={D:.6e}  spatial_ratio={s_ratio:.6e}  r={r_eff}")

        # Warm-start the next candidate from this one, tracking whatever
        # rank pruning left it with.
        U_cur, V_cur, r_cur = U_out, V_out, r_eff
        cur = dict(lam=lam_val, U=U_out, V=V_out, s_ratio=s_ratio)

        if s_ratio >= target_spatial_ratio:
            if prev is not None and abs(prev['s_ratio'] - target_spatial_ratio) <= abs(s_ratio - target_spatial_ratio):
                chosen = prev
            else:
                chosen = cur
            lam_star, U_star, V_star = chosen['lam'], chosen['U'], chosen['V']
            reached = True
            if verbose:
                print(
                    f"  -> crossed target_spatial_ratio={target_spatial_ratio:g} "
                    f"between lam={prev['lam'] if prev else lam_val:g} and lam={lam_val:g}; "
                    f"closest is lam={lam_star:g}"
                )
            break

        prev = cur

    if not reached:
        lam_star, U_star, V_star = prev['lam'], prev['U'], prev['V']
        if verbose:
            print(
                "  -> reached the end of lam_grid without hitting target_spatial_ratio; "
                "consider extending lam_grid upward."
            )

    if verbose:
        print(f"tune_lam_UV chosen lam* = {lam_star}\n")

    return lam_star, U_star, V_star, trace


def tune_mu_UV(
    adata_group, r, lam,
    mu_grid=None, target_temporal_ratio=0.01,
    time=None, outer_max=5, niter_max=1e3, tol=1e-2, eta=1e-2,
    outer_tol=None, patience=5,
    k=10, U_init=None, V_init=None, ot_solver="fgw", one_step=False,
    step_rate="automatic", alpha=0.5, dtype=np.float32, sparsify=False,
    n_jobs=-1, activity_threshold=1e-2, min_active_frac=0.05,
    random_state=0, normalize_terms=True, verbose=True,
):
    """Tune mu for the reference-free (U, V) solver at a fixed lam,
    computing the OT coupling exactly once.

    Solves once at (lam, mu=0) -- this builds the coupling from the
    spatially-regularized (U, V), which is then reused (`coupling=pi_fixed`)
    for every candidate mu below. Each candidate mu is warm-started from that
    same (U, V).

    Same closest-point stopping rule as `tune_mu`.

    Returns (mu_star, result, trace) where result is
    (U, V_list, pi, L, data_fitting, spatial_ratio, temporal_ratio) and trace
    is a list of (mu, data_fitting, temporal_ratio) tuples.
    """
    if mu_grid is None:
        mu_grid = [0.0, 1e0, 3e0, 1e1, 3e1, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6]
    mu_grid = sorted(mu_grid)

    common_kwargs = dict(
        time=time, outer_max=outer_max, niter_max=niter_max, tol=tol, eta=eta,
        outer_tol=outer_tol, patience=patience, k=k, ot_solver=ot_solver,
        one_step=one_step, step_rate=step_rate, alpha=alpha, dtype=dtype,
        sparsify=sparsify, n_jobs=n_jobs, activity_threshold=activity_threshold,
        min_active_frac=min_active_frac, random_state=random_state,
        normalize_terms=normalize_terms,
    )

    if verbose:
        print(f"=== tune_mu_UV: one OT solve at lam={lam:g}, target temporal_ratio={target_temporal_ratio:g} ===")

    # The one OT solve; mu=0 (not the mu<0 sentinel), so the coupling IS built.
    U0, V0, pi_fixed, L0, D0, s0, t0 = solve_UV_all_slices(
        adata_group, r, lam=lam, mu=0.0,
        U_init=U_init, V_init=V_init,
        return_ratio=True, verbose=False, **common_kwargs,
    )
    r_eff = U0.shape[1]

    trace = [(0.0, D0, t0)]
    if verbose:
        print(f"  mu={0.0:<10g} D={D0:.6e}  temporal_ratio={t0:.6e}  r={r_eff}")

    prev = dict(mu=0.0, result=(U0, V0, pi_fixed, L0, D0, s0, t0), t_ratio=t0)
    mu_star, result_star = prev['mu'], prev['result']
    reached = False

    for mu_val in mu_grid:
        if mu_val == 0.0:
            continue

        U_out, V_out, pi_out, L_out, D, s_out, t_out = solve_UV_all_slices(
            adata_group, r_eff, lam=lam, mu=mu_val,
            U_init=U0, V_init=V0, coupling=pi_fixed,
            return_ratio=True, verbose=False, **common_kwargs,
        )
        result = (U_out, V_out, pi_out, L_out, D, s_out, t_out)
        trace.append((mu_val, D, t_out))

        if verbose:
            print(f"  mu={mu_val:<10g} D={D:.6e}  temporal_ratio={t_out:.6e}  r={U_out.shape[1]}")

        cur = dict(mu=mu_val, result=result, t_ratio=t_out)

        if t_out >= target_temporal_ratio:
            if abs(prev['t_ratio'] - target_temporal_ratio) <= abs(t_out - target_temporal_ratio):
                chosen = prev
            else:
                chosen = cur
            mu_star, result_star = chosen['mu'], chosen['result']
            reached = True
            if verbose:
                print(
                    f"  -> crossed target_temporal_ratio={target_temporal_ratio:g} "
                    f"between mu={prev['mu']:g} and mu={mu_val:g}; closest is mu={mu_star:g}"
                )
            break

        prev = cur

    if not reached:
        mu_star, result_star = prev['mu'], prev['result']
        if verbose:
            print(
                "  -> reached the end of mu_grid without hitting target_temporal_ratio; "
                "consider extending mu_grid upward."
            )

    if verbose:
        print(f"tune_mu_UV chosen mu* = {mu_star}\n")

    return mu_star, result_star, trace