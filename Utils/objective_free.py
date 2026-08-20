"""Alternating GNMF across time slices: X ≈ U V^T with no reference profiles.

Entry point: `solve_UV_all_slices`. Unlike `solve_V_all_slices`, the gene basis
U is learned rather than supplied, so the factors are discovered programs
instead of known cell types.

    U : (d x r)  gene basis, column-normalized
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


def init_U_from_svd(X, r, dtype=np.float32, verbose=True):
    """Warm-start U from the left singular vectors of X.

    SVD gives the best rank-r approximation, so it is the natural starting
    point before nonnegativity is imposed; signs are discarded since only the
    magnitudes carry over.
    """
    from sklearn.utils.extmath import randomized_svd

    P, sigma, _ = randomized_svd(X, n_components=r, random_state=0)
    U = np.abs(P)
    U = U / np.maximum(np.sqrt((U * U).sum(axis=0)), 1e-15)[None, :]

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

    Returns (U column-normalized, the norms that were divided out, kept
    column indices) — the caller absorbs `scale` into V to leave U V^T fixed.
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

    scale = np.maximum(np.sqrt((U * U).sum(axis=0)), 1e-15)
    U = U / scale[None, :]

    if verbose:
        print(f"  kept {U.shape[1]} factors")
    return U.astype(dtype), scale.astype(dtype), kept


def solve_UV_all_slices(
    adata_group, r,
    lam=10, mu=1, time=None,
    outer_max=5, niter_max=1e3, tol=1e-5, eta=1e-3,
    verbose=True, coupling=None, k=10,
    V_init=None, U_init=None,
    ot_solver="fgw", one_step=True,
    step_rate="automatic", alpha=0.9,
    dtype=np.float32, sparsify=True,
    n_jobs=-1, activity_threshold=1e-3, min_active_frac=0.01,
):
    """Learn a shared gene basis and per-slice coefficients across time.

    Each outer iteration: compute OT couplings between adjacent slices from
    the current V, rebuild the block Laplacian (per-slice spatial graphs
    weighted by `lam`, cross-slice coupling by `mu`), update V under it, then
    refit U by NNLS and drop factors no longer used anywhere.

    `r` is an upper bound — pruning can return fewer. `mu < 0` computes the
    couplings and skips both factor updates.

    Returns (U, V_per_slice, pi_list, L).
    """
    # lam and mu were tuned against an unnormalized U; column-normalizing U
    # shrinks the data term by ~1e2, so the penalties are rescaled to match.
    lam, mu = lam / 1e2, mu / 1e2

    n_times = len(adata_group)
    n_spots = [0]
    for a_i in adata_group:
        n_spots.append(n_spots[-1] + a_i.shape[0])

    adata_group = [
        a_i[:, ad.concat(adata_group, axis=0, join="inner").var_names].copy()
        for a_i in adata_group
    ]
    genes = adata_group[0].var_names
    d = len(genes)

    if time is None:
        time = list(range(n_times))
    if time[-1] == time[0]:
        rescaled_time = [0.0] * n_times
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

    # X: (d x N)
    parts = []
    for a_i in adata_group:
        Xi = a_i.layers["counts"].T if "counts" in a_i.layers else a_i.X.T
        parts.append(
            Xi.tocsr().astype(dtype) if sp.issparse(Xi)
            else sp.csr_matrix(np.asarray(Xi, dtype=dtype))
        )
    X = sp.hstack(parts, format="csr")
    xnorm2 = float(X.multiply(X).sum())

    # ── Initialize U and V ───────────────────────────────────────────────
    if U_init is not None:
        U = np.asarray(U_init, dtype=dtype).copy()
        if U.shape != (d, r):
            raise ValueError(f"U_init shape {U.shape} != {(d, r)}")
    else:
        U = init_U_from_svd(X, r, dtype=dtype, verbose=verbose)
    U = U / np.maximum(np.sqrt((U * U).sum(axis=0)), 1e-15)[None, :]

    if V_init is not None:
        V_list = [np.asarray(V_i, dtype=dtype) for V_i in V_init]
    else:
        # One unregularized V solve against the initial U, per slice.
        V_list = []
        for i, a_i in enumerate(adata_group):
            Xi = X[:, n_spots[i]:n_spots[i + 1]]
            V_list.append(
                solve_V_universal(
                    Xi, U, base_diag_blocks[i],
                    niter_max=niter_max, tol=tol, eta=eta,
                    verbose=False, step_rate=step_rate,
                ).astype(dtype)
            )
    V = np.vstack(V_list).astype(dtype, copy=False)

    def objective(V, U, L):
        G = X.T @ U
        if sp.issparse(G):
            G = G.toarray()
        H = U.T @ U
        data = xnorm2 - 2.0 * float(np.sum(V * G)) + float(np.sum((V @ H) * V))
        return data + float(np.sum((L @ V) * V))

    L = assemble_blocks(base_diag_blocks).tocsr()
    f_prev = objective(V, U, L)
    pi = [None] * (n_times - 1)

    if one_step:
        outer_max = 1
    if mu < 0:
        outer_max = -1

    for iter_outer in range(outer_max):
        # ── Couplings and block Laplacian ────────────────────────────────
        diag_blocks = [blk.copy() for blk in base_diag_blocks]
        off_diag_blocks = []

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
                pi_i = np.asarray(coupling[i], dtype=dtype)

            pi_i = pi_i / (pi_i.sum() + 1e-15)
            pi_i = pi_i * np.sqrt(pi_i.shape[0] * pi_i.shape[1])
            dt = rescaled_time[i + 1] - rescaled_time[i]
            if dt != 0:
                pi_i = pi_i / dt
            if sparsify:
                pi_i = sparsify_pi_sparse(pi_i, k_row=k, k_col=k).astype(dtype)
            pi[i] = pi_i

            a_marg = np.asarray(pi_i.sum(axis=1)).ravel()
            b_marg = np.asarray(pi_i.sum(axis=0)).ravel()
            diag_blocks[i] = diag_blocks[i] + mu * sp.diags(a_marg, format="csr")
            diag_blocks[i + 1] = diag_blocks[i + 1] + mu * sp.diags(b_marg, format="csr")

            block = -mu * pi_i
            off_diag_blocks.append((i, i + 1, block))
            off_diag_blocks.append((i + 1, i, block.T))

        L = assemble_blocks(diag_blocks, offdiag_blocks=off_diag_blocks).tocsr()
        if mu < 0:
            continue

        # ── V step: graph-regularized, U fixed ───────────────────────────
        V = solve_V_universal(
            X, U, L,
            niter_max=niter_max, tol=tol, eta=eta,
            verbose=verbose, V0=V, step_rate=step_rate,
        ).astype(dtype, copy=False)

        # ── U step: NNLS, then prune and absorb the scale into V ─────────
        U, scale, kept = solve_U_nnls(
            X, V, n_jobs=n_jobs, dtype=dtype, verbose=verbose,
            activity_threshold=activity_threshold,
            min_active_frac=min_active_frac,
        )
        V = V[:, kept] * scale[None, :]
        r = U.shape[1]

        for i in range(n_times):
            V_list[i] = V[n_spots[i]:n_spots[i + 1], :]

        f_curr = objective(V, U, L)
        rel_drop = (f_prev - f_curr) / max(1.0, abs(f_prev))

        if verbose:
            print(f"outer {iter_outer:4d}  f={f_curr:.6e}  "
                  f"rel_drop={rel_drop:.3e}  r={r}")

        if f_curr <= f_prev and rel_drop < tol:
            if verbose:
                print("converged after outer iteration #", iter_outer)
            break

        f_prev = f_curr

    return U, V_list, pi, L


def U_to_dataframe(U, gene_names):
    """Wrap U (d x r) as a factors x genes frame, the pd_B layout."""
    return pd.DataFrame(U.T, columns=gene_names)