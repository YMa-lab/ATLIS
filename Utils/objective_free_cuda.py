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
    """Scale each column to unit L2 norm; returns (U_normalized, norms)."""
    norms = torch.clamp(U.pow(2).sum(dim=0).sqrt(), min=eps)
    return U / norms[None, :], norms


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
    verbose=True, coupling=None, k=10, k_graph=None,
    V_init=None, U_init=None,
    ot_solver="fgw", one_step=True,
    step_rate="automatic", alpha=0.9,
    dtype=torch.float32, device=None, sparsify=True,
    activity_threshold=1e-3, min_active_frac=0.01,
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

    Returns (U, V_per_slice, pi_list, L).
    """
    device = get_device(device)
    if k_graph is None:
        k_graph = k

    # lam and mu were tuned against an unnormalized U; column-normalizing U
    # shrinks the data term by ~1e2, so the penalties are rescaled to match.
    lam, mu = lam / 1e2, mu / 1e2

    n_times = len(adata_group)
    n_spots = [0]
    for a_i in adata_group:
        n_spots.append(n_spots[-1] + a_i.shape[0])

    common = ad.concat(adata_group, axis=0, join="inner").var_names
    adata_group = [a_i[:, common].copy() for a_i in adata_group]
    genes = adata_group[0].var_names
    d = len(genes)

    if time is None:
        time = list(range(n_times))
    if time[-1] == time[0]:
        rescaled_time = [0.0] * n_times
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

    # X: (d x N) on device
    parts = []
    for a_i in adata_group:
        Xi = a_i.layers["counts"].T if "counts" in a_i.layers else a_i.X.T
        parts.append(Xi if sp.issparse(Xi) else sp.csr_matrix(np.asarray(Xi)))
    X = to_torch_csr(sp.hstack(parts, format="csr"), device, dtype=dtype)
    xnorm2 = (X.values() ** 2).sum()

    # ── Initialize U and V ───────────────────────────────────────────────
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

    def objective(V, U, L):
        G = torch.sparse.mm(X.t(), U)
        H = U.T @ U
        data = xnorm2 - 2.0 * (V * G).sum() + ((V @ H) * V).sum()
        return float(data + ((L @ V) * V).sum())

    L = assemble_blocks_torch(base_diag_blocks, device=device, dtype=dtype)
    f_prev = objective(V, U, L)
    pi = [None] * (n_times - 1)

    if one_step:
        outer_max = 1
    if mu < 0:
        outer_max = -1

    for iter_outer in range(outer_max):
        # ── Couplings and block Laplacian ────────────────────────────────
        diag_blocks = list(base_diag_blocks)
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
            pi_i = pi_i * float(np.sqrt(pi_i.shape[0] * pi_i.shape[1]))

            dt = rescaled_time[i + 1] - rescaled_time[i]
            if dt != 0:
                pi_i = pi_i / dt
            if sparsify:
                pi_i = sparsify_pi_torch(pi_i, k_row=k, k_col=k)
            pi[i] = pi_i

            diag_blocks[i] = add_csr(
                diag_blocks[i], diag_csr(mu * pi_i.sum(dim=1), device)
            )
            diag_blocks[i + 1] = add_csr(
                diag_blocks[i + 1], diag_csr(mu * pi_i.sum(dim=0), device)
            )

            block = -mu * pi_i
            off_diag_blocks.append((i, i + 1, block))
            off_diag_blocks.append((i + 1, i, block.T))

        L = assemble_blocks_torch(
            diag_blocks, offdiag_blocks=off_diag_blocks, device=device, dtype=dtype
        )
        if mu < 0:
            continue

        # ── V step: graph-regularized, U fixed ───────────────────────────
        V = solve_V_universal_torch(
            X, U, L,
            niter_max=niter_max, tol=tol, eta=eta,
            verbose=verbose, V0=V, step_rate=step_rate,
            device=device, dtype=dtype,
        )

        # ── U step: NNLS, then prune and absorb the scale into V ─────────
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

    if return_numpy:
        return (
            U.cpu().numpy(),
            [V_i.cpu().numpy() for V_i in V_list],
            [None if p is None else sp.csr_matrix(p.cpu().numpy()) for p in pi],
            L,
        )
    return U, V_list, pi, L


def U_to_dataframe(U, gene_names):
    """Wrap U (d x r) as a factors x genes frame, the pd_B layout."""
    U = U.cpu().numpy() if torch.is_tensor(U) else np.asarray(U)
    return pd.DataFrame(U.T, columns=gene_names)