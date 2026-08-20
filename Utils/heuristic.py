"""Heuristic selection of the spatial (lam) and temporal (mu) weights.

Both penalties are scale-free only relative to the data-fit term, so a value
of lam that regularizes one dataset sensibly can be far too weak or too strong
on another. `calibrate_lam_mu` solves once without regularization, measures the
raw energy of each penalty at that solution, and returns the multipliers that
would make each term a chosen fraction of the data fit.
"""

import anndata as ad
import numpy as np
import scipy.sparse as sp

from objective import laplacian_from_coords, solve_V_all_slices


def evaluate_terms_all_slices(
    adata_group, pd_B, V_list, pi_list, time=None, k=10, dtype=np.float32,
):
    """Raw (un-multiplied) values of the three objective terms at a given V.

    Returns data_fit, spatial_raw = sum_t tr(V_t^T L_t V_t), and coupling_raw
    = sum_t sum_ij pi_ij ||V_t,i - V_t+1,j||^2 — each without its lam / mu
    factor, and each computed on exactly the graph and rescaling the solver
    uses, so the ratios are directly interpretable as multipliers.
    """
    n_times = len(adata_group)

    adata = ad.concat(adata_group, axis=0, join="outer")
    common = adata.var_names.intersection(pd_B.columns)
    adata_group = [a_i[:, common].copy() for a_i in adata_group]

    B = np.asarray(pd_B[common].to_numpy().T, dtype=dtype)
    V_list = [np.asarray(V_i, dtype=dtype) for V_i in V_list]
    V = np.vstack(V_list)

    # Must match solve_V_all_slices, which reads .X — not layers['counts'].
    X_list = []
    for a_i in adata_group:
        Xi = a_i.X.T
        X_list.append(
            Xi.tocsr().astype(dtype) if sp.issparse(Xi)
            else sp.csr_matrix(np.asarray(Xi, dtype=dtype))
        )
    X = sp.hstack(X_list).tocsr()

    G = X.T @ B
    if sp.issparse(G):
        G = G.toarray()
    H = B.T @ B
    xnorm2 = float(X.multiply(X).sum())

    data_fit = xnorm2 - 2.0 * float(np.sum(V * G)) + float(np.sum((V @ H) * V))

    # Same graph the solver builds: normed=True, same k.
    spatial_raw = 0.0
    for a_i, V_i in zip(adata_group, V_list):
        Li, _ = laplacian_from_coords(
            a_i.obs["x"].to_numpy(), a_i.obs["y"].to_numpy(), k=k, normed=True
        )
        spatial_raw += float(np.sum((Li @ V_i) * V_i))

    if time is None:
        time = list(range(n_times))
    if time[-1] == time[0]:
        rescaled_time = [0.0] * len(time)
    else:
        span = time[-1] - time[0]
        rescaled_time = [(t - time[0]) / span for t in time]

    coupling_raw = 0.0
    for i in range(n_times - 1):
        pi_i = None if pi_list is None else pi_list[i]
        if pi_i is None:
            continue

        pi_i = np.asarray(
            pi_i.toarray() if sp.issparse(pi_i) else pi_i, dtype=np.float64
        )
        # Same rescaling chain as the solver.
        pi_i = pi_i / (pi_i.sum() + 1e-15)
        pi_i = pi_i * np.sqrt(pi_i.shape[0] * pi_i.shape[1])
        dt = rescaled_time[i + 1] - rescaled_time[i]
        if dt != 0:
            pi_i = pi_i / dt

        Vi = np.asarray(V_list[i], np.float64)
        Vip = np.asarray(V_list[i + 1], np.float64)

        coupling_raw += (
            float(np.sum(pi_i.sum(1)[:, None] * Vi ** 2))
            + float(np.sum(pi_i.sum(0)[:, None] * Vip ** 2))
            - 2.0 * float(np.sum((pi_i @ Vip) * Vi))
        )

    return {
        "data_fit": data_fit,
        "spatial_raw": spatial_raw,
        "coupling_raw": coupling_raw,
    }


def calibrate_lam_mu(
    adata_group, pd_B,
    time=None, k=10,
    spatial_frac=0.01, coupling_frac=0.01,
    n_rounds=1,
    reuse_couplings=True,
    verbose=True,
    ot_solver = 'pot',
    **solver_kwargs,
):
    """Choose lam and mu so each penalty is a target fraction of the data fit.

    Solves with lam=mu=0, measures the raw penalty energies at that solution,
    and sets

        lam = spatial_frac  * data_fit / spatial_raw
        mu  = coupling_frac * data_fit / coupling_raw

    so that, evaluated at the unregularized V, each term contributes its target
    share. Regularizing then smooths V and shrinks both penalties, so the
    realized share at the regularized solution comes out *below* target —
    `n_rounds > 1` re-solves at the current lam/mu and recalibrates from there,
    which converges the realized fractions toward the requested ones.

    `reuse_couplings` passes the first round's pi back in, skipping the OT
    solve on later rounds. Turn it off if the couplings should track the
    updated V (matters more for `ot_solver="fgw"` than for the low-rank paths,
    and costs a full OT solve per round).

    Extra keyword arguments go to `solve_V_all_slices`.

    Returns (lam, mu, history) where history is one dict of terms per round.
    """
    if n_rounds < 1:
        raise ValueError("n_rounds must be at least 1.")

    lam, mu = 0.0, 0.0
    coupling = None
    history = []

    for round_idx in range(n_rounds):
        if verbose:
            print(f"── calibration round {round_idx}  (lam={lam:.4g}, mu={mu:.4g})")

        V, pi, _ = solve_V_all_slices(
            adata_group, pd_B,
            lam=lam, mu=mu, time=time, k=k,
            coupling=coupling, verbose=verbose,
            ot_solver = ot_solver,
            **solver_kwargs,
        )

        terms = evaluate_terms_all_slices(
            adata_group, pd_B, V, pi, time=time, k=k,
            dtype=solver_kwargs.get("dtype", np.float32),
        )

        lam = spatial_frac * terms["data_fit"] / max(terms["spatial_raw"], 1e-12)
        mu = coupling_frac * terms["data_fit"] / max(terms["coupling_raw"], 1e-12)

        terms = dict(terms, lam=lam, mu=mu)
        history.append(terms)

        if verbose:
            print(f"   data_fit={terms['data_fit']:.4e}  "
                  f"spatial_raw={terms['spatial_raw']:.4e}  "
                  f"coupling_raw={terms['coupling_raw']:.4e}")
            print(f"   -> lam={lam:.4e}  mu={mu:.4e}")

        if reuse_couplings and coupling is None:
            coupling = pi

    return lam, mu, history


def realized_fractions(adata_group, pd_B, V, pi, lam, mu, time=None, k=10,
                       dtype=np.float32):
    """What share of the objective each penalty actually takes at a solution.

    Use this to check a calibration: run the solver at the returned lam/mu,
    pass the result here, and compare against the fractions you asked for.
    """
    terms = evaluate_terms_all_slices(
        adata_group, pd_B, V, pi, time=time, k=k, dtype=dtype
    )
    return {
        "spatial_frac": lam * terms["spatial_raw"] / max(terms["data_fit"], 1e-12),
        "coupling_frac": mu * terms["coupling_raw"] / max(terms["data_fit"], 1e-12),
        **terms,
    }