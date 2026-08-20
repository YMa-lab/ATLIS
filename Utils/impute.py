
import numpy as np
import pandas as pd
import ot

from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def classical_mds(D, n_components=2):
    """Classical MDS from a distance matrix (not squared).

    Returns (N, n_components) coordinates. Negative eigenvalues are clipped,
    which is what makes this safe for the non-Euclidean C_bar that fGW returns.

    We need this function because GW barycenter, technically, is just a distance matrix. 
    We need to put it back to the 2D coordinate space.
    """
    D = np.asarray(D, float)
    N = D.shape[0]

    J = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * J @ (D ** 2) @ J

    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    scale = np.sqrt(np.clip(eigvals[:n_components], 0, None))
    return eigvecs[:, :n_components] * scale[np.newaxis, :]


def farthest_point_sample(X, k, seed=0):
    """Greedy farthest-point sampling; returns indices of k well-spread points.

    Unlike random sampling this covers the tissue outline and sparse regions,
    which is what the GW structure matrix depends on. O(n*k) time, O(n) memory.
    """
    X = np.asarray(X, float)
    n = len(X)
    if k is None or k >= n:
        return np.arange(n)

    rng = np.random.default_rng(seed)
    idx = np.empty(k, dtype=int)
    idx[0] = rng.integers(n)

    d2 = ((X - X[idx[0]]) ** 2).sum(1)
    for i in range(1, k):
        idx[i] = int(np.argmax(d2))
        d2 = np.minimum(d2, ((X - X[idx[i]]) ** 2).sum(1))

    return idx


def _reduce_slice(X, Y, p, n_input, subsample, seed):
    """Subsample one slice to `n_input` spots, carrying the mass with it.

    FPS preserves geometric coverage but not density, so each kept spot
    inherits the mass of the original spots nearest to it and the empirical
    measure still matches the tissue.
    """
    n = len(X)
    if n_input is None or n_input >= n:
        return X, Y, ot.utils.unif(n) if p is None else np.asarray(p, float)

    if subsample == "fps":
        idx = farthest_point_sample(X, n_input, seed=seed)
    elif subsample == "random":
        idx = np.random.default_rng(seed).choice(n, n_input, replace=False)
    else:
        raise ValueError("subsample must be 'fps' or 'random'.")

    Xs = np.asarray(X, float)[idx]
    assign = cKDTree(Xs).query(np.asarray(X, float), k=1)[1]

    weights = None if p is None else np.asarray(p, float)
    mass = np.bincount(assign, weights=weights, minlength=len(idx)).astype(float)

    return Xs, Y[idx], mass / mass.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def impute_slice_fgw_barycenter(
    X0, Y0,
    X1, Y1,
    t=0.5,
    N_bary=None,
    p0=None,
    p1=None,
    alpha=0.5,
    n_components=2,
    lambdas=None,
    loss_fun="square_loss",
    symmetric=True,
    max_iter=100,
    tol=1e-9,
    stop_criterion="barycenter",
    warmstartT=False,
    armijo=False,
    verbose=False,
    random_state=None,
    pd_B=None,
    Y_is_proportion=False,
    dtype=None,
    n_input=None,
    subsample="fps",
    **kwargs,
):
    """Interpolate a slice between (X0, Y0) and (X1, Y1) via an fGW barycenter.

    X* are (n, 2) coordinates; Y* are (n, p) features — either gene expression
    or cell-type proportions over `pd_B.index`. `t` sets the temporal weight
    (0 gives slice 0, 1 gives slice 1) and `alpha` the structure/feature
    trade-off inside fGW.

    Y_is_proportion=True projects the barycenter features to genes via pd_B and
    returns both; False treats them as expression already and only reorders the
    columns to pd_B.columns when pd_B is given.

    n_input reduces each input slice to that many spots before solving.
    Per-iteration cost grows steeply in input size and the (N x n) couplings
    shrink linearly in n, so this buys both time and the memory headroom that
    limits N. dtype=np.float32 halves cost-matrix memory, though POT may upcast
    internally — verify it survives into the solve before relying on it.

    Returns
    -------
    Y_is_proportion : (X_bar_spatial, genes_df, X_bar_feat, C_bar, log)
    otherwise       : (X_bar_spatial, X_bar_feat, C_bar, log)
    """
    X0, X1 = np.asarray(X0), np.asarray(X1)
    Y0, Y1 = np.asarray(Y0), np.asarray(Y1)

    assert X0.shape[0] == Y0.shape[0], "X0 and Y0 disagree on spot count."
    assert X1.shape[0] == Y1.shape[0], "X1 and Y1 disagree on spot count."

    seed = 0 if random_state is None else int(random_state)
    X0s, Y0s, p0s = _reduce_slice(X0, Y0, p0, n_input, subsample, seed)
    X1s, Y1s, p1s = _reduce_slice(X1, Y1, p1, n_input, subsample, seed + 1)

    if N_bary is None:
        N_bary = int(np.round(0.5 * (len(X0s) + len(X1s))))

    # --- structural costs
    C0 = cdist(X0s, X0s, metric="sqeuclidean")
    C1 = cdist(X1s, X1s, metric="sqeuclidean")

    if dtype is not None:
        C0, C1 = C0.astype(dtype), C1.astype(dtype)
        Y0s, Y1s = Y0s.astype(dtype), Y1s.astype(dtype)

    if lambdas is None:
        lambdas = [1.0 - t, t]

    # --- fGW barycenter
    X_bar_feat, C_bar, log = ot.gromov.fgw_barycenters(
        N=N_bary,
        Ys=[Y0s, Y1s],
        Cs=[C0, C1],
        ps=[p0s, p1s],
        lambdas=lambdas,
        alpha=alpha,
        fixed_structure=False,
        fixed_features=False,
        p=None,
        loss_fun=loss_fun,
        armijo=armijo,
        symmetric=symmetric,
        max_iter=max_iter,
        tol=tol,
        stop_criterion=stop_criterion,
        warmstartT=warmstartT,
        verbose=verbose,
        log=True,
        init_C=None,
        init_X=None,
        random_state=random_state,
        **kwargs,
    )

    # --- embed barycenter geometry
    D_bar = np.sqrt(np.maximum(np.asarray(C_bar, float), 0.0))
    X_bar_spatial = classical_mds(D_bar, n_components=n_components)

    # --- project proportions to gene space if requested
    if Y_is_proportion:
        if pd_B is None:
            raise ValueError("pd_B must be provided when Y_is_proportion=True.")
        X_bar_gene = X_bar_feat @ pd_B.values
        return (
            X_bar_spatial,
            pd.DataFrame(X_bar_gene, columns=pd_B.columns),
            X_bar_feat,
            C_bar,
            log,
        )

    if pd_B is not None:
        if X_bar_feat.shape[1] != pd_B.shape[1]:
            raise ValueError("Gene dimension mismatch with pd_B.")
        X_bar_feat = pd.DataFrame(X_bar_feat, columns=pd_B.columns)

    return X_bar_spatial, X_bar_feat, C_bar, log

def align_by_pca(X_src, X_ref):
    src_mean = X_src.mean(axis=0)
    ref_mean = X_ref.mean(axis=0)
    X_src_c  = X_src - src_mean
    X_ref_c  = X_ref - ref_mean
    scale    = X_ref_c.std() / (X_src_c.std() + 1e-12)
    X_src_c  = X_src_c * scale
    tree_ref = cKDTree(X_ref_c)

    def chamfer(theta, X_in):
        c, s     = np.cos(theta), np.sin(theta)
        R        = np.array([[c, -s], [s, c]])
        X_rot    = X_in @ R.T
        tree_src = cKDTree(X_rot)
        d_sr, _  = tree_ref.query(X_rot)
        d_rs, _  = tree_src.query(X_ref_c)
        return d_sr.mean() + d_rs.mean()

    best_cost = np.inf
    best_X    = None
    for flip in [1, -1]:
        X_try      = X_src_c * np.array([flip, 1])
        thetas     = np.linspace(0, 2 * np.pi, 72)
        costs      = [chamfer(th, X_try) for th in thetas]
        best_theta = thetas[np.argmin(costs)]
        result     = minimize_scalar(
            lambda th: chamfer(th, X_try),
            bounds=(best_theta - 0.1, best_theta + 0.1),
            method='bounded'
        )
        if result.fun < best_cost:
            best_cost = result.fun
            c, s      = np.cos(result.x), np.sin(result.x)
            R_opt     = np.array([[c, -s], [s, c]])
            best_X    = X_try @ R_opt.T
    return best_X + ref_mean