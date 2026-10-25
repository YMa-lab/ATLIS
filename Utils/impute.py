
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


def align_by_rotation(X_src, X_ref, n_angles=720, bidirectional=True,
                      refine=True):
    """Align X_src to X_ref by searching rotation + reflection on a grid.

    Replaces the PCA-axis version. PCA matched the two clouds' principal axes,
    which breaks when the cloud is near-isotropic (close eigenvalues leave the
    axes essentially arbitrary) or when src and ref differ enough in shape that
    their axes don't correspond. Searching the actual objective has neither
    failure mode.

    Scores every angle on a [0, 2pi) grid, for both reflection states, by
    Chamfer distance (mean nearest-neighbour distance in both directions), and
    keeps the best. Reflection is searched explicitly because rotations alone
    cannot produce a mirror image, and a flipped barycenter is the common case
    here -- fGW sees only intra-slice distances, so it fixes the shape but not
    the handedness.

    Point counts need not match. Both clouds are centred before the search and
    the result is placed at X_ref's centroid, so translation is handled;
    scale is NOT -- if the two slices differ in physical scale, normalize
    before calling.

    refine : polish the best grid angle with a local golden-section search,
    so accuracy isn't limited by grid spacing (360/n_angles degrees).
    """
    X_src = np.asarray(X_src, float)
    X_ref = np.asarray(X_ref, float)

    mu_src = X_src.mean(0)
    mu_ref = X_ref.mean(0)
    S = X_src - mu_src
    R_ = X_ref - mu_ref

    tree_ref = cKDTree(R_)
    # Trees on the (unrotated) source, one per reflection state. Rotating the
    # source and querying it from the reference is the same as rotating the
    # reference the other way and querying the fixed source, so these are
    # built once instead of rebuilt per angle.
    flips = [np.array([1.0, 1.0]), np.array([1.0, -1.0])]
    trees_src = [cKDTree(S * f) for f in flips]

    def chamfer(theta, fi):
        c, s = np.cos(theta), np.sin(theta)
        Rot = np.array([[c, -s], [s, c]])
        moved = (S * flips[fi]) @ Rot.T
        d1 = tree_ref.query(moved, k=1)[0].mean()
        if not bidirectional:
            return d1
        d2 = trees_src[fi].query(R_ @ Rot, k=1)[0].mean()
        return 0.5 * (d1 + d2)

    grid = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    best = (np.inf, 0.0, 0)
    for fi in range(2):
        for th in grid:
            v = chamfer(th, fi)
            if v < best[0]:
                best = (v, th, fi)

    _, theta, fi = best

    if refine:
        step = 2.0 * np.pi / n_angles
        res = minimize_scalar(lambda th: chamfer(th, fi),
                              bounds=(theta - step, theta + step),
                              method="bounded")
        if res.fun < best[0]:
            theta = float(res.x)

    c, s = np.cos(theta), np.sin(theta)
    Rot = np.array([[c, -s], [s, c]])
    return (S * flips[fi]) @ Rot.T + mu_ref
    
def align_by_pca(X_src, X_ref):
    """Rotate X_src to match the principal axis orientation of X_ref."""
    def pca_axes(X):
        X_c = X - X.mean(axis=0)
        cov = X_c.T @ X_c
        _, vecs = np.linalg.eigh(cov)
        return vecs[:, ::-1]  # sort by descending eigenvalue

    R_src = pca_axes(X_src)
    R_ref = pca_axes(X_ref)

    R = R_ref @ R_src.T

    X_src_c = X_src - X_src.mean(axis=0)
    X_aligned = X_src_c @ R.T + X_ref.mean(axis=0)
    return X_aligned

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
# def impute_slice_fgw_barycenter(
#     X0, Y0,
#     X1, Y1,
#     t=0.5,
#     N_bary=None,
#     p0=None,
#     p1=None,
#     alpha=0.5,
#     n_components=2,
#     lambdas=None,
#     loss_fun="square_loss",
#     symmetric=True,
#     max_iter=100,
#     tol=1e-9,
#     stop_criterion="barycenter",
#     warmstartT=False,
#     armijo=False,
#     verbose=False,
#     random_state=None,
#     pd_B=None,
#     Y_is_proportion=False,
#     dtype=None,
#     n_input=None,
#     subsample="fps",
#     **kwargs,
# ):
#     """Interpolate a slice between (X0, Y0) and (X1, Y1) via an fGW barycenter.

#     X* are (n, 2) coordinates; Y* are (n, p) features — either gene expression
#     or cell-type proportions over `pd_B.index`. `t` sets the temporal weight
#     (0 gives slice 0, 1 gives slice 1) and `alpha` the structure/feature
#     trade-off inside fGW.

#     Y_is_proportion=True projects the barycenter features to genes via pd_B and
#     returns both; False treats them as expression already and only reorders the
#     columns to pd_B.columns when pd_B is given.

#     n_input reduces each input slice to that many spots before solving.
#     Per-iteration cost grows steeply in input size and the (N x n) couplings
#     shrink linearly in n, so this buys both time and the memory headroom that
#     limits N. dtype=np.float32 halves cost-matrix memory, though POT may upcast
#     internally — verify it survives into the solve before relying on it.

#     Returns
#     -------
#     Y_is_proportion : (X_bar_spatial, genes_df, X_bar_feat, C_bar, log)
#     otherwise       : (X_bar_spatial, X_bar_feat, C_bar, log)
#     """
#     X0, X1 = np.asarray(X0), np.asarray(X1)
#     Y0, Y1 = np.asarray(Y0), np.asarray(Y1)

#     assert X0.shape[0] == Y0.shape[0], "X0 and Y0 disagree on spot count."
#     assert X1.shape[0] == Y1.shape[0], "X1 and Y1 disagree on spot count."

#     seed = 0 if random_state is None else int(random_state)
#     X0s, Y0s, p0s = _reduce_slice(X0, Y0, p0, n_input, subsample, seed)
#     X1s, Y1s, p1s = _reduce_slice(X1, Y1, p1, n_input, subsample, seed + 1)

#     if N_bary is None:
#         N_bary = int(np.round(0.5 * (len(X0s) + len(X1s))))

#     # --- structural costs
#     C0 = cdist(X0s, X0s, metric="sqeuclidean")
#     C1 = cdist(X1s, X1s, metric="sqeuclidean")

#     if dtype is not None:
#         C0, C1 = C0.astype(dtype), C1.astype(dtype)
#         Y0s, Y1s = Y0s.astype(dtype), Y1s.astype(dtype)

#     if lambdas is None:
#         lambdas = [1.0 - t, t]

#     # --- fGW barycenter
#     X_bar_feat, C_bar, log = ot.gromov.fgw_barycenters(
#         N=N_bary,
#         Ys=[Y0s, Y1s],
#         Cs=[C0, C1],
#         ps=[p0s, p1s],
#         lambdas=lambdas,
#         alpha=alpha,
#         fixed_structure=False,
#         fixed_features=False,
#         p=None,
#         loss_fun=loss_fun,
#         armijo=armijo,
#         symmetric=symmetric,
#         max_iter=max_iter,
#         tol=tol,
#         stop_criterion=stop_criterion,
#         warmstartT=warmstartT,
#         verbose=verbose,
#         log=True,
#         init_C=None,
#         init_X=None,
#         random_state=random_state,
#         **kwargs,
#     )

#     # --- embed barycenter geometry
#     D_bar = np.sqrt(np.maximum(np.asarray(C_bar, float), 0.0))
#     X_bar_spatial = classical_mds(D_bar, n_components=n_components)

#     # --- project proportions to gene space if requested
#     if Y_is_proportion:
#         if pd_B is None:
#             raise ValueError("pd_B must be provided when Y_is_proportion=True.")
#         X_bar_gene = X_bar_feat @ pd_B.values
#         return (
#             X_bar_spatial,
#             pd.DataFrame(X_bar_gene, columns=pd_B.columns),
#             X_bar_feat,
#             C_bar,
#             log,
#         )

#     if pd_B is not None:
#         if X_bar_feat.shape[1] != pd_B.shape[1]:
#             raise ValueError("Gene dimension mismatch with pd_B.")
#         X_bar_feat = pd.DataFrame(X_bar_feat, columns=pd_B.columns)

#     return X_bar_spatial, X_bar_feat, C_bar, log



def _prepare_fgw_inputs(X0, Y0, X1, Y1, n_input, subsample, dtype, random_state=None):
    """Do the expensive, t-independent work once: reduction + cost matrices.
 
    Cache/reuse the output of this across multiple calls that only vary `t`
    (e.g. when interpolating several time points between the same two slices).
    """
    X0, X1 = np.asarray(X0), np.asarray(X1)
    Y0, Y1 = np.asarray(Y0), np.asarray(Y1)
 
    assert X0.shape[0] == Y0.shape[0], "X0 and Y0 disagree on spot count."
    assert X1.shape[0] == Y1.shape[0], "X1 and Y1 disagree on spot count."
 
    seed = 0 if random_state is None else int(random_state)
    X0s, Y0s, p0s = _reduce_slice(X0, Y0, None, n_input, subsample, seed)
    X1s, Y1s, p1s = _reduce_slice(X1, Y1, None, n_input, subsample, seed + 1)
 
    C0 = cdist(X0s, X0s, metric="sqeuclidean")
    C1 = cdist(X1s, X1s, metric="sqeuclidean")
 
    if dtype is not None:
        C0, C1 = C0.astype(dtype), C1.astype(dtype)
        Y0s, Y1s = Y0s.astype(dtype), Y1s.astype(dtype)
 
    return {
        "X0s": X0s, "Y0s": Y0s, "p0s": p0s, "C0": C0,
        "X1s": X1s, "Y1s": Y1s, "p1s": p1s, "C1": C1,
    }
 
 
def impute_slice_fgw_barycenter(
    X0=None, Y0=None,
    X1=None, Y1=None,
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
    tol=1e-7,
    stop_criterion="barycenter",
    warmstartT=True,
    armijo=False,
    verbose=False,
    random_state=None,
    pd_B=None,
    Y_is_proportion=False,
    dtype=None,
    n_input=None,
    subsample="fps",
    use_gpu=False,
    prepared=None,
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
 
    Speed-oriented additions
    -------------------------
    prepared : dict, optional
        Output of `_prepare_fgw_inputs(...)`. If you're calling this function
        for several values of `t` on the same (X0, Y0, X1, Y1) pair, compute
        this once and pass it in every call to skip re-running `_reduce_slice`
        and the `cdist` cost-matrix construction — by far the most reusable
        (and often most expensive, for large raw inputs) part of the pipeline.
        When `prepared` is given, X0/Y0/X1/Y1/p0/p1/n_input/subsample/dtype
        are ignored (the values baked into `prepared` are used instead).
    use_gpu : bool, default False
        If True, move Cs/Ys/ps to a torch CUDA backend before calling
        `ot.gromov.fgw_barycenters`, letting POT dispatch to GPU ops. Requires
        torch with CUDA available. Silently falls back to CPU/numpy if not.
 
    Returns
    -------
    Y_is_proportion : (X_bar_spatial, genes_df, X_bar_feat, C_bar, log)
    otherwise       : (X_bar_spatial, X_bar_feat, C_bar, log)
    """
    if prepared is None:
        prepared = _prepare_fgw_inputs(
            X0, Y0, X1, Y1, n_input, subsample, dtype, random_state
        )
 
    X0s, Y0s, p0s, C0 = prepared["X0s"], prepared["Y0s"], prepared["p0s"], prepared["C0"]
    X1s, Y1s, p1s, C1 = prepared["X1s"], prepared["Y1s"], prepared["p1s"], prepared["C1"]
 
    if N_bary is None:
        N_bary = int(np.round(0.5 * (len(X0s) + len(X1s))))
 
    if lambdas is None:
        lambdas = [1.0 - t, t]
 
    Cs, Ys, ps = [C0, C1], [Y0s, Y1s], [p0s, p1s]
 
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                # Use float64, not float32: POT's warmstartT path re-checks
                # transport-plan marginals with np.testing.assert_allclose at
                # atol=1e-8, and float32 accumulates enough rounding error
                # over a few hundred/thousand entries to blow that tolerance
                # (a real failure we hit, not a hypothetical). float64 keeps
                # column/row sums accurate enough to pass, and is what fixes
                # the original "double != float" mismatch too, since the
                # mismatch was about *inconsistency* between tensors, not
                # about float64 itself being unsupported.
                Cs = [torch.as_tensor(c, device=device, dtype=torch.float64) for c in Cs]
                Ys = [torch.as_tensor(y, device=device, dtype=torch.float64) for y in Ys]
                ps = [torch.as_tensor(p, device=device, dtype=torch.float64) for p in ps]
        except ImportError:
            pass  # fall back to numpy/CPU silently
 
    # --- fGW barycenter
    X_bar_feat, C_bar, log = ot.gromov.fgw_barycenters(
        N=N_bary,
        Ys=Ys,
        Cs=Cs,
        ps=ps,
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
 
    # bring results back to numpy if we went to torch
    if use_gpu:
        try:
            import torch
            if isinstance(X_bar_feat, torch.Tensor):
                X_bar_feat = X_bar_feat.detach().cpu().numpy()
            if isinstance(C_bar, torch.Tensor):
                C_bar = C_bar.detach().cpu().numpy()
        except ImportError:
            pass
 
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
 