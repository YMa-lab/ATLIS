import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import anndata as ad
import scanpy as sc

from scipy.spatial import distance
import ot
from sklearn.decomposition import PCA

# Part 1: Domain identification metrics. 


def _build_graph(V, n_neighbors=15, random_state=0):
    tmp = ad.AnnData(X=np.asarray(V))
    sc.pp.neighbors(tmp, n_neighbors=n_neighbors, use_rep=None,
                    random_state=random_state)
    return tmp


def _cluster_at_resolution(tmp, method="louvain", resolution=1.0, random_state=0):
    key = "clusters"
    if method == "louvain":
        sc.tl.louvain(tmp, resolution=resolution, key_added=key,
                      random_state=random_state)
    elif method == "leiden":
        sc.tl.leiden(tmp, resolution=resolution, key_added=key,
                     random_state=random_state,
                     flavor="igraph", n_iterations=2, directed=False)
    else:
        raise ValueError("method must be 'louvain' or 'leiden'")
    
    labels = tmp.obs[key].astype("category").cat.codes.to_numpy()
    print('clustering at resolution='+str(resolution)+' has been done! Found '+str(len(set(labels)))+' clusters.')
    return labels, np.unique(labels).size


# def _run_graph_clustering(V, method="louvain", n_neighbors=15,
#                           resolution=1.0, random_state=0):
#     """Kept for the fixed-resolution path; builds the graph and clusters once."""
#     tmp = _build_graph(V, n_neighbors=n_neighbors, random_state=random_state)
#     print('neighborhood graph has been built!')
#     return _cluster_at_resolution(tmp, method=method, resolution=resolution,
#                                   random_state=random_state)


def _find_resolution_for_target_clusters(
    V,
    method,
    target_n_clusters,
    n_neighbors=15,
    random_state=0,
    resolutions=None,
):
    """
    This function does a binary search over resolutions to find the clustering result that returns (close to) target # clusters.

    Returns (best_labels, best_resolution, best_n_clusters).
    """
    if resolutions is None:
        # --- your original grid logic (unchanged) ---
        resolutions = np.unique(np.r_[np.logspace(-2, 0, 20), np.logspace(0, 2, 50)])
        if target_n_clusters < 20:
            resolutions = np.linspace(0.01, 1.2, 100)
        elif target_n_clusters < 30:
            resolutions = np.linspace(0.5, 1.5, 100)
        else:
            resolutions = np.linspace(0.5, 3.0, 100)

    # Use ONLY the ends you chose via `resolutions`
    res_lo = float(resolutions[0])
    res_hi = float(resolutions[-1])
    tmp = _build_graph(V, n_neighbors=n_neighbors, random_state=random_state)


    def run(res):
        return _cluster_at_resolution(
            tmp,
            method=method,
            resolution=res,
            random_state=random_state,
        )

    # Evaluate endpoints
    labels_lo, n_lo = run(res_lo)
    labels_hi, n_hi = run(res_hi)

    # Track best-so-far (same objective as your scan: closest to target)
    best_labels = labels_lo
    best_res = res_lo
    best_n_clust = n_lo
    best_diff = abs(n_lo - target_n_clusters)

    def update(labels, res, n_clust):
        nonlocal best_labels, best_res, best_n_clust, best_diff
        diff = abs(n_clust - target_n_clusters)
        if diff < best_diff:
            best_diff = diff
            best_labels = labels
            best_res = res
            best_n_clust = n_clust

    update(labels_hi, res_hi, n_hi)

    # If the target is not between endpoint cluster counts, bisection cannot help.
    if not (min(n_lo, n_hi) <= target_n_clusters <= max(n_lo, n_hi)):
        return best_labels, best_res, best_n_clust

    # Ensure we bisect in a direction where n_clust is increasing with res.
    # (If your method happens to produce n_lo > n_hi, swap ends.)
    if n_lo > n_hi:
        res_lo, res_hi = res_hi, res_lo
        n_lo, n_hi = n_hi, n_lo

    # Bisection
    max_iter = 25
    tol_res = 1e-3

    for _ in range(max_iter):
        res_mid = 0.5 * (res_lo + res_hi)
        labels_mid, n_mid = run(res_mid)
        update(labels_mid, res_mid, n_mid)

        if n_mid == target_n_clusters:
            return labels_mid, res_mid, n_mid

        if n_mid < target_n_clusters:
            res_lo, n_lo = res_mid, n_mid
        else:
            res_hi, n_hi = res_mid, n_mid

        if abs(res_hi - res_lo) < tol_res:
            break
        if n_lo == n_hi:
            break  # plateau/saturation

    return best_labels, best_res, best_n_clust

def cluster_compute_ari_and_plot(
    V,
    adata,
    celltype_key="cell_type",
    embedding_key="spatial",
    clustering="louvain",
    n_clusters=None,
    n_neighbors=15,
    resolution=1.0,                 # <--- default fixed resolution
    auto_resolution=False,          
    resolution_grid=None,
    random_state=0,
    figsize=(12, 5),
    plot=True,
):
    V = np.asarray(V)
    true_labels = adata.obs[celltype_key].astype("category").cat.codes.to_numpy()

    if V.shape[0] != len(true_labels):
        raise ValueError("Row count in V must match number of cells in adata.")

    clustering = clustering.lower()

    # ------------------------------------------------------------------
    # KMEANS
    # ------------------------------------------------------------------
    if clustering == "kmeans":
        if n_clusters is None:
            n_clusters = np.unique(true_labels).size

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto"
        )
        pred_labels = kmeans.fit_predict(V)

    # ------------------------------------------------------------------
    # GRAPH CLUSTERING (Louvain / Leiden)
    # ------------------------------------------------------------------
    elif clustering in {"louvain", "leiden"}:

        method = clustering

        # -------- AUTO RESOLUTION SEARCH --------
        if auto_resolution:

            target_k = n_clusters if n_clusters is not None else np.unique(true_labels).size

            pred_labels, best_res, n_clust = _find_resolution_for_target_clusters(
                    V,
                    method=method,
                    target_n_clusters=target_k,
                    n_neighbors=n_neighbors,
                    random_state=random_state,
                    resolutions=resolution_grid,
            )

            print(f"{method.capitalize()}: target_k={target_k}, "
                      f"best_resolution={best_res:.3f}, n_clusters={n_clust}")


        # -------- FIXED RESOLUTION (DEFAULT) --------
        else:
            tmp = _build_graph(V, n_neighbors=n_neighbors, random_state=random_state)

            pred_labels, n_clust = _cluster_at_resolution(
                tmp,
                method=method,
                # n_neighbors=n_neighbors,
                resolution=resolution,
                random_state=random_state,
            )

            print(f"{method.capitalize()}: resolution={resolution:.3f}, "
                  f"n_clusters={n_clust}")

    else:
        raise ValueError(f"Unknown clustering method: {clustering}")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    if plot:
        emb = adata.obsm[embedding_key]
        x, y = emb[:, 0], emb[:, 1]

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].scatter(x, y, c=true_labels, cmap="tab20c", s=8)
        axes[0].set_title("True labels\n" + celltype_key)

        axes[1].scatter(x, y, c=pred_labels, cmap="tab20c", s=8)
        axes[1].set_title(f"{clustering.capitalize()} clusters\n(ARI = {ari:.3f})")

        plt.tight_layout()
        plt.show()

    return ari, nmi, true_labels, pred_labels

# Part 2: Optimal transport-based metrics. Retrieved from DeST-OT GitHub.


import numpy as np
import pandas as pd

def same_label_mass_only(
    adata_insamp,
    pi,
    celltype_col="annotation",
    time_labels=("E12.5", "E14.5", "E16.5"),
    normalize_each_step=True,
):
    def _is_sparse(x):
        try:
            import scipy.sparse as sp
            return sp.issparse(x)
        except Exception:
            return False

    def _aggregate_pi_by_class(ad0, ad1, P):
        ct0 = ad0.obs[celltype_col].astype(str).values
        ct1 = ad1.obs[celltype_col].astype(str).values
        cat0 = pd.Categorical(ct0)
        cat1 = pd.Categorical(ct1)

        idx0 = cat0.codes
        idx1 = cat1.codes
        src_cats = list(cat0.categories)
        tgt_cats = list(cat1.categories)
        K0, K1 = len(src_cats), len(tgt_cats)

        F = np.zeros((K0, K1), dtype=float)

        if _is_sparse(P):
            P = P.tocoo()
            for i, j, v in zip(P.row, P.col, P.data):
                F[idx0[i], idx1[j]] += float(v)
        else:
            A = np.asarray(P, dtype=float)
            for k in range(K0):
                rows = np.where(idx0 == k)[0]
                if rows.size == 0:
                    continue
                s = A[rows, :].sum(axis=0)
                np.add.at(F[k, :], idx1, s)

        return src_cats, tgt_cats, F

    # --- sanity ---
    T = len(adata_insamp)
    assert len(pi) == T - 1
    if time_labels is None:
        time_labels = [f"t{t}" for t in range(T)]
    assert len(time_labels) == T

    for t in range(T - 1):
        if pi[t].shape != (adata_insamp[t].n_obs, adata_insamp[t + 1].n_obs):
            raise ValueError(f"Mismatch at t={t}: pi[{t}].shape={pi[t].shape}")

    out = []

    for t in range(T - 1):
        P = pi[t]
        if normalize_each_step:
            s = float(P.sum())
            if s > 0:
                P = P / s

        src_cats, tgt_cats, F = _aggregate_pi_by_class(adata_insamp[t], adata_insamp[t + 1], P)

        # index maps
        src_index = {c: i for i, c in enumerate(src_cats)}
        tgt_index = {c: j for j, c in enumerate(tgt_cats)}
        shared = sorted(set(src_cats).intersection(set(tgt_cats)))

        total_mass = float(F.sum())

        # observed diagonal mass over shared labels
        diag_mass = 0.0
        for lab in shared:
            diag_mass += float(F[src_index[lab], tgt_index[lab]])

        same_frac = diag_mass / total_mass if total_mass > 0 else np.nan

        # baseline expected diagonal under independence given marginals
        row_marg = F.sum(axis=1)  # shape (K0,)
        col_marg = F.sum(axis=0)  # shape (K1,)
        expected_diag = 0.0
        for lab in shared:
            expected_diag += float(row_marg[src_index[lab]] * col_marg[tgt_index[lab]])

        # enrichment ratio
        enrichment = (diag_mass / expected_diag) if expected_diag > 0 else np.nan

        # adjusted-for-chance (bounded, like "chance-corrected retention")
        adjusted = ((diag_mass - expected_diag) / (1.0 - expected_diag)) if expected_diag < 1 else np.nan

        out.append(
            dict(
                transition=f"{time_labels[t]}→{time_labels[t+1]}",
                total_mass=total_mass,
                same_label_mass=diag_mass,
                same_label_fraction=same_frac,
                expected_same_label_mass=expected_diag,
                enrichment_over_expected=enrichment,
                adjusted_same_label=adjusted,
                n_shared_labels=len(shared),
                n_src_labels=len(src_cats),
                n_tgt_labels=len(tgt_cats),
            )
        )

    return pd.DataFrame(out)

import numpy as np
from scipy.spatial.distance import cdist

def expected_expression_change(slice_t1, slice_t2, Pi):
    """
    E1: (N1, d) expression embedding at t1
    E2: (N2, d) expression embedding at t2
    Pi: (N1, N2) coupling
    Returns: mean_i E_{j~Pi_hat(i,:)} [ ||E2[j]-E1[i]|| ]
    """
    E1 = slice_t1.X.todense()
    E2 = slice_t2.X.todense()
    row_sums = Pi.sum(axis=1)
    ma = row_sums > 0
    Pi_hat = Pi[ma] / row_sums[ma][:, None]          # (N1_kept, N2)
    D = cdist(E2, E1[ma])                            # (N2, N1_kept), distances
    deltas = np.einsum("ij,ji->i", Pi_hat, D)        # (N1_kept,)
    return float(np.mean(deltas))


def migration_metric(slice_t1, slice_t2, Pi):
    """
    Compute the migration metric given two slices and an aligment matrix Pi

    Parameters:
    slice_t1: AnnData object
        An AnnData object of slice t1, with .obsm['spatial'] field storing the spatial coordinates.
    slice_t2: AnnData object
        An AnnData object of slice t2, with .obsm['spatial'] field storing the spatial coordinates.
    Pi: numpy array of shape (N1 x N2)
        Alignment matrix between slice t1 and slice t2

    Returns:
    The migration metric of Pi
    """
    slice_t1_newcoor_adata, slice_t2_newcoor_adata = partial_stack_slices_pairwise([slice_t1, slice_t2], [Pi])
    
    slice_t1_newcoor = slice_t1_newcoor_adata.obsm['spatial']
    slice_t2_newcoor = slice_t2_newcoor_adata.obsm['spatial']
    
    V = distance.cdist(slice_t2_newcoor, slice_t1_newcoor)
    
    # Handling zero entries in Pi @ 1_n (i.e. spot not mapped at all to next timepoint)
    ma = (Pi @ np.ones(Pi.shape[1])) > 0

    Pi_hat = Pi[ma, :] / (Pi @ np.ones(Pi.shape[1]))[ma][:, None]
    # Pi_hat = pi / (pi @ np.ones(pi.shape[1]))[:, None]
    V = V[:,ma]

    migs = np.einsum('ij,ji->i', Pi_hat, V)
    
    avg_weighted_migration_distance = np.mean(migs)
    return avg_weighted_migration_distance


def partial_procrustes_analysis(X, Y, pi):
    m = np.sum(pi)
    Z = (X - pi.sum(axis=1).dot(X) * (1.0 / m)).T
    W = (Y - pi.sum(axis=0).dot(Y) * (1.0 / m)).T
    H = W.dot(pi.T.dot(Z.T))
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T.dot(U.T)
    W = R.dot(W)
    return Z.T, W.T

def partial_stack_slices_pairwise(slices, pis):
    assert len(slices) == len(pis) + 1, "'slices' should have length one more than 'pis'. Please double check."
    assert len(slices) > 1, "You should have at least 2 layers."

    new_coor = []
    S1, S2 = partial_procrustes_analysis(slices[0].obsm['spatial'], slices[1].obsm['spatial'], pis[0])
    new_coor.append(S1)
    new_coor.append(S2)
    for i in range(1, len(slices) - 1):
        x, y = partial_procrustes_analysis(new_coor[i], slices[i + 1].obsm['spatial'], pis[i])
        shift = new_coor[i][0,:] - x[0,:]
        y = y + shift
        new_coor.append(y)
    new_slices = []
    for i in range(len(slices)):
        s = slices[i].copy()
        s.obsm['spatial'] = new_coor[i]
        new_slices.append(s)
    return new_slices


# Part 3: Whole slice imputation accuracy scores.


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def as_array(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X, float)
 
 
def positive_median(D):
    v = D[np.isfinite(D) & (D > 0)]
    return float(np.median(v)) if v.size else 1.0
 
 
def get_spatial(a):
    if "spatial" in a.obsm:
        return np.asarray(a.obsm["spatial"], float)
    return a.obs[["x", "y"]].to_numpy(float)
 
 
def subsample(n, max_spots, seed):
    """Index array of at most `max_spots` rows out of n (None = keep all)."""
    if n <= max_spots:
        return None
    return np.random.default_rng(seed).choice(n, max_spots, replace=False)
 
 
def take(idx, *arrays):
    return arrays if idx is None else tuple(a[idx] for a in arrays)
 
 
def expression_scale(X, max_spots=500, seed=0):
    """Median pairwise expression distance in truth — sets the cost unit."""
    (X,) = take(subsample(len(X), max_spots, seed), X)
    return positive_median(cdist(X, X))
 
 
def normalized_spatial_cost(S):
    C = cdist(S, S)
    return C / positive_median(C)
 
 
def uniform(n):
    return np.full(n, 1.0 / n)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Shared low-dimensional embedding
# ─────────────────────────────────────────────────────────────────────────────
def fit_embedding(X_true, *X_preds, n_pcs=30, log1p=True, seed=0):
    """Project truth and any number of predictions onto PCs fitted on TRUTH ONLY.
 
    Fitting on truth keeps the basis neutral: no method gets a subspace tuned
    to its own output. Raw sparse counts make Euclidean distance degenerate
    (near-orthogonal spots, distances driven by dropout), which breaks the OT
    scores; a shared dense embedding fixes that.
 
    Pass several predictions at once so competing methods share one basis.
    n_pcs=None passes the inputs through unchanged.
 
    Returns (Z_true, Z_pred_1, ..., Z_pred_k).
    """
    if n_pcs is None:
        return (X_true, *X_preds)
 
    prep = (lambda X: np.log1p(np.maximum(X, 0))) if log1p else (lambda X: X)
 
    Tt = prep(X_true)
    n_comp = int(min(n_pcs, Tt.shape[0] - 1, Tt.shape[1]))
    if n_comp < 2:
        raise ValueError(f"Not enough spots/genes for {n_pcs} PCs.")
 
    pca = PCA(n_components=n_comp, random_state=seed).fit(Tt)
    return (pca.transform(Tt), *(pca.transform(prep(X)) for X in X_preds))
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Individual OT scores (all lower = better)
# ─────────────────────────────────────────────────────────────────────────────
def expression_ot_score(X_true, X_pred, max_spots=500, seed=0, numItermaxEmd=1_000_000):
    """Wasserstein distance between expression clouds, ignoring coordinates."""
    scale = expression_scale(X_true, max_spots, seed)
    (X_true,) = take(subsample(len(X_true), max_spots, seed), X_true)
    (X_pred,) = take(subsample(len(X_pred), max_spots, seed), X_pred)
 
    M = cdist(X_true, X_pred) / scale
    return float(
        ot.emd2(uniform(len(X_true)), uniform(len(X_pred)), M, numItermax=numItermaxEmd)
    )
 
 
def gw_score(spatial_true, spatial_pred, max_spots=500, seed=0, numItermaxEmd=1_000_000):
    """Gromov–Wasserstein discrepancy between the two spot geometries."""
    (spatial_true,) = take(subsample(len(spatial_true), max_spots, seed), spatial_true)
    (spatial_pred,) = take(subsample(len(spatial_pred), max_spots, seed), spatial_pred)
 
    return float(
        ot.gromov.gromov_wasserstein2(
            normalized_spatial_cost(spatial_true),
            normalized_spatial_cost(spatial_pred),
            p=uniform(len(spatial_true)),
            q=uniform(len(spatial_pred)),
            loss_fun="square_loss",
            numItermaxEmd=numItermaxEmd,
        )
    )
 
 
def fused_gw_score(
    X_true, spatial_true, X_pred, spatial_pred,
    alpha=0.5, max_spots=500, seed=0, numItermaxEmd=1_000_000,
):
    """Median-normalized fused GW: expression cost + geometry cost."""
    scale = expression_scale(X_true, max_spots, seed)
 
    X_true, spatial_true = take(
        subsample(len(X_true), max_spots, seed), X_true, spatial_true
    )
    X_pred, spatial_pred = take(
        subsample(len(X_pred), max_spots, seed), X_pred, spatial_pred
    )
 
    M = cdist(X_true, X_pred) / scale
    return float(
        ot.gromov.fused_gromov_wasserstein2(
            M,
            normalized_spatial_cost(spatial_true),
            normalized_spatial_cost(spatial_pred),
            p=uniform(len(X_true)),
            q=uniform(len(X_pred)),
            loss_fun="square_loss",
            alpha=alpha,
            numItermaxEmd=numItermaxEmd,
        )
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Entry point: score one prediction against one truth
# ─────────────────────────────────────────────────────────────────────────────
def score_imputation(
    X_true,
    spatial_true,
    X_pred,
    spatial_pred,
    alpha=0.5,
    max_spots=500,
    seed=0,
    numItermaxEmd=1_000_000,
    n_pcs=30,
    pca_log1p=True,
    embed=True,
):
    """All three OT distances between one prediction and the truth.
 
    X_* are spot x feature matrices on the SAME feature set; spatial_* are
    spot x 2 coordinates. Pass `embed=False` if X_* are already PC scores from
    a shared `fit_embedding` call — useful when scoring several methods against
    the same truth, so they all sit in one basis.
 
    Returns {"expression": ..., "spatial": ..., "fused": ...}. Lower is better
    for all three.
    """
    X_true = as_array(X_true)
    X_pred = as_array(X_pred)
    spatial_true = np.asarray(spatial_true, float)
    spatial_pred = np.asarray(spatial_pred, float)
 
    if X_true.shape[1] != X_pred.shape[1]:
        raise ValueError(
            f"Feature mismatch: truth has {X_true.shape[1]}, "
            f"prediction has {X_pred.shape[1]}."
        )
    for name, X, S in (
        ("truth", X_true, spatial_true), ("prediction", X_pred, spatial_pred)
    ):
        if len(X) != len(S):
            raise ValueError(
                f"{name}: {len(X)} expression rows vs {len(S)} coordinates."
            )
 
    if embed:
        X_true, X_pred = fit_embedding(
            X_true, X_pred, n_pcs=n_pcs, log1p=pca_log1p, seed=seed
        )
 
    common = dict(max_spots=max_spots, seed=seed, numItermaxEmd=numItermaxEmd)
    return {
        "expression": expression_ot_score(X_true, X_pred, **common),
        "spatial": gw_score(spatial_true, spatial_pred, **common),
        "fused": fused_gw_score(
            X_true, spatial_true, X_pred, spatial_pred, alpha=alpha, **common
        ),
    }
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Convenience: line up genes across truth, prediction, and a basis matrix
# ─────────────────────────────────────────────────────────────────────────────
def common_genes(*var_name_sets, exclude_prefixes=None):
    """Intersect gene name indices, optionally dropping unwanted prefixes."""
    genes = var_name_sets[0]
    for other in var_name_sets[1:]:
        genes = genes.intersection(other)
 
    if exclude_prefixes and len(genes):
        keep = np.array(
            [not str(g).startswith(tuple(exclude_prefixes)) for g in genes], bool
        )
        genes = genes[keep]
 
    if len(genes) == 0:
        raise ValueError("No common genes remain after filtering.")
    return genes