"""
simulate_realistic.py
=====================
Realistic synthetic data generator for ATLIS benchmarking — cluster-anchored.

Pipeline
--------
1. Leiden-cluster spots on V_card rows → assigns each spot a region label.
2. Assign a unique dominant cell type to each cluster (either by
   auto-picking from V_card's cluster averages with uniqueness enforced,
   or via an explicit `cluster_dominants` list you supply).
3. For each cluster, reconstruct V: `dom_frac` mass on dominant +
   `sec_frac` mass on second-dominant + `noise_frac` Dirichlet across
   the remaining types. (Defaults 0.5 / 0.2 / 0.3.)
4. Identify "cancer" clusters as those whose dominant matches the user-
   supplied `cancer_types` list.
5. At each slice t>0, every non-cancer spot adjacent to a cancer-cluster
   spot converts with probability `invasion_prob`. On conversion the
   spot inherits the cancer cluster's V profile.
6. Counts regenerated each slice via V @ B + NegBinom, calibrated to
   match real-data sparsity. Coordinates sine-warp + rotate + scale
   gradually with t.

Returns counts, ground-truth V per slice, and ground-truth cluster
labels per slice.
"""

import numpy as np
import pandas as pd
import anndata as ad


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sine_warp(coords, ax=0.03, ay=0.03, f=3.0, psi_x=0.0, psi_y=0.0):
    """
    Smooth nonlinear sine-warp:
        x' = x + ax*Lx*sin(2π f · y/Ly + ψ_y)
        y' = y + ay*Ly*sin(2π f · x/Lx + ψ_x)
    Spans are rescaled back to original after warping.
    """
    x, y = coords[0].copy(), coords[1].copy()
    Lx = x.max() - x.min()
    Ly = y.max() - y.min()

    xw = x + ax * Lx * np.sin(2 * np.pi * f * (y / Ly) + psi_y)
    yw = y + ay * Ly * np.sin(2 * np.pi * f * (x / Lx) + psi_x)

    def _rescale(w, orig):
        rng = w.max() - w.min()
        if rng > 0:
            w = (w - w.min()) / rng * (orig.max() - orig.min()) + orig.min()
        return w

    return np.vstack([_rescale(xw, x), _rescale(yw, y)])


def _build_neighbor_graph(coords, n_neighbors=6):
    """k-NN spatial graph from (2, n_spots) coords."""
    from scipy.spatial import cKDTree
    tree = cKDTree(coords.T)
    _, idxs = tree.query(coords.T, k=n_neighbors + 1)
    return [row[1:] for row in idxs]


# ─────────────────────────────────────────────────────────────────────────────
# Clustering on V
# ─────────────────────────────────────────────────────────────────────────────

def _leiden_cluster_V(V, n_clusters=4, seed=0, n_neighbors=15):
    """
    Cluster spots by their V row using Leiden, tuning resolution to hit
    the target n_clusters via binary search.
    """
    import scanpy as sc

    tmp = ad.AnnData(V.astype(np.float32))
    sc.pp.neighbors(tmp, n_neighbors=n_neighbors,
                    use_rep='X', random_state=seed)

    lo, hi = 0.05, 5.0
    best_labels = None
    best_k = -1
    for _ in range(25):
        mid = (lo + hi) / 2.0
        sc.tl.leiden(tmp, resolution=mid, random_state=seed,
                     key_added='leiden', flavor='igraph',
                     directed=False, n_iterations=2)
        labels = tmp.obs['leiden'].astype(int).values
        k = labels.max() + 1
        if k == n_clusters:
            best_labels = labels
            best_k = k
            break
        if best_labels is None or abs(k - n_clusters) < abs(best_k - n_clusters):
            best_labels = labels
            best_k = k
        if k < n_clusters:
            lo = mid
        else:
            hi = mid

    if best_k != n_clusters:
        print(f"  Warning: Leiden produced {best_k} clusters, "
              f"target was {n_clusters}. Using {best_k}.")

    return best_labels.astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Assigning a dominant cell type to each cluster
# ─────────────────────────────────────────────────────────────────────────────

def _assign_cluster_dominants(
    V_card, cluster_labels, type_names,
    cluster_dominants=None,
    dom_override=None, sec_override=None,
):
    """
    Decide a unique dominant cell type for each cluster, plus a
    (non-unique) second-dominant.

    Modes
    -----
    dom_override is a dict {cluster_id -> type index}
        Use it verbatim. Overrides every abundance-based rule below.
        This exists because `avg_V` is recomputed from whatever
        `cluster_labels` it is handed, so as cancer invasion changes
        cluster membership the abundance ranking drifts and both other
        modes can silently reassign dominants between time points.

    cluster_dominants is None (default — auto with uniqueness)
        Process clusters in descending order of how "peaked" their V
        argmax is. For each cluster, walk its V-rank order and pick
        the first cell type not yet taken by an earlier cluster.

    cluster_dominants is a list of cell-type names
        Greedy assignment. For each name in the list, find the
        unassigned cluster where that cell type has the highest mean V,
        assign it there. Any clusters left over (if the list is shorter
        than n_clusters) fall back to the auto rule.

    Returns
    -------
    dom_for_cluster, sec_for_cluster : dict {cluster_id (int) -> idx (int)}
    """
    K = V_card.shape[1]
    unique_clusters = sorted(int(c) for c in np.unique(cluster_labels))

    avg_V = {c: V_card[cluster_labels == c].mean(axis=0)
             for c in unique_clusters}

    # ── Explicit cluster→type map wins over any abundance-based rule. ──
    if dom_override is not None:
        dom_for_cluster = {c: int(dom_override[c]) for c in unique_clusters}
        if sec_override is not None:
            sec_for_cluster = {c: int(sec_override[c]) for c in unique_clusters}
        else:
            sec_for_cluster = {}
            for c in unique_clusters:
                order = np.argsort(avg_V[c])[::-1]
                dom = dom_for_cluster[c]
                sec_for_cluster[c] = next(
                    (int(i) for i in order if int(i) != dom), dom
                )
        return dom_for_cluster, sec_for_cluster

    dom_for_cluster = {}
    taken = set()

    if cluster_dominants is not None:
        name_to_idx = {n: i for i, n in enumerate(type_names)}
        missing = [n for n in cluster_dominants if n not in name_to_idx]
        if missing:
            raise ValueError(
                f"cluster_dominants contains unknown cell types: {missing}. "
                f"Valid options: {list(type_names)}"
            )
        if len(set(cluster_dominants)) != len(cluster_dominants):
            raise ValueError(
                f"cluster_dominants contains duplicates: {cluster_dominants}"
            )

        unassigned = set(unique_clusters)
        for name in cluster_dominants:
            if not unassigned:
                print(f"  Warning: more cluster_dominants than clusters; "
                      f"ignoring trailing entries starting at '{name}'.")
                break
            idx = name_to_idx[name]
            best_c = max(unassigned, key=lambda c: avg_V[c][idx])
            dom_for_cluster[best_c] = idx
            taken.add(idx)
            unassigned.remove(best_c)

        remaining = sorted(unassigned)
    else:
        remaining = list(unique_clusters)

    # Auto rule for clusters still without an assigned dominant.
    if remaining:
        # Peakiness = gap between #1 and #2 in cluster-average V.
        if K >= 2:
            peakiness = {c: avg_V[c].max() - np.partition(avg_V[c], -2)[-2]
                         for c in remaining}
        else:
            peakiness = {c: 0.0 for c in remaining}
        # Process most-peaked first so they get first pick.
        remaining = sorted(remaining, key=lambda c: -peakiness[c])

        for c in remaining:
            order = np.argsort(avg_V[c])[::-1]
            chosen = None
            for idx in order:
                if int(idx) not in taken:
                    chosen = int(idx)
                    break
            if chosen is None:
                # More clusters than cell types — shouldn't normally happen,
                # but fall back to the cluster's natural argmax (allowing
                # duplicate) so we never crash.
                chosen = int(order[0])
                print(f"  Warning: cluster {c} forced to share dominant "
                      f"({type_names[chosen]}) — more clusters than cell types.")
            dom_for_cluster[c] = chosen
            taken.add(chosen)

    # Second-dominant = highest V cell type that isn't this cluster's dominant.
    # Second identities are NOT forced unique across clusters.
    sec_for_cluster = {}
    for c in unique_clusters:
        order = np.argsort(avg_V[c])[::-1]
        dom = dom_for_cluster[c]
        sec = None
        for idx in order:
            if int(idx) != dom:
                sec = int(idx)
                break
        sec_for_cluster[c] = sec if sec is not None else dom

    return dom_for_cluster, sec_for_cluster


# ─────────────────────────────────────────────────────────────────────────────
# V reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def _reconstruct_V_from_clusters(
    V_card, cluster_labels, type_names,
    dom_frac=0.5, sec_frac=0.2, noise_frac=0.3,
    dirichlet_alpha=1.0, cluster_dominants=None, rng=None,
    dom_override=None, sec_override=None,
):
    """
    Build a structured V from cluster labels.

    For each cluster:
      - dom_frac mass on its (unique) dominant cell type
      - sec_frac mass on its second-dominant
      - noise_frac mass spread across all OTHER types via Dirichlet

    dom_override / sec_override : dict {cluster_id -> type index} or None
        Pin the cluster→type assignment instead of re-deriving it from
        cluster-average abundance. Pass the t=0 map here for t>0 slices.

    Returns
    -------
    V_new        : (n_spots, K) array, rows sum to 1
    cluster_info : dict {cluster_id -> {'dominant', 'second', 'n_spots'}}
    """
    if rng is None:
        rng = np.random.default_rng(0)

    total = dom_frac + sec_frac + noise_frac
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"dom_frac + sec_frac + noise_frac must sum to 1, got {total:.3f}"
        )

    n_spots, K = V_card.shape
    V_new = np.zeros_like(V_card, dtype=np.float32)
    cluster_info = {}

    dom_for_cluster, sec_for_cluster = _assign_cluster_dominants(
        V_card, cluster_labels, type_names,
        cluster_dominants=cluster_dominants,
        dom_override=dom_override, sec_override=sec_override,
    )

    for c in sorted(dom_for_cluster.keys()):
        mask = cluster_labels == c
        dom_idx = dom_for_cluster[c]
        sec_idx = sec_for_cluster[c]
        cluster_info[int(c)] = {
            'dominant': dom_idx,
            'second':   sec_idx,
            'n_spots':  int(mask.sum()),
        }

        other_idx = np.array([i for i in range(K)
                              if i != dom_idx and i != sec_idx])

        n_in = mask.sum()
        if other_idx.size > 0:
            alpha_vec   = np.full(other_idx.size, dirichlet_alpha)
            noise_props = rng.dirichlet(alpha_vec, size=n_in)
        else:
            noise_props = np.zeros((n_in, 0))

        cluster_V = np.zeros((n_in, K), dtype=np.float32)
        cluster_V[:, dom_idx] += dom_frac
        if sec_idx != dom_idx:
            cluster_V[:, sec_idx] += sec_frac
        else:
            cluster_V[:, dom_idx] += sec_frac
        if other_idx.size > 0:
            cluster_V[:, other_idx] += noise_frac * noise_props

        V_new[mask] = cluster_V

    V_new = V_new / V_new.sum(axis=1, keepdims=True)
    return V_new, cluster_info


# ─────────────────────────────────────────────────────────────────────────────
# Cluster-label propagation (cancer invasion)
# ─────────────────────────────────────────────────────────────────────────────

def _propagate_cancer_labels(
    cluster_labels, cancer_cluster_ids, neighbors,
    invasion_prob=0.3, rng=None,
):
    """
    One step of label propagation: every non-cancer spot adjacent to ≥1
    cancer-cluster neighbor converts with probability `invasion_prob`.
    If multiple cancer clusters are adjacent, the invading cluster is
    drawn proportionally to neighbor count.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    cancer_set = set(int(c) for c in cancer_cluster_ids)
    new_labels = cluster_labels.copy()
    is_cancer = np.array([lab in cancer_set for lab in cluster_labels])

    for i in range(len(cluster_labels)):
        if is_cancer[i]:
            continue
        nbr_labels = cluster_labels[neighbors[i]]
        cancer_nbrs = [int(l) for l in nbr_labels if int(l) in cancer_set]
        if not cancer_nbrs:
            continue
        if rng.random() < invasion_prob:
            uniq, counts = np.unique(cancer_nbrs, return_counts=True)
            probs = counts / counts.sum()
            new_labels[i] = rng.choice(uniq, p=probs)

    return new_labels


# ─────────────────────────────────────────────────────────────────────────────
# Count generation
# ─────────────────────────────────────────────────────────────────────────────

def _calibrate_lib_mean(V, B_matrix, target_sparsity, negbinom_r, lib_sigma, rng,
                        lo=0.5, hi=8.0, n_iter=20, n_sample=500):
    idx = rng.choice(V.shape[0], size=min(n_sample, V.shape[0]), replace=False)
    V_sub = V[idx]
    mean_expr = V_sub @ B_matrix

    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        lib = np.exp(mid + 0.5 * lib_sigma**2)
        lam = mean_expr * lib
        p   = negbinom_r / (negbinom_r + np.clip(lam, 1e-8, None))
        prob_zero = np.mean(p ** negbinom_r)
        if prob_zero < target_sparsity:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def _generate_counts(V, B_matrix, rng, negbinom_r=5.0,
                     lib_mean=8.0, lib_sigma=0.5):
    n_spots = V.shape[0]
    mean_expr    = V @ B_matrix
    library_size = rng.lognormal(lib_mean, lib_sigma, size=(n_spots, 1))
    lam          = mean_expr * library_size

    p          = negbinom_r / (negbinom_r + np.clip(lam, 1e-8, None))
    raw_counts = rng.negative_binomial(negbinom_r, p).astype(np.float32)

    totals     = raw_counts.sum(axis=1, keepdims=True)
    totals     = np.where(totals == 0, 1.0, totals)
    normalised = raw_counts / totals * 1e4
    return normalised, raw_counts


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def create_realistic_data(
    ref_adata,
    pd_B,
    V_card,
    n_time         = 5,
    # ── Clustering ──────────────────────────────────────────────────────
    n_clusters     = 4,
    leiden_neighbors = 15,
    # ── V reconstruction (fractions must sum to 1) ──────────────────────
    dom_frac       = 0.5,
    sec_frac       = 0.2,
    noise_frac     = 0.3,
    dirichlet_alpha = 1.0,
    cluster_dominants = None,     # list[str] of cell-type names (optional)
    # ── Cancer invasion ────────────────────────────────────────────────
    cancer_types   = None,        # cell-type names; clusters dominated by
                                  # any of these are treated as cancer
    invasion_prob  = 0.3,
    n_neighbors    = 6,
    # ── Geometry ───────────────────────────────────────────────────────
    warp_amplitude = 0.03,
    warp_freq      = 3.0,
    extra_scale    = 1.1,
    extra_rotate_deg = 25.0,
    # ── Count model ────────────────────────────────────────────────────
    negbinom_r     = 5.0,
    lib_mean       = 8.0,
    lib_sigma      = 0.5,
    seed           = 0,
):
    """
    Build a cluster-anchored multi-slice synthetic dataset.

    Parameters
    ----------
    ref_adata : AnnData
        Real slice, preprocessed (normalize_total 1e4 + filter_genes).
        Must have obs['x'], obs['y'].
    pd_B : pd.DataFrame
        Cell-type × gene matrix. Index = cell-type names matching V_card
        columns. Columns = gene names intersected with ref_adata.var_names.
    V_card : pd.DataFrame
        CARD result, spots × cell types. Used to seed Leiden clustering
        and to identify each cluster's dominant cell type.
    n_time : int
    n_clusters : int
        Target Leiden cluster count.
    leiden_neighbors : int
    dom_frac, sec_frac, noise_frac : float
        Must sum to 1. Default 0.5/0.2/0.3.
    dirichlet_alpha : float
        Concentration for the noise Dirichlet. <1 peakier, >1 flatter.
    cluster_dominants : list[str] or None
        Optional ordered list of cell-type names that will be each
        cluster's dominant identity. Greedy match to clusters by V
        abundance — list ORDER does not have to match Leiden cluster
        IDs. If None, dominants are auto-picked from V_card with
        uniqueness enforced (no two clusters share a dominant).
        Either way the resulting map is resolved once at t=0 and then
        held fixed for all later slices.
    cancer_types : list[str] or None
        Clusters dominated by any of these cell types are flagged as
        cancer and propagate via invasion. None disables invasion.
    invasion_prob : float
        P(convert) per slice for non-cancer spots with ≥1 cancer neighbor.
    n_neighbors : int
        k for the spatial k-NN graph used for label propagation.
    warp_amplitude, warp_freq, extra_scale, extra_rotate_deg :
        Geometric distortion, accumulates with t.
    negbinom_r, lib_sigma : NB model. lib_mean is auto-calibrated.
    seed : int

    Returns
    -------
    adata_list : list[AnnData]
        .X = UMI-normalised, .layers['counts'] = raw NB ints,
        .obs['x','y'] = warped coords, .obs['cluster'] = GT label,
        .obsm['V_true'] = GT composition.
    V_true_temporal : list[ndarray]
    cluster_labels_temporal : list[ndarray]
    ref_adata_out : AnnData (mean-centred, unwarped)
    pd_B_out : pd.DataFrame
    warp_params : list[dict]
    """
    rng = np.random.default_rng(seed)

    # ── 1. Align spots ───────────────────────────────────────────────────
    # shared_spots must be in ref_adata's own row order, and the subset must
    # be unconditional. Index.intersection() may return sorted output, which
    # would silently decouple V0 (shared_spots order) from ref_coords
    # (ref_adata order) whenever no spots are dropped.
    mask = ref_adata.obs_names.isin(V_card.index)
    if not mask.any():
        raise ValueError("ref_adata.obs_names and V_card.index share no spots.")

    shared_spots = ref_adata.obs_names[mask]
    n_dropped = ref_adata.n_obs - len(shared_spots)
    if n_dropped:
        print(f"  Warning: {n_dropped} spots in adata not in V_card — "
              f"subsetting to {len(shared_spots)}.")

    ref_adata = ref_adata[shared_spots].copy()
    assert (ref_adata.obs_names == shared_spots).all()

    V0 = V_card.loc[shared_spots, :].values.astype(np.float32)
    V0 /= V0.sum(axis=1, keepdims=True)

    n_spots    = ref_adata.n_obs
    type_names = list(V_card.columns)
    K          = len(type_names)

    # ── 2. Align genes ───────────────────────────────────────────────────
    shared_genes = pd_B.columns.intersection(ref_adata.var_names)
    if len(shared_genes) == 0:
        raise ValueError("pd_B columns and ref_adata.var_names share no genes.")
    pd_B      = pd_B[shared_genes].copy()
    B_matrix  = pd_B.values.astype(np.float64)
    gene_names = list(shared_genes)
    n_genes    = len(gene_names)
    print(f"  Spots: {n_spots}  |  Cell types: {K}  |  Genes: {n_genes}")

    # ── 3. Leiden cluster on V ───────────────────────────────────────────
    print(f"  Clustering V into {n_clusters} regions via Leiden...")
    base_cluster_labels = _leiden_cluster_V(
        V0, n_clusters=n_clusters, seed=seed,
        n_neighbors=leiden_neighbors,
    )

    # ── 4. Reconstruct V (handles dominants assignment) ──────────────────
    print(f"  Reconstructing V  (dom={dom_frac}, sec={sec_frac}, noise={noise_frac})")
    if cluster_dominants is not None:
        print(f"  Explicit dominants: {cluster_dominants}")
    V_structured, cluster_info = _reconstruct_V_from_clusters(
        V0, base_cluster_labels, type_names,
        dom_frac=dom_frac, sec_frac=sec_frac, noise_frac=noise_frac,
        dirichlet_alpha=dirichlet_alpha,
        cluster_dominants=cluster_dominants,
        rng=rng,
    )
    for c in sorted(cluster_info):
        info = cluster_info[c]
        print(f"    cluster {c}: dominant={type_names[info['dominant']]}, "
              f"second={type_names[info['second']]}, n={info['n_spots']}")

    # Freeze the t=0 cluster→type map. Invasion changes cluster membership,
    # which changes cluster-average abundance, which would otherwise let
    # _assign_cluster_dominants hand a cluster a different dominant at t>0.
    dom_map = {c: info['dominant'] for c, info in cluster_info.items()}
    sec_map = {c: info['second']   for c, info in cluster_info.items()}

    # ── 5. Resolve cancer clusters ───────────────────────────────────────
    cancer_cluster_ids = []
    if cancer_types is not None:
        missing = [t for t in cancer_types if t not in type_names]
        if missing:
            raise ValueError(f"cancer_types not in V_card columns: {missing}")
        cancer_type_idx = {type_names.index(t) for t in cancer_types}
        for c, info in cluster_info.items():
            if info['dominant'] in cancer_type_idx:
                cancer_cluster_ids.append(c)
        if not cancer_cluster_ids:
            raise ValueError(
                f"No cluster has any of {cancer_types} as dominant. "
                f"Either pass cluster_dominants=[...] to force one, or "
                f"increase n_clusters. "
                f"Current dominants: "
                f"{[type_names[cluster_info[c]['dominant']] for c in sorted(cluster_info)]}"
            )
        print(f"  Cancer clusters  : {cancer_cluster_ids}  "
              f"(invasion_prob={invasion_prob})")
    else:
        print("  Cancer invasion  : disabled")

    # ── 6. Coords + neighbor graph ───────────────────────────────────────
    ref_coords = np.vstack([ref_adata.obs['x'].values,
                            ref_adata.obs['y'].values]).astype(float)
    ref_coords -= ref_coords.mean(axis=1, keepdims=True)
    neighbors = _build_neighbor_graph(ref_coords, n_neighbors=n_neighbors)

    # ── 7. Calibrate lib_mean ────────────────────────────────────────────
    real_X = ref_adata[:, gene_names].X
    if hasattr(real_X, "toarray"):
        real_X = real_X.toarray()
    target_sparsity = float((np.asarray(real_X) == 0).mean())
    print(f"  Target sparsity  : {target_sparsity:.3f}")

    lib_mean = _calibrate_lib_mean(
        V_structured, B_matrix, target_sparsity,
        negbinom_r=negbinom_r, lib_sigma=lib_sigma, rng=rng,
    )
    print(f"  Calibrated lib_mean: {lib_mean:.3f}")

    # ── 8. Evolve cluster labels ─────────────────────────────────────────
    cluster_labels_temporal = [base_cluster_labels.copy()]
    current_labels = base_cluster_labels.copy()
    for t in range(1, n_time):
        if cancer_cluster_ids:
            current_labels = _propagate_cancer_labels(
                current_labels, cancer_cluster_ids, neighbors,
                invasion_prob=invasion_prob, rng=rng,
            )
        cluster_labels_temporal.append(current_labels.copy())

    # ── 9. Build slices ──────────────────────────────────────────────────
    adata_list      = []
    V_true_temporal = []
    warp_params     = []
    psi_x_fixed = rng.uniform(0, 2 * np.pi)
    psi_y_fixed = rng.uniform(0, 2 * np.pi)

    for t in range(n_time):
        if t == 0:
            V_true_t = V_structured
        else:
            V_true_t, _ = _reconstruct_V_from_clusters(
                V0, cluster_labels_temporal[t], type_names,
                dom_frac=dom_frac, sec_frac=sec_frac, noise_frac=noise_frac,
                dirichlet_alpha=dirichlet_alpha,
                rng=rng,
                dom_override=dom_map, sec_override=sec_map,
            )
        V_true_temporal.append(V_true_t)

        frac_warp = t / max(n_time - 1, 1)
        wp = dict(
            ax    = warp_amplitude * frac_warp,
            ay    = warp_amplitude * frac_warp,
            f     = warp_freq,
            psi_x = psi_x_fixed,
            psi_y = psi_y_fixed,
        )
        warp_params.append(wp)

        warped = (_sine_warp(ref_coords, **wp) if warp_amplitude > 0
                  else ref_coords.copy())

        scale_t = 1.0 + (extra_scale - 1.0) * frac_warp
        theta_t = np.deg2rad(extra_rotate_deg) * frac_warp
        R_t = np.array([[ np.cos(theta_t), np.sin(theta_t)],
                        [-np.sin(theta_t), np.cos(theta_t)]])
        warped = scale_t * (R_t @ warped)

        normalised, raw_counts = _generate_counts(
            V_true_t, B_matrix, rng,
            negbinom_r=negbinom_r, lib_mean=lib_mean, lib_sigma=lib_sigma,
        )

        adata_t                  = ad.AnnData(normalised)
        adata_t.layers['counts'] = raw_counts
        adata_t.obs['x']         = warped[0]
        adata_t.obs['y']         = warped[1]
        adata_t.obsm['spatial'] = warped.T.copy()   # (n_spots, 2)
        adata_t.obs['cluster']   = pd.Categorical(cluster_labels_temporal[t])
        adata_t.obs_names        = list(shared_spots)
        adata_t.var_names        = gene_names
        adata_t.obsm['V_true']        = V_true_t
        adata_t.uns['warp_params']    = wp
        adata_t.uns['B_true']         = B_matrix
        adata_t.uns['time_index']     = t
        adata_t.uns['cluster_info']   = cluster_info
        adata_t.uns['cancer_clusters']= cancer_cluster_ids

        adata_list.append(adata_t)
        n_cancer = sum(1 for l in cluster_labels_temporal[t]
                       if int(l) in set(cancer_cluster_ids))
        print(f"  t={t}  cancer spots={n_cancer}/{n_spots}  "
              f"V range [{V_true_t.min():.3f}, {V_true_t.max():.3f}]")

    # ── 10. Reference AnnData ────────────────────────────────────────────
    ref_adata_out                = ad.AnnData(adata_list[0].X.copy())
    ref_adata_out.obs['x']       = ref_coords[0]
    ref_adata_out.obs['y']       = ref_coords[1]
    ref_adata_out.obs['cluster'] = pd.Categorical(base_cluster_labels)
    ref_adata_out.obs_names      = list(shared_spots)
    ref_adata_out.var_names      = gene_names
    ref_adata_out.obsm['V_true'] = V_structured

    return (
        adata_list,
        V_true_temporal,
        cluster_labels_temporal,
        ref_adata_out,
        pd_B,
        warp_params,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def clustering_ari(V_true, V_pred):
    """ARI between argmax assignments of V_true and V_pred."""
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(V_true.argmax(1), V_pred.argmax(1))


def program_recovery_cosine(B_true, B_pred):
    """Best-permutation mean cosine similarity between rows of B_true and B_pred."""
    from itertools import permutations
    K   = B_true.shape[0]
    Bn  = lambda M: M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    C   = Bn(B_true) @ Bn(B_pred).T
    return max(
        np.mean([C[k, p[k]] for k in range(K)])
        for p in permutations(range(K))
    )


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Patch


def _colors(n):
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(n)]


def scatter_pies(
    ax, x, y, percents,
    radius=0.35, labels=None, colors=None,
    start_angle=90, edgecolor="white", lw=0.2,
    top_n=None, mode="full", leg_vis=True,
    keep_labels=None, others_label="Others", others_color=None,
):
    x = np.asarray(x); y = np.asarray(y)
    P = np.asarray(percents, float)
    P = np.nan_to_num(P, nan=0.0)
    n, k = P.shape
    if labels is None:
        labels = [f"C{i+1}" for i in range(k)]
    else:
        labels = list(labels)

    if keep_labels is not None:
        label_to_idx = {lab: i for i, lab in enumerate(labels)}
        keep_idx  = [label_to_idx[lb] for lb in keep_labels]
        drop_mask = np.ones(k, dtype=bool); drop_mask[keep_idx] = False
        drop_idx  = np.where(drop_mask)[0]
        P_keep    = P[:, keep_idx]
        P_others  = P[:, drop_idx].sum(axis=1, keepdims=True) if drop_idx.size > 0 else np.zeros((n, 1))
        P = np.concatenate([P_keep, P_others], axis=1)
        labels = keep_labels + [others_label]; k = P.shape[1]
        if colors is not None:
            new_colors = [colors[i] for i in keep_idx]
            if others_color is None:
                others_color = plt.get_cmap("tab20")(len(new_colors) % 20)
            new_colors.append(others_color); colors = new_colors

    rs = P.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0; P = P / rs

    if top_n is not None and top_n < k:
        avg = P.mean(axis=0)
        keep_idx = np.argsort(avg)[::-1][:top_n]
        P = P[:, keep_idx]; labels = [labels[i] for i in keep_idx]; k = top_n
        colors = [colors[i] for i in keep_idx] if colors is not None else None

    if colors is None:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % 20) for i in range(k)]

    if mode == "full":
        for xi, yi, fracs in zip(x, y, P):
            theta = start_angle
            for f, c in zip(fracs, colors):
                if f <= 0: continue
                ax.add_patch(Wedge((xi, yi), r=radius,
                                   theta1=theta, theta2=theta + 360.0 * f,
                                   facecolor=c, edgecolor=edgecolor, linewidth=lw))
                theta += 360.0 * f
    elif mode == "argmax":
        max_idx = np.argmax(P, axis=1)
        for xi, yi, j in zip(x, y, max_idx):
            ax.add_patch(Wedge((xi, yi), r=radius, theta1=0, theta2=360,
                               facecolor=colors[j], edgecolor=edgecolor, linewidth=lw))

    ax.set_xlim(x.min() - radius * 1.1, x.max() + radius * 1.1)
    ax.set_ylim(y.min() - radius * 1.1, y.max() + radius * 1.1)
    ax.set_aspect("equal", adjustable="box")

    handles = [Patch(facecolor=colors[i], edgecolor=edgecolor, label=labels[i])
               for i in range(k)]
    if leg_vis:
        ax.legend(handles=handles, title="Cell types", frameon=False,
                  loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    return handles


def plot_domain_maps(adata_list, V_true_temporal, pd_B,
                     max_slices=6, radius=0.35, mode="full", figsize=None):
    """Per-slice pie chart visualisation of V_true."""
    n_show     = min(len(adata_list), max_slices)
    type_names = list(pd_B.index)
    colors     = _colors(len(type_names))

    ncols = min(n_show, 5)
    nrows = int(np.ceil(n_show / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=figsize or (7 * ncols, 6.5 * nrows))
    axes = np.array(axes).ravel()

    handles = None
    for t in range(n_show):
        ax = axes[t]
        ad_t = adata_list[t]
        handles = scatter_pies(
            ax,
            x=ad_t.obs['x'].values, y=ad_t.obs['y'].values,
            percents=V_true_temporal[t],
            radius=radius, labels=type_names, colors=colors,
            mode=mode, leg_vis=False,
        )
        ax.set_title(f"Slice  t = {t}", fontsize=11)
        ax.axis('off')

    if handles:
        fig.legend(handles=handles, title="Cell types", frameon=False,
                   loc="lower center", ncol=min( len(type_names),5),
                   bbox_to_anchor=(0.5, -0.1), fontsize=16)

    for ax in axes[n_show:]:
        ax.axis('off')

    fig.suptitle(f"Ground-truth compositions  (mode='{mode}')", fontsize=13)
    fig.tight_layout()
    return fig


def plot_cluster_maps(adata_list, max_slices=6, figsize=None, s=10,
                      type_names=None):
    """Per-slice scatter coloured by ground-truth cluster label."""
    n_show = min(len(adata_list), max_slices)
    all_labels = set()
    for adt in adata_list[:n_show]:
        all_labels.update(np.asarray(adt.obs['cluster']).astype(int).tolist())
    all_labels = sorted(int(l) for l in all_labels)
    color_map = {lab: _colors(len(all_labels))[i] for i, lab in enumerate(all_labels)}

    ncols = min(n_show, 5)
    nrows = int(np.ceil(n_show / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=figsize or (5 * ncols, 5 * nrows))
    axes = np.array(axes).ravel()

    cancer_clusters = set()
    if 'cancer_clusters' in adata_list[0].uns:
        cancer_clusters = set(int(c) for c in adata_list[0].uns['cancer_clusters'])

    # Build cluster→dominant-name lookup for legend, if available.
    dom_name = {}
    if type_names is not None and 'cluster_info' in adata_list[0].uns:
        for c, info in adata_list[0].uns['cluster_info'].items():
            dom_name[int(c)] = type_names[info['dominant']]

    for t in range(n_show):
        ax = axes[t]
        adt = adata_list[t]
        labs = np.asarray(adt.obs['cluster']).astype(int)
        c = [color_map[int(l)] for l in labs]
        ax.scatter(adt.obs['x'].values, adt.obs['y'].values, c=c, s=s,
                   linewidths=0, rasterized=True)
        n_cancer = int(np.isin(labs, list(cancer_clusters)).sum()) if cancer_clusters else 0
        title = f"t = {t}"
        if cancer_clusters:
            title += f"   cancer={n_cancer}/{len(labs)}"
        ax.set_title(title, fontsize=11)
        ax.set_aspect('equal'); ax.axis('off')

    def _legend_label(lab):
        s = f"cluster {lab}"
        if lab in dom_name:
            s += f": {dom_name[lab]}"
        if lab in cancer_clusters:
            s += "  (cancer)"
        return s

    handles = [Patch(facecolor=color_map[lab], label=_legend_label(lab))
               for lab in all_labels]
    fig.legend(handles=handles, frameon=False, loc='lower center',
               ncol=len(all_labels), bbox_to_anchor=(0.5, -0.02),
               fontsize=16)

    for ax in axes[n_show:]:
        ax.axis('off')
    fig.suptitle("Ground-truth cluster labels", fontsize=13)
    fig.tight_layout()
    return fig




def visualize_all(adata_list, V_true_temporal, pd_B, warp_params, ref_adata,
                  radius=0.35, mode="full", save_prefix=None,
                  view="composition", **kwargs):
    """
    Generate all visualisation figures.

    Parameters
    ----------
    view : {'composition', 'clusters', 'both'}
        - 'composition' (default): per-slice spatial map shows pie-chart
          cell-type compositions from V_true.
        - 'clusters': per-slice spatial map shows ground-truth cluster
          labels as flat scatter colours.
        - 'both': include both 'domains' (composition pies) and
          'clusters' (label scatter) figures.
        The 'warp' and 'counts' figures are always included.
    radius : float
        Pie radius for composition mode — tune to your coordinate scale.
    mode : {'full', 'argmax'}
        Passed through to scatter_pies when view includes composition.
    save_prefix : str or None
        If given, saves {save_prefix}_{name}.png for each figure.
    """
    if view not in ('composition', 'clusters', 'both'):
        raise ValueError(
            f"view must be 'composition', 'clusters', or 'both'; got '{view}'"
        )

    figs = {}
    type_names = list(pd_B.index)

    if view in ('composition', 'both'):
        figs['domains'] = plot_domain_maps(
            adata_list, V_true_temporal, pd_B,
            radius=radius, mode=mode,
        )
    if view in ('clusters', 'both'):
        figs['clusters'] = plot_cluster_maps(adata_list, type_names=type_names, s=30)



    if save_prefix:
        for name, fig in figs.items():
            path = f"{save_prefix}_{name}.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved  {path}")
    return figs