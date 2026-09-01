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

# def _sine_warp(coords, ax=0.03, ay=0.03, f=3.0, psi_x=0.0, psi_y=0.0):
#     x, y = coords[0].copy(), coords[1].copy()
#     Lx = x.max() - x.min(); Ly = y.max() - y.min()
#     xw = x + ax * Lx * np.sin(2*np.pi*f*(y/Ly) + psi_y)
#     yw = y + ay * Ly * np.sin(2*np.pi*f*(x/Lx) + psi_x)
#     def _rescale(w, orig):
#         rng = w.max() - w.min()
#         if rng > 0:
#             w = (w - w.min())/rng*(orig.max()-orig.min()) + orig.min()
#         return w
#     return np.vstack([_rescale(xw, x), _rescale(yw, y)])

def _sine_warp(coords, ax=0.03, ay=0.03, f=3.0, psi_x=0.0, psi_y=0.0):
    x, y = coords[0].copy(), coords[1].copy()
    Lx = x.max() - x.min(); Ly = y.max() - y.min()
    xw = x + ax * Lx * np.sin(2*np.pi*f*(y/Ly) + psi_y)
    yw = y + ay * Ly * np.sin(2*np.pi*f*(x/Lx) + psi_x)
    return np.vstack([xw, yw])

def _build_neighbor_graph(coords, n_neighbors=6):
    from scipy.spatial import cKDTree
    tree = cKDTree(coords.T)
    _, idxs = tree.query(coords.T, k=n_neighbors+1)
    return [row[1:] for row in idxs]

def _assign_cluster_dominants(V_card, cluster_labels, type_names,
                              cluster_dominants=None, dom_override=None, sec_override=None):
    K = V_card.shape[1]
    unique_clusters = sorted(int(c) for c in np.unique(cluster_labels))
    avg_V = {c: V_card[cluster_labels == c].mean(axis=0) for c in unique_clusters}
    if dom_override is not None:
        dom_for_cluster = {c: int(dom_override[c]) for c in unique_clusters}
        if sec_override is not None:
            sec_for_cluster = {c: int(sec_override[c]) for c in unique_clusters}
        else:
            sec_for_cluster = {}
            for c in unique_clusters:
                order = np.argsort(avg_V[c])[::-1]; dom = dom_for_cluster[c]
                sec_for_cluster[c] = next((int(i) for i in order if int(i) != dom), dom)
        return dom_for_cluster, sec_for_cluster
    dom_for_cluster = {}; taken = set()
    if cluster_dominants is not None:
        name_to_idx = {n: i for i, n in enumerate(type_names)}
        missing = [n for n in cluster_dominants if n not in name_to_idx]
        if missing: raise ValueError(f"unknown cell types: {missing}")
        if len(set(cluster_dominants)) != len(cluster_dominants):
            raise ValueError("duplicates")
        unassigned = set(unique_clusters)
        for name in cluster_dominants:
            if not unassigned: break
            idx = name_to_idx[name]
            best_c = max(unassigned, key=lambda c: avg_V[c][idx])
            dom_for_cluster[best_c] = idx; taken.add(idx); unassigned.remove(best_c)
        remaining = sorted(unassigned)
    else:
        remaining = list(unique_clusters)
    if remaining:
        if K >= 2:
            peakiness = {c: avg_V[c].max()-np.partition(avg_V[c],-2)[-2] for c in remaining}
        else:
            peakiness = {c: 0.0 for c in remaining}
        remaining = sorted(remaining, key=lambda c: -peakiness[c])
        for c in remaining:
            order = np.argsort(avg_V[c])[::-1]; chosen = None
            for idx in order:
                if int(idx) not in taken: chosen = int(idx); break
            if chosen is None: chosen = int(order[0])
            dom_for_cluster[c] = chosen; taken.add(chosen)
    sec_for_cluster = {}
    for c in unique_clusters:
        order = np.argsort(avg_V[c])[::-1]; dom = dom_for_cluster[c]; sec = None
        for idx in order:
            if int(idx) != dom: sec = int(idx); break
        sec_for_cluster[c] = sec if sec is not None else dom
    return dom_for_cluster, sec_for_cluster

def _reconstruct_V_from_clusters(V_card, cluster_labels, type_names,
        dom_frac=0.5, sec_frac=0.2, noise_frac=0.3, dirichlet_alpha=1.0,
        cluster_dominants=None, rng=None, dom_override=None, sec_override=None):
    if rng is None: rng = np.random.default_rng(0)
    total = dom_frac+sec_frac+noise_frac
    if not np.isclose(total,1.0): raise ValueError(f"must sum to 1, got {total:.3f}")
    n_spots, K = V_card.shape
    V_new = np.zeros_like(V_card, dtype=np.float32); cluster_info = {}
    dom_for_cluster, sec_for_cluster = _assign_cluster_dominants(
        V_card, cluster_labels, type_names, cluster_dominants=cluster_dominants,
        dom_override=dom_override, sec_override=sec_override)
    for c in sorted(dom_for_cluster.keys()):
        mask = cluster_labels == c
        dom_idx = dom_for_cluster[c]; sec_idx = sec_for_cluster[c]
        cluster_info[int(c)] = {'dominant':dom_idx,'second':sec_idx,'n_spots':int(mask.sum())}
        other_idx = np.array([i for i in range(K) if i != dom_idx and i != sec_idx])
        n_in = mask.sum()
        if other_idx.size > 0:
            noise_props = rng.dirichlet(np.full(other_idx.size, dirichlet_alpha), size=n_in)
        else:
            noise_props = np.zeros((n_in,0))
        cluster_V = np.zeros((n_in,K), dtype=np.float32)
        cluster_V[:, dom_idx] += dom_frac
        if sec_idx != dom_idx: cluster_V[:, sec_idx] += sec_frac
        else: cluster_V[:, dom_idx] += sec_frac
        if other_idx.size > 0: cluster_V[:, other_idx] += noise_frac*noise_props
        V_new[mask] = cluster_V
    V_new = V_new/V_new.sum(axis=1, keepdims=True)
    return V_new, cluster_info

def _propagate_cancer_labels(cluster_labels, cancer_cluster_ids, neighbors,
                             invasion_prob=0.3, rng=None):
    if rng is None: rng = np.random.default_rng(0)
    cancer_set = set(int(c) for c in cancer_cluster_ids)
    new_labels = cluster_labels.copy()
    is_cancer = np.array([lab in cancer_set for lab in cluster_labels])
    for i in range(len(cluster_labels)):
        if is_cancer[i]: continue
        nbr_labels = cluster_labels[neighbors[i]]
        cancer_nbrs = [int(l) for l in nbr_labels if int(l) in cancer_set]
        if not cancer_nbrs: continue
        if rng.random() < invasion_prob:
            uniq, counts = np.unique(cancer_nbrs, return_counts=True)
            new_labels[i] = rng.choice(uniq, p=counts/counts.sum())
    return new_labels

def _calibrate_lib_mean(V, B_matrix, target_sparsity, negbinom_r, lib_sigma, rng,
                        lo=0.5, hi=8.0, n_iter=20, n_sample=500):
    idx = rng.choice(V.shape[0], size=min(n_sample, V.shape[0]), replace=False)
    mean_expr = V[idx] @ B_matrix
    for _ in range(n_iter):
        mid = (lo+hi)/2.0
        lib = np.exp(mid+0.5*lib_sigma**2)
        lam = mean_expr*lib
        p = negbinom_r/(negbinom_r+np.clip(lam,1e-8,None))
        if np.mean(p**negbinom_r) < target_sparsity: hi = mid
        else: lo = mid
    return (lo+hi)/2.0

def _generate_counts(V, B_matrix, rng, negbinom_r=5.0, lib_mean=8.0, lib_sigma=0.5):
    n_spots = V.shape[0]
    mean_expr = V @ B_matrix
    library_size = rng.lognormal(lib_mean, lib_sigma, size=(n_spots,1))
    lam = mean_expr*library_size
    p = negbinom_r/(negbinom_r+np.clip(lam,1e-8,None))
    raw_counts = rng.negative_binomial(negbinom_r, p).astype(np.float32)
    totals = raw_counts.sum(axis=1, keepdims=True)
    totals = np.where(totals==0, 1.0, totals)
    return raw_counts/totals*1e4, raw_counts


def load_region_annotation(path, strip_x=True):
    """clusters.csv -> pd.Series of region name indexed by spot barcode."""
    ann = pd.read_csv(path, index_col=0).iloc[:, 0]
    if strip_x:
        ann.index = ann.index.str.lstrip('X')
    return ann.str.strip().str.lower()


def create_realistic_data(
    ref_adata,
    pd_B,
    V_card,
    region_annotation,            # path to clusters.csv, or a pd.Series
    n_time            = 5,
    # ── Domains ─────────────────────────────────────────────────────────
    region_order      = ('cancer', 'pancreatic', 'duct epthelium', 'stroma'),
    cancer_regions    = ('cancer',),
    region_dominants  = None,      # {region name -> cell type name}
    region_seconds    = None,      # {region name -> cell type name}, optional
    # ── V reconstruction ────────────────────────────────────────────────
    dom_frac          = 0.5,
    sec_frac          = 0.2,
    noise_frac        = 0.3,
    dirichlet_alpha   = 1.0,
    # ── Cancer invasion ─────────────────────────────────────────────────
    invasion_prob     = 0.3,
    n_neighbors       = 6,
    # ── Geometry ────────────────────────────────────────────────────────
    warp_amplitude    = 0.03,
    warp_freq         = 3.0,
    extra_scale       = 1.1,
    extra_rotate_deg  = 25.0,
    # ── Spot dropout ────────────────────────────────────────────────────
    dropout_frac      = 0.0,       # 0 = disabled; e.g. 0.1 drops 10% per slice
    # ── Count model ─────────────────────────────────────────────────────
    negbinom_r        = 5.0,
    lib_sigma         = 0.5,
    seed              = 0,
):
    """
    Returns
    -------
    adata_list, V_true_temporal, cluster_labels_temporal,
    ref_adata_out, pd_B_out, warp_params, region_map

    region_map : dict {cluster_id -> region name}

    Notes
    -----
    Domains are the regions in clusters.csv. Each region's dominant cell type
    is chosen from V_card abundance inside that region (or forced with
    cluster_dominants). cancer_regions names which regions invade — it refers
    to region names in the annotation, not to cell types.
    """
    rng = np.random.default_rng(seed)

    if isinstance(region_annotation, (str, bytes)):
        ann = load_region_annotation(region_annotation)
    else:
        ann = region_annotation.copy()
        ann.index = ann.index.astype(str).str.lstrip('X')
        ann = ann.str.strip().str.lower()

    # ── 1. Align spots: must be in ref_adata order, and in the annotation ─
    mask = ref_adata.obs_names.isin(V_card.index) & ref_adata.obs_names.isin(ann.index)
    if not mask.any():
        raise ValueError("no spots shared between ref_adata, V_card and the annotation.")
    shared_spots = ref_adata.obs_names[mask]
    n_dropped = ref_adata.n_obs - len(shared_spots)
    if n_dropped:
        print(f"  Warning: dropping {n_dropped} spots -> {len(shared_spots)}.")
    ref_adata = ref_adata[shared_spots].copy()

    # ── 2. Cell-type order comes from pd_B; V0 follows it ────────────────
    type_names = list(pd_B.index)
    K          = len(type_names)
    V0 = V_card.loc[shared_spots, type_names].values.astype(np.float32)
    V0 /= V0.sum(axis=1, keepdims=True)
    n_spots = ref_adata.n_obs

    shared_genes = pd_B.columns.intersection(ref_adata.var_names)
    if len(shared_genes) == 0:
        raise ValueError("pd_B columns and ref_adata.var_names share no genes.")
    pd_B       = pd_B[shared_genes].copy()
    B_matrix   = pd_B.values.astype(np.float64)
    gene_names = list(shared_genes)
    print(f"  Spots: {n_spots}  |  Cell types: {K}  |  Genes: {len(gene_names)}")

    # ── 3. Domains from the annotation ───────────────────────────────────
    region_order = list(region_order)
    unknown = set(ann.reindex(shared_spots)) - set(region_order)
    if unknown:
        raise ValueError(f"regions in annotation not listed in region_order: {unknown}")

    region_to_id = {r: i for i, r in enumerate(region_order)}
    base_cluster_labels = ann.reindex(shared_spots).map(region_to_id).values.astype(int)
    region_map = {i: r for r, i in region_to_id.items()}

    for c in sorted(region_map):
        print(f"    cluster {c} = {region_map[c]:14s} n={(base_cluster_labels == c).sum()}")

    # ── 4. Dominants: explicit per region, else abundance ────────────────
    if region_dominants is not None:
        bad = [r for r in region_dominants if r not in region_to_id]
        if bad:
            raise ValueError(f"region_dominants keys not in region_order: {bad}\n"
                             f"  valid regions: {list(region_to_id)}")
        bad = [t for t in region_dominants.values() if t not in type_names]
        if bad:
            raise ValueError(f"region_dominants values not in pd_B.index: {bad}")
        missing = [r for r in region_to_id if r not in region_dominants]
        if missing:
            raise ValueError(f"region_dominants must name every region; "
                             f"missing: {missing}")
        dups = [t for t in set(region_dominants.values())
                if list(region_dominants.values()).count(t) > 1]
        if dups:
            raise ValueError(f"region_dominants assigns the same type twice: {dups}")

        dom_map = {region_to_id[r]: type_names.index(t)
                   for r, t in region_dominants.items()}

        if region_seconds is not None:
            bad = [t for t in region_seconds.values() if t not in type_names]
            if bad:
                raise ValueError(f"region_seconds values not in pd_B.index: {bad}")
            sec_map = {}
            for c, r in region_map.items():
                if r in region_seconds:
                    s = type_names.index(region_seconds[r])
                    if s == dom_map[c]:
                        raise ValueError(
                            f"region '{r}': second equals dominant "
                            f"({region_seconds[r]})")
                    sec_map[c] = s
                else:
                    sec_map[c] = next(
                        int(i) for i in np.argsort(V0[base_cluster_labels == c].mean(0))[::-1]
                        if int(i) != dom_map[c])
        else:
            sec_map = {c: next(
                int(i) for i in np.argsort(V0[base_cluster_labels == c].mean(0))[::-1]
                if int(i) != dom_map[c]) for c in region_map}

        V_structured, cluster_info = _reconstruct_V_from_clusters(
            V0, base_cluster_labels, type_names,
            dom_frac=dom_frac, sec_frac=sec_frac, noise_frac=noise_frac,
            dirichlet_alpha=dirichlet_alpha, rng=rng,
            dom_override=dom_map, sec_override=sec_map,
        )
    else:
        V_structured, cluster_info = _reconstruct_V_from_clusters(
            V0, base_cluster_labels, type_names,
            dom_frac=dom_frac, sec_frac=sec_frac, noise_frac=noise_frac,
            dirichlet_alpha=dirichlet_alpha, rng=rng,
        )
        dom_map = {c: i['dominant'] for c, i in cluster_info.items()}
        sec_map = {c: i['second']   for c, i in cluster_info.items()}

    for c in sorted(cluster_info):
        i = cluster_info[c]
        print(f"    cluster {c} ({region_map[c]}): "
              f"dominant={type_names[i['dominant']]}, "
              f"second={type_names[i['second']]}")

    # program similarity between the chosen dominants — high values mean
    # those domains are not separable by any reference-based method
    _d = [cluster_info[c]['dominant'] for c in sorted(cluster_info)]
    _P = B_matrix[_d]
    _P = _P / (np.linalg.norm(_P, axis=1, keepdims=True) + 1e-12)
    _S = _P @ _P.T
    _worst = max(((_S[a, b], a, b) for a in range(len(_d))
                  for b in range(a + 1, len(_d))), default=(0, 0, 0))
    print(f"  Max dominant-program cosine: {_worst[0]:.3f} "
          f"({region_map[sorted(cluster_info)[_worst[1]]]} vs "
          f"{region_map[sorted(cluster_info)[_worst[2]]]})")

    # ── 5. Cancer clusters (by region name, not cell type) ───────────────
    cancer_regions = list(cancer_regions) if cancer_regions else []
    unknown = [r for r in cancer_regions if r not in region_to_id]
    if unknown:
        raise ValueError(f"cancer_regions not in region_order: {unknown}")
    cancer_cluster_ids = [region_to_id[r] for r in cancer_regions]
    if cancer_cluster_ids:
        print(f"  Cancer regions   : {cancer_regions} -> clusters "
              f"{cancer_cluster_ids}, p={invasion_prob}")
    else:
        print("  Cancer invasion  : disabled")

    # ── 6. Coords + graph ────────────────────────────────────────────────
    ref_coords = np.vstack([ref_adata.obs['x'].values,
                            ref_adata.obs['y'].values]).astype(float)
    ref_coords -= ref_coords.mean(axis=1, keepdims=True)
    neighbors = _build_neighbor_graph(ref_coords, n_neighbors=n_neighbors)

    # ── 7. Calibrate ─────────────────────────────────────────────────────
    real_X = ref_adata[:, gene_names].X
    if hasattr(real_X, "toarray"):
        real_X = real_X.toarray()
    target_sparsity = float((np.asarray(real_X) == 0).mean())
    lib_mean = _calibrate_lib_mean(
        V_structured, B_matrix, target_sparsity,
        negbinom_r=negbinom_r, lib_sigma=lib_sigma, rng=rng,
    )
    print(f"  Sparsity {target_sparsity:.3f} -> lib_mean {lib_mean:.3f}")

    # ── 8. Evolve labels ─────────────────────────────────────────────────
    cluster_labels_temporal = [base_cluster_labels.copy()]
    current = base_cluster_labels.copy()
    for t in range(1, n_time):
        if cancer_cluster_ids:
            current = _propagate_cancer_labels(
                current, cancer_cluster_ids, neighbors,
                invasion_prob=invasion_prob, rng=rng,
            )
        cluster_labels_temporal.append(current.copy())

    # ── 9. Slices ────────────────────────────────────────────────────────
    # h5py cannot write dicts with integer keys, so uns gets string-keyed
    # copies; the returned region_map / cluster_info keep their int keys.
    region_map_str   = {str(k): v for k, v in region_map.items()}
    cluster_info_str = {str(k): v for k, v in cluster_info.items()}

    adata_list, V_true_temporal, warp_params = [], [], []
    labels_temporal_sub, keep_indices = [], []
    psi_x = rng.uniform(0, 2 * np.pi)
    psi_y = rng.uniform(0, 2 * np.pi)

    for t in range(n_time):
        if t == 0:
            V_true_t = V_structured
        else:
            V_true_t, _ = _reconstruct_V_from_clusters(
                V0, cluster_labels_temporal[t], type_names,
                dom_frac=dom_frac, sec_frac=sec_frac, noise_frac=noise_frac,
                dirichlet_alpha=dirichlet_alpha, rng=rng,
                dom_override=dom_map, sec_override=sec_map,
            )
        frac = t / max(n_time - 1, 1)
        wp = dict(ax=warp_amplitude * frac, ay=warp_amplitude * frac,
                  f=warp_freq, psi_x=psi_x, psi_y=psi_y)
        warp_params.append(wp)

        warped = (_sine_warp(ref_coords, **wp) if warp_amplitude > 0
                  else ref_coords.copy())
        scale_t = 1.0 + (extra_scale - 1.0) * frac
        theta_t = np.deg2rad(extra_rotate_deg) * frac
        R_t = np.array([[np.cos(theta_t), np.sin(theta_t)],
                        [-np.sin(theta_t), np.cos(theta_t)]])
        warped = scale_t * (R_t @ warped)

        # ── spot dropout ─────────────────────────────────────────────────
        if dropout_frac > 0:
            n_keep        = max(1, int(n_spots * (1 - dropout_frac)))
            keep          = np.sort(rng.choice(n_spots, size=n_keep, replace=False))
            warped_t      = warped[:, keep]
            V_true_t_sub  = V_true_t[keep]
            labels_t      = cluster_labels_temporal[t][keep]
            base_labels_t = base_cluster_labels[keep]
            spots_t       = [shared_spots[i] for i in keep]
        else:
            keep          = np.arange(n_spots)
            warped_t      = warped
            V_true_t_sub  = V_true_t
            labels_t      = cluster_labels_temporal[t]
            base_labels_t = base_cluster_labels
            spots_t       = list(shared_spots)

        # keep every returned per-slice array on the same (post-dropout) rows
        V_true_temporal.append(V_true_t_sub)
        labels_temporal_sub.append(labels_t)
        keep_indices.append(keep)

        normalised, raw = _generate_counts(
            V_true_t_sub, B_matrix, rng, negbinom_r=negbinom_r,
            lib_mean=lib_mean, lib_sigma=lib_sigma,
        )

        obs_t = pd.DataFrame(
            {
                'x':         warped_t[0],
                'y':         warped_t[1],
                'cluster':   pd.Categorical(labels_t),
                'region':    pd.Categorical(
                    [region_map[int(l)] for l in labels_t]),
                'region_t0': pd.Categorical(
                    [region_map[int(l)] for l in base_labels_t]),
            },
            index=pd.Index([f"{s}-{t}" for s in spots_t], name=None),
        )
        var_t = pd.DataFrame(index=pd.Index(gene_names, name=None))

        a = ad.AnnData(
            X=np.asarray(normalised, dtype=np.float32),
            obs=obs_t,
            var=var_t,
        )
        a.layers['counts'] = np.asarray(raw, dtype=np.float32)
        a.obsm['spatial']  = warped_t.T.copy()
        a.obsm['V_true']   = V_true_t_sub
        a.uns['warp_params']     = wp
        a.uns['B_true']          = B_matrix
        a.uns['time_index']      = t
        # string keys so the object is h5ad-writable (h5py rejects int keys)
        a.uns['cluster_info']    = cluster_info_str
        a.uns['cancer_clusters'] = cancer_cluster_ids
        a.uns['region_map']      = region_map_str
        adata_list.append(a)

        n_can = int(np.isin(labels_t, cancer_cluster_ids).sum())
        print(f"  t={t}  spots={len(spots_t)}/{n_spots}  "
              f"cancer={n_can}/{len(spots_t)} ({n_can / len(spots_t):.0%})")

    # ── 10. Reference ────────────────────────────────────────────────────
    ref_norm, _ = _generate_counts(
        V_structured, B_matrix, rng, negbinom_r=negbinom_r,
        lib_mean=lib_mean, lib_sigma=lib_sigma,
    )
    ref_obs = pd.DataFrame(
        {
            'x':       ref_coords[0],
            'y':       ref_coords[1],
            'cluster': pd.Categorical(base_cluster_labels),
            'region':  pd.Categorical(
                [region_map[int(l)] for l in base_cluster_labels]),
        },
        index=pd.Index(list(shared_spots), name=None),
    )
    ref_out = ad.AnnData(
        X=np.asarray(ref_norm, dtype=np.float32),
        obs=ref_obs,
        var=pd.DataFrame(index=pd.Index(gene_names, name=None)),
    )
    ref_out.obsm['spatial'] = ref_coords.T.copy()
    ref_out.obsm['V_true']  = V_structured

    # cluster_labels_temporal is full-length (n_spots) because the invasion
    # model runs on the full grid; return the dropout-matched version so it
    # lines up with adata_list / V_true_temporal row-for-row.
    cluster_labels_temporal = labels_temporal_sub
    ref_out.uns['keep_indices'] = keep_indices

    return (adata_list, V_true_temporal, cluster_labels_temporal,
            ref_out, pd_B, warp_params, region_map)




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
    P = np.nan_to_num(np.asarray(percents, float), nan=0.0)
    n, k = P.shape
    labels = [f"C{i+1}" for i in range(k)] if labels is None else list(labels)

    if keep_labels is not None:
        label_to_idx = {lab: i for i, lab in enumerate(labels)}
        keep_idx  = [label_to_idx[lb] for lb in keep_labels]
        drop_mask = np.ones(k, dtype=bool); drop_mask[keep_idx] = False
        drop_idx  = np.where(drop_mask)[0]
        P_keep   = P[:, keep_idx]
        P_others = (P[:, drop_idx].sum(axis=1, keepdims=True)
                    if drop_idx.size > 0 else np.zeros((n, 1)))
        P = np.concatenate([P_keep, P_others], axis=1)
        labels = list(keep_labels) + [others_label]; k = P.shape[1]
        if colors is not None:
            new_colors = [colors[i] for i in keep_idx]
            if others_color is None:
                others_color = plt.get_cmap("tab20")(len(new_colors) % 20)
            new_colors.append(others_color); colors = new_colors

    rs = P.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0; P = P / rs

    if top_n is not None and top_n < k:
        keep_idx = np.argsort(P.mean(axis=0))[::-1][:top_n]
        P = P[:, keep_idx]; labels = [labels[i] for i in keep_idx]; k = top_n
        colors = [colors[i] for i in keep_idx] if colors is not None else None

    if colors is None:
        colors = _colors(k)

    if mode == "full":
        for xi, yi, fracs in zip(x, y, P):
            theta = start_angle
            for f, c in zip(fracs, colors):
                if f <= 0:
                    continue
                ax.add_patch(Wedge((xi, yi), r=radius,
                                   theta1=theta, theta2=theta + 360.0 * f,
                                   facecolor=c, edgecolor=edgecolor, linewidth=lw))
                theta += 360.0 * f
    elif mode == "argmax":
        for xi, yi, j in zip(x, y, np.argmax(P, axis=1)):
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
                     max_slices=6, radius=0.35, mode="full", figsize=None,
                     keep_labels=None):
    """Per-slice pie visualisation of V_true. keep_labels collapses the rest
    into 'Others' — with K=20 that is usually what you want."""
    n_show     = min(len(adata_list), max_slices)
    type_names = list(pd_B.index)
    colors     = _colors(len(type_names))

    ncols = min(n_show, 5)
    nrows = int(np.ceil(n_show / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=figsize or (5 * ncols, 5 * nrows))
    axes = np.array(axes).ravel()

    handles = None
    for t in range(n_show):
        handles = scatter_pies(
            axes[t],
            x=adata_list[t].obs['x'].values, y=adata_list[t].obs['y'].values,
            percents=V_true_temporal[t],
            radius=radius, labels=type_names, colors=colors,
            mode=mode, leg_vis=False, keep_labels=keep_labels,
        )
        axes[t].set_title(f"t = {t}", fontsize=11)
        axes[t].axis('off')

    if handles:
        fig.legend(handles=handles, title="Cell types", frameon=False,
                   loc="lower center", ncol=min(len(handles), 5),
                   bbox_to_anchor=(0.5, -0.04), fontsize=8)
    for ax in axes[n_show:]:
        ax.axis('off')
    fig.suptitle(f"Ground-truth compositions (mode='{mode}')", fontsize=13)
    fig.tight_layout()
    return fig


def plot_region_maps(adata_list, max_slices=6, figsize=None, s=20,
                     region_colors=None):
    """Per-slice scatter coloured by obs['region'] — the PDAC-A figure layout."""
    n_show = min(len(adata_list), max_slices)
    region_map = adata_list[0].uns.get('region_map', {})

    names = []
    for a in adata_list[:n_show]:
        for r in pd.unique(a.obs['region']):
            if r not in names:
                names.append(str(r))
    names = sorted(names)

    default = {'cancer_A': '#F1665A', 'cancer_B': '#F7E463',
               'cancer': '#F1665A', 'pancreatic': '#7EB6E8',
               'duct epthelium': '#E8A020', 'duct epithelium': '#E8A020',
               'stroma': '#A8A8A8'}
    if region_colors is None:
        region_colors = {}
    cmap = {r: region_colors.get(r, default.get(r, _colors(len(names))[i]))
            for i, r in enumerate(names)}

    ncols = min(n_show, 5)
    nrows = int(np.ceil(n_show / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=figsize or (4 * ncols, 4 * nrows))
    axes = np.array(axes).ravel()

    cancer_ids = set(int(c) for c in adata_list[0].uns.get('cancer_clusters', []))
    for t in range(n_show):
        a = adata_list[t]
        c = [cmap[str(r)] for r in a.obs['region']]
        axes[t].scatter(a.obs['x'].values, a.obs['y'].values, c=c, s=s,
                        linewidths=0, rasterized=True)
        title = f"t = {t}"
        if cancer_ids:
            n_can = int(np.isin(np.asarray(a.obs['cluster']).astype(int),
                                list(cancer_ids)).sum())
            title += f"   cancer={n_can}/{a.n_obs}"
        axes[t].set_title(title, fontsize=11)
        axes[t].set_aspect('equal'); axes[t].axis('off')

    handles = [Patch(facecolor=cmap[r], label=r) for r in names]
    fig.legend(handles=handles, frameon=False, loc='lower center',
               ncol=min(len(names), 5), bbox_to_anchor=(0.5, -0.04), fontsize=9)
    for ax in axes[n_show:]:
        ax.axis('off')
    fig.suptitle("Ground-truth regions", fontsize=13)
    fig.tight_layout()
    return fig


def plot_warp_distortion(adata_list, ref_adata=None, max_slices=5,
                         n_sample=2000, figsize=None):
    """Grey = t0 coordinates, colour = slice t. ref_adata is ignored."""
    n_show = min(len(adata_list), max_slices)
    colors = _colors(n_show)
    ncols  = min(n_show, 5)
    nrows  = int(np.ceil(n_show / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=figsize or (4 * ncols, 4 * nrows))
    axes = np.array(axes).ravel()

    a0 = adata_list[0]
    rx, ry = a0.obs['x'].values, a0.obs['y'].values
    ridx = np.random.default_rng(0).choice(
        len(rx), size=min(n_sample, len(rx)), replace=False)

    for t in range(n_show):
        a = adata_list[t]
        idx = np.random.default_rng(t).choice(
            a.n_obs, size=min(n_sample, a.n_obs), replace=False)

        axes[t].scatter(rx[ridx], ry[ridx], s=4, c='#cccccc',
                        linewidths=0, label='t=0', rasterized=True)
        axes[t].scatter(a.obs['x'].values[idx], a.obs['y'].values[idx],
                        s=4, c=[colors[t]], linewidths=0, alpha=0.6,
                        label=f't={t}', rasterized=True)
        axes[t].set_title(f"Warp  t = {t}   n={a.n_obs}", fontsize=11)
        axes[t].set_aspect('equal'); axes[t].axis('off')
        axes[t].legend(fontsize=7, loc='upper left', markerscale=3,
                       frameon=False)

    for ax in axes[n_show:]:
        ax.axis('off')
    fig.suptitle("t=0 (grey) vs warped coordinates", fontsize=13)
    fig.tight_layout()
    return fig


def plot_count_stats(adata_list, figsize=(12, 4)):
    n_slices = len(adata_list)
    colors   = _colors(n_slices)
    totals, sparsity = [], []
    for a in adata_list:
        raw = a.layers['counts'] if 'counts' in a.layers else a.X
        if hasattr(raw, 'toarray'):
            raw = raw.toarray()
        raw = np.asarray(raw)
        totals.append(raw.sum(axis=1)); sparsity.append((raw == 0).mean())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    parts = ax1.violinplot(totals, positions=range(n_slices),
                           showmedians=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i]); pc.set_alpha(0.7)
    parts['cmedians'].set_color('black')
    ax1.set_xticks(range(n_slices))
    ax1.set_xticklabels([f"t={t}" for t in range(n_slices)])
    ax1.set_ylabel("Total UMI counts per spot")
    ax1.set_title("Library size distribution")

    bars = ax2.bar(range(n_slices), sparsity, color=colors[:n_slices],
                   edgecolor='white')
    ax2.set_xticks(range(n_slices))
    ax2.set_xticklabels([f"t={t}" for t in range(n_slices)])
    ax2.set_ylabel("Fraction of zeros"); ax2.set_ylim(0, 1)
    ax2.set_title("Sparsity per slice")
    for bar, val in zip(bars, sparsity):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f"{val:.2f}", ha='center', va='bottom', fontsize=9)
    fig.suptitle("Count statistics", fontsize=13)
    fig.tight_layout()
    return fig


def visualize_all(adata_list, V_true_temporal, pd_B, ref_adata=None,
                  view="regions", radius=0.35, mode="full",
                  keep_labels=None, save_prefix=None):
    """
    view : 'regions' | 'composition' | 'both'
        'regions'     — scatter coloured by obs['region'] (the Fig. 2a layout)
        'composition' — pie charts of V_true
    """
    if view not in ('regions', 'composition', 'both'):
        raise ValueError("view must be 'regions', 'composition' or 'both'")

    figs = {}
    if view in ('regions', 'both'):
        figs['regions'] = plot_region_maps(adata_list)
    if view in ('composition', 'both'):
        figs['domains'] = plot_domain_maps(adata_list, V_true_temporal, pd_B,
                                           radius=radius, mode=mode,
                                           keep_labels=keep_labels)
    if ref_adata is not None:
        figs['warp'] = plot_warp_distortion(adata_list, ref_adata)
    figs['counts'] = plot_count_stats(adata_list)

    if save_prefix:
        for name, fig in figs.items():
            path = f"{save_prefix}_{name}.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved  {path}")
    return figs