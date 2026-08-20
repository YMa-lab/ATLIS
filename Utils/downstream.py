
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
import re
import textwrap
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors

import gseapy as gp
from scipy import sparse
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests

# Task 1: Based on the V array, make annotations and show cluster proportions across time points


def get_ordered_clusters_and_times(meta, cluster_col="cluster", cluster_sort_col="cluster_sortkey", time_col="time"):
    """
    clusters: ORIGINAL cluster ids, ordered by numeric sort key if possible, else first appearance
    times: ordered numerically if possible, else first appearance
    """
    clusters_first = list(pd.unique(meta[cluster_col]))

    if cluster_sort_col in meta.columns:
        try:
            tmp = meta[[cluster_col, cluster_sort_col]].drop_duplicates(subset=[cluster_col], keep="first").copy()
            tmp["__sk__"] = pd.to_numeric(tmp[cluster_sort_col], errors="raise")
            tmp = tmp.sort_values("__sk__")
            clusters = tmp[cluster_col].tolist()
        except Exception:
            clusters = clusters_first
    else:
        clusters = clusters_first

    times_first = list(pd.unique(meta[time_col]))
    try:
        tnum = pd.to_numeric(pd.Series(times_first), errors="raise")
        order = np.argsort(tnum.to_numpy())
        times = [times_first[i] for i in order]
    except Exception:
        times = times_first

    return clusters, times

def analyze_cluster_dynamics(
    adata_list,
    V_list,
    pd_B,
    cluster_key="cluster",
    time_key="time",
    n_perm=1000,
    min_cells=10,
    q_thresh=0.05,
    n_markers=20,
    random_state=0,
    add_C_labels=True,     
    C_prefix="C",
    rank_chunk=500,        # gene-block size when ranking, to bound memory
    normalize_V = True
):



    rng = np.random.default_rng(random_state)
    celltypes = pd_B.index.to_list()

    # --------------------------
    # collect
    # --------------------------
    cluster_chunks = []
    sortkey_chunks = []
    time_chunks = []
    V_all = []
    X_all = []

    sortkey_ok = True

    if normalize_V:
        V_list=[V_list[i]/np.sum(V_list[i], axis=1, keepdims=True) for i in range(len(V_list))]

    for ad, V in zip(adata_list, V_list):
        if V.shape[0] != ad.n_obs:
            raise ValueError("V rows must match adata.obs rows")
        if cluster_key not in ad.obs:
            raise KeyError(f"cluster_key='{cluster_key}' not in adata.obs")
        if time_key not in ad.obs:
            raise KeyError(f"time_key='{time_key}' not in adata.obs")

        clusters_raw = ad.obs[cluster_key].to_numpy()
        times_raw = ad.obs[time_key].to_numpy()

        # numeric sort key if possible (does not overwrite original)
        try:
            cl_sort = pd.Series(clusters_raw).astype(int).to_numpy()
        except Exception:
            cl_sort = clusters_raw
            sortkey_ok = False

        X = ad.X.toarray() if hasattr(ad.X, "toarray") else np.asarray(ad.X)

        cluster_chunks.append(clusters_raw)
        sortkey_chunks.append(cl_sort)
        time_chunks.append(times_raw)

        V_all.append(np.asarray(V))
        X_all.append(X)

    meta = pd.DataFrame({
        "cluster": np.concatenate(cluster_chunks),
        "cluster_sortkey": np.concatenate(sortkey_chunks),
        "time": np.concatenate(time_chunks),
    })
    meta.insert(0, "idx", np.arange(len(meta)))

    V_all = np.vstack(V_all)
    X_all = np.vstack(X_all)

    # --------------------------
    # restrict genes
    # --------------------------
    adata_genes = np.array(adata_list[0].var_names)
    B_genes = np.array(pd_B.columns)
    genes = np.intersect1d(adata_genes, B_genes)
    gene_idx = np.where(np.isin(adata_genes, genes))[0]
    X_all = X_all[:, gene_idx]

    # --------------------------
    # ordered clusters/times (clusters returned as ORIGINAL ids)
    # --------------------------
    clusters, times = get_ordered_clusters_and_times(
        meta,
        cluster_col="cluster",
        cluster_sort_col="cluster_sortkey",
        time_col="time",
    )

    # --------------------------
    # optional CXX mapping (adds column only)
    # --------------------------
    if add_C_labels:
        orig_to_C = {
            k: f"{C_prefix}{i:02d}"
            for i, k in enumerate(reversed(clusters))
        }
        meta["cluster_C"] = meta["cluster"].map(orig_to_C)

    # ============================================================
    # Part 1: cluster × time counts
    # ============================================================
    cluster_time_counts = (
        meta.groupby(["cluster", "time"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=clusters, columns=times, fill_value=0)
    )

    cluster_time = []
    for k in clusters:
        counts = cluster_time_counts.loc[k]
        present = counts >= min_cells
        first = present.idxmax() if present.any() else None
        last = present[::-1].idxmax() if present.any() else None
        row = dict(
            cluster=k,
            first_time=first,
            last_time=last,
            total_cells=int(counts.sum()),
        )
        if add_C_labels:
            row["cluster_C"] = orig_to_C[k]
        cluster_time.append(row)

    cluster_time_df = pd.DataFrame(cluster_time)

    # ============================================================
    # Part 2: cluster–cell-type enrichment (stratified permutation)
    # ============================================================
    results = []
    meta_cluster = meta["cluster"].to_numpy()
    meta_time = meta["time"].to_numpy()

    n_cells = len(meta_cluster)
    V_total = V_all.sum(0)

    # index blocks per time stratum, built once
    time_groups = [np.where(meta_time == t)[0] for t in times]
    time_groups = [ids for ids in time_groups if ids.size > 0]

    for k in clusters:
        idx_k = (meta_cluster == k)
        m = int(idx_k.sum())
        if m < min_cells:
            continue

        # observed effect, all cell types at once
        in_sum = V_all[idx_k].sum(0)
        delta_obs = in_sum / m - (V_total - in_sum) / (n_cells - m)

        # stratified permutation: shuffle membership WITHIN each time point,
        # so the per-time group sizes (and therefore m) are preserved exactly
        count = np.zeros(len(celltypes), dtype=float)

        for _ in range(n_perm):
            perm_idx = np.zeros(n_cells, dtype=bool)
            for ids in time_groups:
                perm_idx[ids] = idx_k[rng.permutation(ids)]

            s = V_all[perm_idx].sum(0)
            d = s / m - (V_total - s) / (n_cells - m)
            count += np.abs(d) >= np.abs(delta_obs)

        pvals = count / max(n_perm, 1)

        for c, cname in enumerate(celltypes):
            row = dict(
                cluster=k,
                celltype=cname,
                delta=float(delta_obs[c]),
                p_value=float(pvals[c]),
            )
            if add_C_labels:
                row["cluster_C"] = orig_to_C[k]
            results.append(row)

    cluster_celltype_df = pd.DataFrame(results)

    # BH-FDR
    if len(cluster_celltype_df) > 0:
        p = cluster_celltype_df["p_value"].values
        order = np.argsort(p)
        q = np.empty_like(p)
        m_tests = len(p)
        prev = 1.0
        for i in reversed(range(m_tests)):
            prev = min(prev, p[order[i]] * m_tests / (i + 1))
            q[order[i]] = prev
        cluster_celltype_df["q_value"] = q

    # ============================================================
    # Part 3: assign labels
    # ============================================================
    labels = []
    for k in clusters:
        dfk = cluster_celltype_df[cluster_celltype_df["cluster"] == k]
        sig = dfk[(dfk.get("q_value", 1.0) <= q_thresh) & (dfk["delta"] > 0)]
        if len(sig) == 0:
            row = dict(cluster=k, label="mixed / unknown", confidence=0.0)
        else:
            top = sig.sort_values("delta", ascending=False).iloc[0]
            row = dict(cluster=k, label=top["celltype"], confidence=float(top["delta"]))
        if add_C_labels:
            row["cluster_C"] = orig_to_C[k]
        labels.append(row)

    cluster_label_df = pd.DataFrame(labels)

    # ============================================================
    # Part 4: marker genes
    # ============================================================
    n_obs, n_genes = X_all.shape

    # rank each gene column once; reused by every cluster
    R = np.empty((n_obs, n_genes), dtype=np.float32)
    for a in range(0, n_genes, rank_chunk):
        b = min(a + rank_chunk, n_genes)
        R[:, a:b] = rankdata(X_all[:, a:b], axis=0)

    col_sum = X_all.sum(0)

    cluster_markers = {}
    for k in clusters:
        idx_k = (meta_cluster == k)
        m = int(idx_k.sum())
        if m < min_cells:
            continue

        # Wilcoxon rank-sum, normal approximation (same as scipy.ranksums)
        rank_sum_in = R[idx_k].sum(0, dtype=np.float64)
        expected = m * (n_obs + 1) / 2.0
        sd = np.sqrt(m * (n_obs - m) * (n_obs + 1) / 12.0)
        z = (rank_sum_in - expected) / sd
        pvals = 2.0 * norm.sf(np.abs(z))

        in_sum = X_all[idx_k].sum(0)
        delta = in_sum / m - (col_sum - in_sum) / (n_obs - m)

        pos = np.where(delta > 0)[0]

        dfm = pd.DataFrame({
            "gene": genes[pos],
            "delta_expr": delta[pos].astype(float),
            "p_value": pvals[pos].astype(float),
        })
        cluster_markers[k] = (
            dfm.sort_values("delta_expr", ascending=False).head(n_markers)
        )

    return (
        cluster_time_df,
        cluster_celltype_df,
        cluster_label_df,
        cluster_time_counts,
        cluster_markers,
    )




# Region-level analysis: cell-type proportions, differential expression, GO.


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
AGGREGATORS = {"mean": np.mean, "median": np.median, "sum": np.sum}


def natural_sort_key(x):
    """Sort key that orders 'region2' before 'region10'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(x))]


def to_dense_1d(x):
    return x.toarray().ravel() if sparse.issparse(x) else np.asarray(x).ravel()


def get_matrix(adata, layer=None, use_raw=False):
    """Return (expression matrix, gene names) from raw, a layer, or .X."""
    if use_raw:
        if adata.raw is None:
            raise ValueError("adata.raw is None.")
        return adata.raw.X, np.asarray(adata.raw.var_names).astype(str)
    X = adata.X if layer is None else adata.layers[layer]
    return X, np.asarray(adata.var_names).astype(str)


def mean_expr(X):
    """Column-wise mean, dense or sparse."""
    if sparse.issparse(X):
        return np.asarray(X.mean(axis=0)).ravel()
    return np.asarray(X).mean(axis=0)


def frac_expr(X, threshold=0):
    """Fraction of spots with expression above `threshold`, per gene."""
    if sparse.issparse(X):
        return np.asarray((X > threshold).mean(axis=0)).ravel()
    return (np.asarray(X) > threshold).mean(axis=0)


def resolve_celltype(celltype, celltype_names=None):
    """Map a cell-type index or name to (index, display name)."""
    if isinstance(celltype, str):
        if celltype_names is None:
            raise ValueError("celltype_names is required when celltype is a string.")
        names = list(celltype_names)
        if celltype not in names:
            raise ValueError(f"{celltype!r} not found in celltype_names.")
        return names.index(celltype), celltype

    idx = int(celltype)
    if celltype_names is None:
        return idx, f"celltype {idx}"
    return idx, str(list(celltype_names)[idx])


def scaled_sizes(values, lo, hi, default=None):
    """Linearly rescale `values` into [lo, hi]; flat input maps to the midpoint."""
    v = np.asarray(values, dtype=float)
    if default is not None:
        return np.full(len(v), default, dtype=float)
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(vmax) or vmax <= vmin:
        return np.full_like(v, (lo + hi) / 2, dtype=float)
    return lo + (hi - lo) * (v - vmin) / (vmax - vmin)


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: cell-type proportions by region and time
# ─────────────────────────────────────────────────────────────────────────────
def summarize_V_by_region(
    adata_list,
    V_list,
    celltype,
    region_key,
    celltype_names=None,
    times=None,
    agg="mean",
):
    """Aggregate one cell-type's V column by region within each time point.

    Returns (df, celltype_name) where df has columns
    time / region / value / cluster_size.
    """
    if len(adata_list) != len(V_list):
        raise ValueError("adata_list and V_list must have the same length.")
    if times is None:
        times = list(range(len(adata_list)))
    if len(times) != len(adata_list):
        raise ValueError("times must have the same length as adata_list.")
    if agg not in AGGREGATORS:
        raise ValueError(f"agg must be one of {set(AGGREGATORS)}.")

    idx, celltype_name = resolve_celltype(celltype, celltype_names)
    reducer = AGGREGATORS[agg]

    def reduce_finite(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        return float(reducer(x)) if x.size else np.nan

    frames = []
    for k, (adata, V, t) in enumerate(zip(adata_list, V_list, times)):
        V = np.asarray(V, dtype=float)
        if V.ndim != 2:
            raise ValueError(f"V_list[{k}] must be 2D (n_obs x n_celltypes).")
        if V.shape[0] != adata.n_obs:
            raise ValueError(
                f"V_list[{k}] has {V.shape[0]} rows but adata_list[{k}] has "
                f"{adata.n_obs} observations."
            )
        if not 0 <= idx < V.shape[1]:
            raise IndexError(
                f"celltype index {idx} out of bounds for V_list[{k}] "
                f"with {V.shape[1]} columns."
            )
        if region_key not in adata.obs.columns:
            raise KeyError(f"{region_key!r} not found in adata_list[{k}].obs.")

        tmp = pd.DataFrame({
            "time": t,
            "region": adata.obs[region_key].astype(str).to_numpy(),
            "prop": V[:, idx],
        })
        frames.append(
            tmp.groupby(["time", "region"], observed=True)
            .agg(value=("prop", reduce_finite), cluster_size=("prop", "size"))
            .reset_index()
        )

    return pd.concat(frames, ignore_index=True), celltype_name

def add_cluster_annotation_from_V(
    adata,
    V,
    celltype_order,
    cluster_key="cluster",
    key_added="cluster_annotated_celltype",
    confidence_key="cluster_annotated_confidence",
    normalize=True,
):
    """
    Assign one cell-type label per cluster using average V within each cluster.
    """

    import numpy as np
    import pandas as pd

    V = np.asarray(V)

    if V.shape[0] != adata.n_obs:
        raise ValueError(
            f"V has {V.shape[0]} rows, but adata has {adata.n_obs} observations."
        )

    if V.shape[1] != len(celltype_order):
        raise ValueError(
            f"V has {V.shape[1]} columns, but celltype_order has "
            f"{len(celltype_order)} entries."
        )

    if cluster_key not in adata.obs:
        raise KeyError(f"'{cluster_key}' not found in adata.obs")

    clusters = pd.unique(adata.obs[cluster_key])

    cluster_to_label = {}
    cluster_to_conf = {}

    for k in clusters:
        idx = adata.obs[cluster_key].values == k
        mean_v = V[idx].mean(axis=0)
        if normalize:
            mean_v = mean_v / (mean_v.sum() + 1e-12)
        best = int(np.argmax(mean_v))
        cluster_to_label[k] = celltype_order[best]
        cluster_to_conf[k] = float(mean_v[best])

    adata.obs[key_added] = (
        adata.obs[cluster_key]
        .map(cluster_to_label)
        .astype("category")
    )

    adata.obs[confidence_key] = (
        adata.obs[cluster_key]
        .map(cluster_to_conf)
        .astype(float)
    )

    return adata



# ─────────────────────────────────────────────────────────────────────────────
# Part 2: differential expression, region vs rest
# ─────────────────────────────────────────────────────────────────────────────

 

def region_de_wilcoxon(
    adata,
    region_key,
    layer=None,
    fc_layer="normalized_counts",
    use_raw=False,
    counts_layer="counts",
    target_sum=1e4,
    genes=None,
    min_region_size=10,
    min_frac=0.05,
    logfc_eps=1e-6,
    verbose=True,
):
    """Wilcoxon rank-sum DE for each region against all others.
 
    Tests run on log-normalized expression; fold changes use normalized
    non-log expression, since log1p'd means don't give interpretable ratios.
 
    If `counts_layer` names a layer that exists, both matrices are rebuilt
    from those raw counts (normalize to `target_sum`, then log1p for the test
    matrix) and `layer` / `fc_layer` are ignored. Otherwise `layer` supplies
    the test matrix and `fc_layer` the fold-change matrix, both assumed to be
    preprocessed already. Set `counts_layer=None` to force that second path.
 
    `genes` restricts testing to a subset (names or a boolean/index mask over
    var_names). Subsetting happens before the multiple-testing correction, so
    p_adj is computed over the kept genes only — a smaller list means less
    correction and more hits, which is a real change to the result, not just
    a filter on the output.
    """
 
    def natural_sort_key(x):
        return [
            int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", str(x))
        ]
 
    def to_dense_1d(x):
        return x.toarray().ravel() if sparse.issparse(x) else np.asarray(x).ravel()
 
    def mean_expr(X):
        if sparse.issparse(X):
            return np.asarray(X.mean(axis=0)).ravel()
        return np.asarray(X).mean(axis=0)
 
    def frac_expr(X, threshold=0):
        if sparse.issparse(X):
            return np.asarray((X > threshold).mean(axis=0)).ravel()
        return (np.asarray(X) > threshold).mean(axis=0)
 
    def get_matrix(from_layer, raw=False):
        if raw:
            if adata.raw is None:
                raise ValueError("adata.raw is None.")
            return adata.raw.X, np.asarray(adata.raw.var_names).astype(str)
        X = adata.X if from_layer is None else adata.layers[from_layer]
        return X, np.asarray(adata.var_names).astype(str)
 
    def normalize_umi(X):
        """Scale each spot to `target_sum` total counts."""
        if sparse.issparse(X):
            X = X.tocsr(copy=True).astype(float)
            totals = np.asarray(X.sum(axis=1)).ravel()
            return sparse.diags(target_sum / np.maximum(totals, 1e-12)) @ X
        X = np.asarray(X, float)
        return X * target_sum / np.maximum(X.sum(axis=1, keepdims=True), 1e-12)
 
    if region_key not in adata.obs.columns:
        raise KeyError(f"{region_key!r} not found in adata.obs.")
 
    # ── Resolve the test / fold-change matrix pair ───────────────────────
    if counts_layer is not None and counts_layer in adata.layers:
        if verbose:
            print(f"Renormalizing from layers[{counts_layer!r}] "
                  f"(target_sum={target_sum:g}, log1p for tests).")
        counts, genes_all = get_matrix(counts_layer)
        X_fc = normalize_umi(counts)
        X_test = X_fc.log1p() if sparse.issparse(X_fc) else np.log1p(X_fc)
 
    elif fc_layer is None:
        X_test, genes_all = get_matrix(layer, raw=use_raw)
        X_fc = X_test
 
    else:
        X_test, genes_all = get_matrix(layer, raw=use_raw)
        if fc_layer not in adata.layers:
            raise KeyError(
                f"{fc_layer!r} not found in adata.layers. Pass counts_layer= "
                f"to derive both matrices from raw counts, or fc_layer=None to "
                f"reuse the test matrix for fold changes."
            )
        X_fc, genes_fc = get_matrix(fc_layer)
        if not np.array_equal(genes_all, genes_fc):
            raise ValueError(
                "Testing and fold-change matrices have different genes."
            )
 
    # ── Optional gene subset ─────────────────────────────────────────────
    if genes is not None:
        wanted = np.asarray(genes)
        if wanted.dtype == bool:
            if len(wanted) != len(genes_all):
                raise ValueError(
                    f"boolean gene mask has length {len(wanted)}, "
                    f"expected {len(genes_all)}."
                )
            keep_genes = np.where(wanted)[0]
        else:
            pos = {g: i for i, g in enumerate(genes_all)}
            missing = [g for g in wanted.astype(str) if g not in pos]
            if missing:
                print(f"  Warning: {len(missing)} gene(s) not in the matrix, "
                      f"e.g. {missing[:5]}")
            keep_genes = np.array(
                [pos[g] for g in wanted.astype(str) if g in pos], dtype=int
            )
 
        if keep_genes.size == 0:
            raise ValueError("No requested genes are present in the matrix.")
 
        genes_all = genes_all[keep_genes]
        X_test = X_test[:, keep_genes]
        X_fc = X_fc[:, keep_genes]
        if verbose:
            print(f"  Testing {len(genes_all)} of the supplied genes.")
 
    # ── Region vs rest ───────────────────────────────────────────────────
    regions = adata.obs[region_key].astype(str).to_numpy()
    frames = []
 
    for region in sorted(np.unique(regions), key=natural_sort_key):
        mask_in = regions == region
        n_in, n_out = int(mask_in.sum()), int((~mask_in).sum())
        if n_in < min_region_size or n_out < min_region_size:
            continue
        if verbose:
            print(f"Computing region {region}...")
 
        X_in, X_out = X_test[mask_in], X_test[~mask_in]
        frac_in, frac_out = frac_expr(X_in), frac_expr(X_out)
        keep = (frac_in >= min_frac) | (frac_out >= min_frac)
 
        pvals = np.ones(len(genes_all))
        stats = np.zeros(len(genes_all))
 
        for j in np.where(keep)[0]:
            xin, xout = to_dense_1d(X_in[:, j]), to_dense_1d(X_out[:, j])
            # Skip genes constant at the same value in both groups.
            if xin.min() == xin.max() == xout.min() == xout.max():
                continue
            stats[j], pvals[j] = ranksums(xin, xout)
 
        mean_in = mean_expr(X_fc[mask_in])
        mean_out = mean_expr(X_fc[~mask_in])
 
        frames.append(pd.DataFrame({
            "region": region,
            "gene": genes_all,
            "stat": stats,
            "pval": pvals,
            "p_adj": multipletests(pvals, method="fdr_bh")[1],
            "mean_in": mean_in,
            "mean_out": mean_out,
            "frac_in": frac_in,
            "frac_out": frac_out,
            "log2fc": np.log2((mean_in + logfc_eps) / (mean_out + logfc_eps)),
            "n_in": n_in,
            "n_out": n_out,
        }).sort_values(
            ["p_adj", "pval", "log2fc"],
            ascending=[True, True, False],
            kind="mergesort",
        ))
 
    if not frames:
        raise ValueError("No regions passed the size filter.")
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Part 3: per-gene expression across regions
# ─────────────────────────────────────────────────────────────────────────────
def compute_gene_region_summary(
    adata, genes, region_key, layer=None, use_raw=False, region_order=None
):
    """Mean expression and fraction expressing for selected genes, by region."""
    X, gene_names = get_matrix(adata, layer=layer, use_raw=use_raw)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    missing = [g for g in genes if g not in gene_to_idx]
    if missing:
        print("Warning: missing genes:", missing)
    genes = [g for g in genes if g in gene_to_idx]

    regions = adata.obs[region_key].astype(str).to_numpy()
    if region_order is None:
        region_order = sorted(np.unique(regions), key=natural_sort_key)

    rows = []
    for region in region_order:
        mask = regions == region
        Xr = X[mask]
        for gene in genes:
            x = to_dense_1d(Xr[:, gene_to_idx[gene]])
            rows.append({
                "region": region,
                "gene": gene,
                "mean_expr": float(np.mean(x)),
                "frac_expr": float(np.mean(x > 0)),
                "n_spots": int(mask.sum()),
            })
    return pd.DataFrame(rows)


def plot_gene_lollipop_by_region(
    adata,
    genes,
    region_key,
    layer=None,
    use_raw=False,
    region_order=None,
    value="mean_expr",
    color_by="gene",
    size_by=None,
    ncols=3,
    figsize_per_panel=(4.8, 3.3),
    cmap_name="tab20",
    stem_color="0.75",
    stem_lw=1.4,
    dot_size_min=45,
    dot_size_max=190,
    edgecolor="black",
    linewidth=0.35,
    rotate_xticks=True,
    title_prefix="",
    share_colorbar=False,
    panel_facecolor="#fafafa",
    gene_col="gene",
    gene_direction_map=None,
    gene_region_map=None,
    gene_color_map=None,
    gene_label_map=None,
):
    """One lollipop panel per gene: region on x, expression on y.

    `color_by="gene"` gives each gene a categorical color; a numeric column
    ("mean_expr" / "frac_expr") maps onto `cmap_name` with a shared scale.
    The optional *_map arguments annotate panel titles and override colors.

    Returns (fig, axes, df).
    """
    df = compute_gene_region_summary(
        adata, genes, region_key,
        layer=layer, use_raw=use_raw, region_order=region_order,
    )
    genes_present = df[gene_col].drop_duplicates().tolist()

    if region_order is None:
        region_order = sorted(df["region"].unique(), key=natural_sort_key)
    region_to_x = {r: i for i, r in enumerate(region_order)}

    color_by = color_by or value
    for name, col in (("value", value), ("size_by", size_by)):
        if col is not None and col not in df.columns:
            raise ValueError(f"{name}={col!r} not found in dataframe columns.")
    if color_by != "gene" and color_by not in df.columns:
        raise ValueError(f"color_by={color_by!r} not found in dataframe columns.")

    cmap = plt.get_cmap(cmap_name)
    norm = None
    if color_by == "gene":
        if gene_color_map is None:
            gene_color_map = {
                g: cmap(i % cmap.N) for i, g in enumerate(genes_present)
            }
    else:
        vals = df[color_by].to_numpy(dtype=float)
        cmin, cmax = np.nanmin(vals), np.nanmax(vals)
        norm = Normalize(vmin=cmin, vmax=max(cmax, cmin + 1e-12))

    # Size scale is global across panels so dots stay comparable.
    size_range = None
    if size_by is not None:
        all_sizes = df[size_by].to_numpy(dtype=float)
        size_range = (np.nanmin(all_sizes), np.nanmax(all_sizes))

    def sizes_for(sub):
        if size_by is None:
            return np.full(len(sub), dot_size_max * 0.75)
        lo, hi = size_range
        v = sub[size_by].to_numpy(dtype=float)
        if not np.isfinite(hi) or hi <= lo:
            return np.full(len(v), (dot_size_min + dot_size_max) / 2)
        return dot_size_min + (dot_size_max - dot_size_min) * (v - lo) / (hi - lo)

    n = len(genes_present)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )

    for i, gene in enumerate(genes_present):
        ax = axes[i // ncols, i % ncols]
        sub = df[df[gene_col] == gene].copy()
        sub["x"] = sub["region"].map(region_to_x)
        sub = sub.sort_values("x")

        x = sub["x"].to_numpy()
        y = sub[value].to_numpy(dtype=float)

        ax.set_facecolor(panel_facecolor)
        ax.vlines(x, 0, y, color=stem_color, linewidth=stem_lw, alpha=0.9, zorder=1)

        shared = dict(
            s=sizes_for(sub), edgecolor=edgecolor, linewidth=linewidth, zorder=3
        )
        if color_by == "gene":
            ax.scatter(x, y, color=gene_color_map.get(gene, "tab:gray"), **shared)
        else:
            ax.scatter(
                x, y, c=sub[color_by].to_numpy(dtype=float),
                cmap=cmap, norm=norm, **shared,
            )

        ax.set_xticks(range(len(region_order)))
        ax.set_xticklabels(region_order)
        if rotate_xticks:
            ax.tick_params(axis="x", rotation=45)

        ax.set_xlabel("Cluster")
        ax.set_ylabel(
            {"mean_expr": "Mean expression", "frac_expr": "Fraction expressing"}
            .get(value, value)
        )

        label = (gene_label_map or {}).get(gene) or gene
        bits = [str(label)]
        if gene_region_map and gene in gene_region_map:
            bits.append(f"cluster {gene_region_map[gene]}")
        if gene_direction_map and gene in gene_direction_map:
            bits.append(str(gene_direction_map[gene]))
        ax.set_title(f"{title_prefix}{' | '.join(bits)}", fontsize=12)

        ax.grid(True, axis="y", linewidth=0.35, alpha=0.35)
        ax.grid(True, axis="x", linewidth=0.2, alpha=0.12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    if share_colorbar and color_by != "gene":
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.75, pad=0.015)
        cbar.set_label(
            {"mean_expr": "Mean expression", "frac_expr": "Fraction expressing"}
            .get(color_by, color_by)
        )

    plt.tight_layout()
    return fig, axes, df


# ─────────────────────────────────────────────────────────────────────────────
# Part 4: gene-name parsing (human symbol | AMEX id)
# ─────────────────────────────────────────────────────────────────────────────
NULL_TOKENS = {"", "nan", "none", "na"}

def clean_gene_for_enrichr(x):
    # This is specifically for the ARISTA data. Their gene names already have human symbols in front of AMEX names
    if pd.isna(x):
        return None

    s = str(x).strip()

    # Prefer the human ortholog when explicitly provided
    hs_match = re.search(
        r"([A-Za-z0-9_.-]+)\[hs\]",
        s,
        flags=re.IGNORECASE,
    )
    if hs_match:
        s = hs_match.group(1)
    else:
        s = s.split("||")[0]

    # Remove species tags and axolotl paralog suffixes
    s = re.sub(r"\[[^\]]+\]", "", s)
    s = re.sub(r"\.(L|S)$", "", s, flags=re.IGNORECASE)
    s = s.strip().upper()

    # Exclude missing or uninformative identifiers
    if s in {"", "NAN", "NONE", "NA"}:
        return None
    if s.startswith("LOC"):
        return None
    if re.match(r"^(AMEX|AXO|SCAFFOLD|CONTIG)", s):
        return None

    return s
def parse_human_amex_gene_name(gene_name):
    """Split 'VIM|AMEX60DD017849' into ('VIM', 'AMEX60DD017849').

    A bare AMEX id yields (None, id); a bare symbol yields (symbol, None).
    """
    s = str(gene_name).strip()
    if "|" in s:
        left, right = s.split("|", 1)
        human, amex = left.strip(), right.strip()
    elif s.upper().startswith("AMEX"):
        human, amex = None, s
    else:
        human, amex = s, None

    blank = lambda v: None if v is None or v.lower() in NULL_TOKENS else v
    return blank(human), blank(amex)


def clean_human_symbol(x):
    """Normalize a human gene symbol for Enrichr, or None if unusable."""
    if x is None or pd.isna(x):
        return None
    s = re.sub(r"^human[:_]", "", str(x).strip(), flags=re.IGNORECASE)
    s = re.sub(r"_HUMAN$", "", s, flags=re.IGNORECASE)
    s = s.split()[0] if s.split() else ""
    return None if s.lower() in NULL_TOKENS else s.upper()


def make_gene_name_table(gene_names):
    """Table mapping original gene name -> human symbol -> AMEX id."""
    rows = []
    for g in gene_names:
        human, amex = parse_human_amex_gene_name(g)
        rows.append({
            "gene": str(g),
            "human_symbol_raw": human,
            "human_symbol": clean_human_symbol(human),
            "amex_id": amex,
        })
    return pd.DataFrame(rows)


def add_parsed_gene_names_to_de(de_df, gene_col="gene"):
    """Attach human_symbol and amex_id columns to a DE table."""
    out = de_df.copy().merge(
        make_gene_name_table(de_df[gene_col].unique()),
        left_on=gene_col, right_on="gene", how="left", suffixes=("", "_parsed"),
    )
    return out.drop(columns=["gene_parsed"], errors="ignore")


def get_region_marker_human_symbols(
    de_df,
    regions_of_interest,
    p_adj_cutoff=0.05,
    log2fc_cutoff=0.25,
    top_n=150,
    region_col="region",
    gene_col="gene",
    human_col="human_symbol",
):
    """Top markers per region as human symbols.

    Filters on FDR and |log2fc|, keeps the `top_n` most significant per region,
    and tags each row 'up' or 'down'.

    Returns (dict region -> symbols, filtered marker table).
    """
    df = de_df.copy()
    if human_col not in df.columns:
        df = add_parsed_gene_names_to_de(df, gene_col=gene_col)

    df[region_col] = df[region_col].astype(str)
    wanted = {str(r) for r in regions_of_interest}

    marker_df = df[
        df[region_col].isin(wanted)
        & (df["p_adj"] <= p_adj_cutoff)
        & (df["log2fc"].abs() >= log2fc_cutoff)
        & df[human_col].notna()
    ].copy()

    marker_df["abs_log2fc"] = marker_df["log2fc"].abs()
    marker_df["direction"] = np.where(marker_df["log2fc"] >= 0, "up", "down")
    marker_df = (
        marker_df
        .sort_values([region_col, "p_adj", "abs_log2fc"], ascending=[True, True, False])
        .groupby(region_col, group_keys=False)
        .head(top_n)
        .copy()
    )

    marker_dict = {
        str(region): (
            sub[human_col].dropna().astype(str)
            .map(clean_human_symbol).dropna().drop_duplicates().tolist()
        )
        for region, sub in marker_df.groupby(region_col)
    }
    return marker_dict, marker_df


def print_region_marker_summary(marker_dict, max_genes=20):
    """Print how many usable symbols each region contributed."""
    for region, genes in marker_dict.items():
        genes = list(pd.Series(genes).dropna().drop_duplicates())
        print(f"\nRegion {region}: {len(genes)} usable human-symbol genes")
        print(", ".join(genes[:max_genes]))
        if len(genes) > max_genes:
            print(f"... + {len(genes) - max_genes} more")


# ─────────────────────────────────────────────────────────────────────────────
# Part 5: GO enrichment
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_GENE_SETS = [
    "GO_Biological_Process_2023",
    "GO_Molecular_Function_2023",
    "GO_Cellular_Component_2023",
]


def run_go_enrichr_by_region(
    marker_dict,
    organism="Human",
    gene_sets=None,
    cutoff=0.05,
    max_retries=5,
    initial_wait=5,
    delay_between_regions=2,
    min_genes=5,
):
    """Run Enrichr GO enrichment per region, backing off on HTTP 429."""
    gene_sets = gene_sets or DEFAULT_GENE_SETS
    frames = []

    for region, genes in marker_dict.items():
        genes = (
            pd.Series(genes, dtype="object").dropna().astype(str)
            .map(clean_human_symbol).dropna().drop_duplicates().tolist()
        )
        if len(genes) < min_genes:
            print(f"Skipping region {region}: only {len(genes)} usable genes.")
            continue

        print(f"Running GO for region {region}: {len(genes)} genes")
        enr = None
        for attempt in range(max_retries):
            try:
                enr = gp.enrichr(
                    gene_list=genes, gene_sets=gene_sets,
                    organism=organism, outdir=None, cutoff=cutoff,
                )
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait = initial_wait * 2 ** attempt
                    print(
                        f"Region {region} rate-limited; retrying in {wait}s "
                        f"({attempt + 1}/{max_retries})..."
                    )
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Enrichr failed for region {region} after "
                        f"{attempt + 1} attempt(s)."
                    ) from e

        if enr is None or enr.results is None or enr.results.empty:
            print(f"No enrichment found for region {region}.")
        else:
            temp = enr.results.copy()
            temp["region"] = str(region)
            temp["n_marker_genes"] = len(genes)
            frames.append(temp)
            print(f"Region {region}: {len(temp)} terms returned")

        time.sleep(delay_between_regions)

    if not frames:
        print("No GO enrichment results returned.")
        return pd.DataFrame()

    go_df = pd.concat(frames, ignore_index=True)
    sort_cols = [c for c in ("region", "Adjusted P-value", "P-value", "Term")
                 if c in go_df.columns]
    return go_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def shorten_go_term(term, max_length=85):
    """Strip the GO id, abbreviate stock phrases, and truncate to `max_length`."""
    replacements = {
        "Positive Regulation Of": "Positive Regulation:",
        "Negative Regulation Of": "Negative Regulation:",
        "regulation Of": "regulation:",
        "Regulation Of": "Regulation:",
        "Modulation Of": "Modulation:",
        "cellular response to": "response to",
        "biological process involved in": "process in",
        "cellular component organization": "component organization",
    }
    s = re.sub(r"\s*\(GO:\d+\)\s*$", "", str(term))
    for old, new in replacements.items():
        s = s.replace(old, new)
    s = re.sub(r"\s+", " ", s).strip()
    return textwrap.shorten(s, width=max_length, placeholder="…")


def prepare_go_plot_df(
    go_df,
    regions_of_interest=None,
    library=None,
    top_n=10,
    p_col="Adjusted P-value",
    term_col="Term",
    direction_col="direction",
    directions=None,
):
    """Filter, clean, and take the top `top_n` GO terms per region (or per
    region-direction pair when a direction column is present)."""
    if go_df is None or go_df.empty:
        raise ValueError("go_df is empty. Run GO enrichment first.")
    for col in (p_col, term_col):
        if col not in go_df.columns:
            raise ValueError(f"{col!r} not found in go_df.")

    df = go_df.copy()

    if regions_of_interest is not None:
        df["region"] = df["region"].astype(str)
        df = df[df["region"].isin({str(r) for r in regions_of_interest})]

    if library is not None:
        if "Gene_set" not in df.columns:
            raise ValueError("Column 'Gene_set' not found in go_df.")
        df = df[df["Gene_set"] == library]

    has_direction = direction_col in df.columns
    if has_direction:
        df[direction_col] = df[direction_col].astype(str)
        if directions is not None:
            df = df[df[direction_col].isin({str(d) for d in directions})]

    df[p_col] = pd.to_numeric(df[p_col], errors="coerce")
    df = df[np.isfinite(df[p_col]) & (df[p_col] > 0)].copy()

    df["term_clean"] = df[term_col].map(shorten_go_term)
    df["minus_log10_adj_p"] = -np.log10(df[p_col])

    group = ["region", direction_col] if has_direction else ["region"]
    return (
        df.sort_values(group + [p_col])
        .groupby(group, group_keys=False)
        .head(top_n)
        .copy()
    )


def plot_go_barplot_by_region(
    go_df,
    regions_of_interest=None,
    library="GO_Biological_Process_2023",
    top_n=10,
    figsize_per_panel=(6.2, 3.4),
    direction_col="direction",
    directions=("up", "down"),
    color_up="Reds",
    color_down="Blues",
    color_default="Greys",
):
    """Horizontal GO barplots, one panel per nonempty region-direction pair.

    Bar length is -log10(adjusted p); color intensity shares a global scale.

    Returns (fig, axes, plot_df).
    """
    plot_df = prepare_go_plot_df(
        go_df, regions_of_interest=regions_of_interest, library=library,
        top_n=top_n, direction_col=direction_col, directions=directions,
    )
    if plot_df.empty:
        raise ValueError("No GO terms left after filtering.")

    plot_df["region"] = plot_df["region"].astype(str)
    has_direction = direction_col in plot_df.columns

    if regions_of_interest is not None:
        available = set(plot_df["region"])
        region_order = [str(r) for r in regions_of_interest if str(r) in available]
    else:
        region_order = plot_df["region"].drop_duplicates().tolist()

    if has_direction:
        available = set(plot_df[direction_col])
        panel_keys = [
            (region, direction)
            for region in region_order
            for direction in directions
            if str(direction) in available
            and not plot_df[
                (plot_df["region"] == region)
                & (plot_df[direction_col] == str(direction))
            ].empty
        ]
    else:
        panel_keys = [(region, None) for region in region_order]

    if not panel_keys:
        raise ValueError("No nonempty region-direction panels to plot.")

    fig, axes = plt.subplots(
        nrows=len(panel_keys), ncols=1,
        figsize=(figsize_per_panel[0], figsize_per_panel[1] * len(panel_keys)),
        squeeze=False,
    )

    vmax = plot_df["minus_log10_adj_p"].max()
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    cmaps = {"up": color_up, "down": color_down}
    labels = {"up": "upregulated", "down": "downregulated"}

    for i, (region, direction) in enumerate(panel_keys):
        ax = axes[i, 0]
        sub = plot_df[plot_df["region"] == region]
        if has_direction:
            sub = sub[sub[direction_col] == str(direction)]
        sub = sub.sort_values("minus_log10_adj_p")

        values = sub["minus_log10_adj_p"].to_numpy()
        cmap = plt.get_cmap(cmaps.get(direction, color_default))

        ax.barh(
            sub["term_clean"], values,
            color=cmap(values / vmax), edgecolor="black", linewidth=0.3,
        )
        ax.set_xlabel(r"$-\log_{10}$(adjusted p-value)")

        label = labels.get(direction)
        ax.set_title(f"region {region}" + (f", {label}" if label else ""))
        ax.grid(True, axis="x", alpha=0.25, linewidth=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig, axes, plot_df



def compute_transition_tables_from_couplings(
    adata_list,
    couplings,
    celltype_key="annotation",
    keep_labels=None,         # e.g. ["RGC","NeuB","GlioB"]
    others_label="Others",    # collapsed remainder label
    normalize="row",          # "row", "col", "total", or None
):
    """
    Aggregate cell-level couplings into cell-type transition tables.

    Parameters
    ----------
    adata_list : list of AnnData
        Length T. adata_list[t].obs[celltype_key] contains cell-type labels at time t.
    couplings : list of array-like
        Length T-1. couplings[t] is the coupling matrix between time t and t+1,
        shape (n_t, n_{t+1}).
    celltype_key : str
        obs column name with categorical labels.
    keep_labels : list of str or None
        If provided, keep only these labels and collapse all others into `others_label`.
    others_label : str or None
        Label name used for collapsed types. If None, unknown labels are dropped (rows become all-zero).
    normalize : {"row","col","total",None}
        Normalization for each transition table:
        - "row": each row sums to 1 (P(target type | source type))
        - "col": each column sums to 1
        - "total": whole table sums to 1
        - None: raw transported mass

    Returns
    -------
    tables : list of pd.DataFrame
        Length T-1. Each is (K x K) transition table from time t to t+1.
    labels_out : list of str
        The label ordering used for rows/cols.
    """
    # ---- small helper: collapse labels ----
    def _collapse(arr):
        arr = pd.Series(arr, dtype="string")
        if keep_labels is None:
            return arr.to_numpy()
        keep = set(keep_labels)
        if others_label is None:
            # drop unknowns by setting them to NA; later they contribute nothing
            arr = arr.where(arr.isin(keep), other=pd.NA)
        else:
            arr = arr.where(arr.isin(keep), other=others_label)
        return arr.to_numpy()

    T = len(adata_list)
    if len(couplings) != T - 1:
        raise ValueError(f"Expected len(couplings)=T-1={T-1}, got {len(couplings)}")

    # ---- collect labels per time point (collapsed if requested) ----
    labs_per_t = []
    for t, ad in enumerate(adata_list):
        if celltype_key not in ad.obs:
            raise KeyError(f"adata_list[{t}] missing obs['{celltype_key}']")
        labs = ad.obs[celltype_key].astype(str).to_numpy()
        labs_per_t.append(_collapse(labs))

    # ---- build global label ordering ----
    if keep_labels is not None:
        labels_out = list(keep_labels)
        if others_label is not None:
            # add Others only if it appears
            if any(np.any(l == others_label) for l in labs_per_t):
                if others_label not in labels_out:
                    labels_out.append(others_label)
        # if others_label is None, unknowns are NA and won't appear
    else:
        labels_out = sorted(set(np.concatenate([l[~pd.isna(l)] for l in labs_per_t])))

    K = len(labels_out)
    lab_to_idx = {lab: i for i, lab in enumerate(labels_out)}

    # ---- sparse support (optional) ----
    try:
        from scipy import sparse
        _has_sparse = True
    except Exception:
        _has_sparse = False

    tables = []
    for t in range(T - 1):
        G = couplings[t]
        n0 = adata_list[t].n_obs
        n1 = adata_list[t + 1].n_obs

        if getattr(G, "shape", None) != (n0, n1):
            raise ValueError(
                f"couplings[{t}] has shape {getattr(G,'shape',None)}, expected {(n0, n1)}"
            )

        src = labs_per_t[t]
        tgt = labs_per_t[t + 1]

        M = np.zeros((K, K), dtype=float)

        is_sparse = _has_sparse and sparse.issparse(G)
        if is_sparse:
            G = G.tocoo()
            for i, j, v in zip(G.row, G.col, G.data):
                si = src[i]
                tj = tgt[j]
                if pd.isna(si) or pd.isna(tj):
                    continue
                a = lab_to_idx[si]
                b = lab_to_idx[tj]
                M[a, b] += float(v)
        else:
            # dense: accumulate by source label blocks
            tgt_idx = np.array([-1 if pd.isna(l) else lab_to_idx[l] for l in tgt], dtype=int)
            for a_lab, a in lab_to_idx.items():
                src_mask = (src == a_lab)
                if not np.any(src_mask):
                    continue
                block = np.asarray(G[src_mask, :])
                s = block.sum(axis=0).ravel()  # mass to each target cell
                good = tgt_idx >= 0
                np.add.at(M[a, :], tgt_idx[good], s[good])

        # ---- normalize if requested ----
        if normalize is None:
            Mn = M
        elif normalize == "row":
            rs = M.sum(axis=1, keepdims=True)
            rs[rs == 0] = 1.0
            Mn = M / rs
        elif normalize == "col":
            cs = M.sum(axis=0, keepdims=True)
            cs[cs == 0] = 1.0
            Mn = M / cs
        elif normalize == "total":
            tot = M.sum()
            Mn = M / (tot if tot != 0 else 1.0)
        else:
            raise ValueError("normalize must be one of {'row','col','total',None}")

        tables.append(pd.DataFrame(Mn, index=labels_out, columns=labels_out))

    return tables, labels_out


import matplotlib.colors as mcolors

def shades(base, n=3, lo=0.35, hi=1.0):
    """n tints of `base`, palest first."""
    rgb = np.array(mcolors.to_rgb(base))
    return [tuple(1 - f * (1 - rgb)) for f in np.linspace(lo, hi, n)]

def cluster_gene_means(adata_all, timepoints, cluster, genes,
                       label_key='pred_label', log1p=False, min_spots=20):
    """Mean expression per gene per age, restricted to one cluster.
 
    Returns (means, counts):
        means   DataFrame, index = ages, columns = genes.  Rows where the
                cluster has fewer than `min_spots` spots are set to NaN so
                they break the line rather than plotting a noisy point.
        counts  Series of spots in the cluster at each age.
    """
    cluster = str(cluster)
    rows, counts = {}, {}
 
    for age, a in zip(timepoints, adata_all):
        mask = a.obs[label_key].astype(str).to_numpy() == cluster
        n = int(mask.sum())
        counts[age] = n
 
        present = [g for g in genes if g in a.var_names]
        if n == 0:
            rows[age] = pd.Series(np.nan, index=genes)
            continue
 
        sub = a[mask, present]
        X = sub.X
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        if log1p:
            X = np.log1p(X)
 
        vals = pd.Series(np.nan, index=genes, dtype=float)
        vals[present] = X.mean(axis=0)
        rows[age] = vals if n >= min_spots else pd.Series(np.nan, index=genes)
 
    means = pd.DataFrame(rows).T
    means.index.name = "age"
    return means, pd.Series(counts, name="n_spots")
 

import matplotlib.colors as mcolors

def plot_gene_dynamics_by_gene(adata_all, timepoints, genes,
                               clusters=(15, 16, 17),
                               cluster_names=None,
                               gene_colors = None,
                               label_key='pred_label_matchk',
                               min_spots=20, smooth_window=None, log1p=False,
                               figsize=None):
    cluster_names = cluster_names or {}

    means = {cl: cluster_gene_means(adata_all, timepoints, cl, genes,
                                    label_key=label_key, min_spots=min_spots,
                                    log1p=log1p)[0]
             for cl in clusters}

    figsize = figsize or (3.4 * len(genes), 3.4)
    fig, axes = plt.subplots(1, len(genes), figsize=figsize)
    axes = np.atleast_1d(axes)
    import matplotlib.colors as mcolors

    def shades(base, n=3, lo=0.25, hi=1.0):
        """n tints of `base`, palest first."""
        rgb = np.array(mcolors.to_rgb(base))
        return [tuple(1 - f * (1 - rgb)) for f in np.linspace(lo, hi, n)]

    for ax, g in zip(axes, genes):
        ramp = shades(gene_colors[g], n=len(clusters))
        for i, cl in enumerate(clusters):
            if g not in means[cl].columns:
                continue
            if gene_colors is None:
                color = f"C{i}"
            else:
                color = ramp[i]
            label = cluster_names.get(cl, f"cluster {cl}")
            y = means[cl][g]
            if smooth_window:
                ax.plot(y.index, y, "o", ms=3, alpha=0.3, color=color)
                y = y.rolling(smooth_window, center=True, min_periods=1).mean()
                ax.plot(y.index, y, "-", lw=2, color=color, label=label)
            else:
                ax.plot(y.index, y, "-o", ms=3, lw=1.6, color=color, label=label)
        ax.set_title(g)#, fontsize=10)
        ax.set_xlabel("Age (months)")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Mean log1p expression" if log1p
                       else "Mean expression")
    for ax, g in zip(axes, genes):
        ax.legend(frameon=False, loc="upper left",
                    bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    return fig, means