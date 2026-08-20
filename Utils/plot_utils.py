
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import linear_sum_assignment

# This function displays the domain identified


def _safe_crosstab(ad, cluster_key, truth_key):
    df = ad.obs[[cluster_key, truth_key]].copy()
    df = df.dropna(subset=[cluster_key, truth_key])  # CRITICAL
    cser = df[cluster_key].astype("category") if not pd.api.types.is_categorical_dtype(df[cluster_key]) else df[cluster_key]
    tser = df[truth_key].astype("category") if not pd.api.types.is_categorical_dtype(df[truth_key]) else df[truth_key]
    tab = pd.crosstab(cser, tser)
    return tab, list(cser.cat.categories), list(tser.cat.categories)


def plot_slices_horizontal_centered(
    adata_list,
    obs_key,
    coord_keys=("x", "y"),
    obsm_key="spatial",
    order_key=None,
    order_map=None,
    x_gap=None,
    y_scale=1.0,
    s=6,
    alpha=0.9,
    figsize=None,
    legend=True,
    legend_ncol=None,
    legend_loc="center left",
    legend_bbox=(1.02, 0.5),
    legend_font=7,
    legend_markerscale=5,
    show_ari=None,
    truth_key=None,
    align_colors=True,
    truth_cmap="Set3",
    unmatched_cmap="tab20c",
    return_alignment=False,
    color_map=None,
    plot_order=None,
    highlight_labels=None,
    highlight_size_factor=1.8,
    highlight_alpha=None,
    highlight_zorder=5,
):
    """
    color_map behavior:
      - truth_key is None:           color_map keyed by obs_key labels.
      - truth_key == obs_key:        color_map keyed by truth labels.
      - truth_key != obs_key:        color_map is the TRUTH palette; clusters
                                     inherit colors via Hungarian matching.

    New options:
      - plot_order: controls both drawing order and legend order.
                    Later labels are drawn on top.
      - legend_markerscale: controls legend marker size.
      - highlight_labels: labels to emphasize, e.g. ["Oligodendrocyte", "OPC"].
      - highlight_size_factor: multiplicative size boost for highlighted labels.
      - highlight_alpha: alpha for highlighted labels. If None, uses alpha.
      - highlight_zorder: zorder for highlighted labels.
    """

    def _get_xy(ad):
        xk, yk = coord_keys
        if xk in ad.obs.columns and yk in ad.obs.columns:
            xy = ad.obs[[xk, yk]].to_numpy()
        elif obsm_key in ad.obsm_keys():
            xy = np.asarray(ad.obsm[obsm_key])[:, :2]
        else:
            raise KeyError(
                f"Could not find coords in ad.obs[{coord_keys}] nor ad.obsm['{obsm_key}']."
            )
        return np.asarray(xy, dtype=np.float64)

    # decide order of slices
    slice_indices = list(range(len(adata_list)))
    if order_key is not None:
        order_vals = []
        usable = True

        for i, ad in enumerate(adata_list):
            if order_key not in ad.obs.columns:
                usable = False
                break

            v = pd.unique(ad.obs[order_key])
            if len(v) != 1:
                usable = False
                break

            v = v[0]
            if order_map is not None:
                if v not in order_map:
                    usable = False
                    break
                v = order_map[v]

            order_vals.append(v)

        if usable:
            slice_indices = [
                i for i, _ in sorted(
                    zip(slice_indices, order_vals),
                    key=lambda t: t[1],
                )
            ]

    # pass 1: gather per-slice centered coords, widths, and categories
    centered = []
    widths = []
    categories_all = []

    for i in slice_indices:
        ad = adata_list[i]
        xy = _get_xy(ad)
        xy = xy - xy.mean(axis=0, keepdims=True)
        xy[:, 1] *= float(y_scale)

        if obs_key not in ad.obs.columns:
            raise KeyError(f"obs_key='{obs_key}' not found in adata.obs for slice index {i}.")

        c = ad.obs[obs_key]
        if not pd.api.types.is_categorical_dtype(c):
            c = c.astype("category")

        categories_all.extend(list(c.cat.categories))

        w = float(np.nanmax(xy[:, 0]) - np.nanmin(xy[:, 0]))
        widths.append(w)

        centered.append((xy, c, ad))

    if x_gap is None:
        valid_widths = [w for w in widths if np.isfinite(w) and w > 0]
        med_w = np.median(valid_widths) if valid_widths else 1.0
        x_gap = 1.2 * (med_w if np.isfinite(med_w) and med_w > 0 else 1.0)

    # global category set, stable by first appearance
    cat_order_fallback = []
    seen = set()
    for cat in categories_all:
        if cat not in seen:
            seen.add(cat)
            cat_order_fallback.append(cat)

    # ============= PALETTE / ORDER =============
    alignment = None
    contingency = None

    # Case A: no truth_key
    if truth_key is None:
        if color_map is not None:
            user_order = [c for c in color_map.keys() if c in cat_order_fallback]
            extras = [c for c in cat_order_fallback if c not in color_map]
            cat_order = user_order + extras

            fb = plt.get_cmap(unmatched_cmap)
            fbN = getattr(fb, "N", 20)

            color_map = dict(color_map)
            for j, cat in enumerate(extras):
                color_map.setdefault(cat, fb(j % fbN))
        else:
            cat_order = cat_order_fallback
            cmap = plt.get_cmap("tab20")
            color_map = {cat: cmap(i % cmap.N) for i, cat in enumerate(cat_order)}

    # Case B: truth_key given, obs_key == truth_key
    elif align_colors and (obs_key == truth_key):
        cat_order = cat_order_fallback

        if color_map is not None:
            fb = plt.get_cmap(truth_cmap) if isinstance(truth_cmap, str) else truth_cmap
            fbN = getattr(fb, "N", 20)
            color_map = {
                cat: color_map.get(cat, fb(i % fbN))
                for i, cat in enumerate(cat_order)
            }
        else:
            cmap = plt.get_cmap(truth_cmap) if isinstance(truth_cmap, str) else truth_cmap
            cmapN = getattr(cmap, "N", 20)
            color_map = {cat: cmap(i % cmapN) for i, cat in enumerate(cat_order)}

        alignment = {cat: cat for cat in cat_order}

    # Case C: truth_key given, obs_key != truth_key → Hungarian
    elif align_colors:
        for i in slice_indices:
            if truth_key not in adata_list[i].obs.columns:
                raise KeyError(f"truth_key='{truth_key}' not found in slice {i}.")

        clusters_global = []
        truths_global = []
        contingency = None

        for i in slice_indices:
            ad = adata_list[i]
            tab, c_cats, t_cats = _safe_crosstab(ad, obs_key, truth_key)

            for x in c_cats:
                if x not in clusters_global:
                    clusters_global.append(x)
            for x in t_cats:
                if x not in truths_global:
                    truths_global.append(x)

            if contingency is None:
                contingency = tab.copy()
            else:
                contingency = contingency.add(tab, fill_value=0)

        contingency = contingency.reindex(
            index=clusters_global,
            columns=truths_global,
            fill_value=0.0,
        )
        contingency = contingency.fillna(0.0).astype(float)

        M = contingency.to_numpy()
        M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

        if M.sum() <= 0:
            cat_order = cat_order_fallback
            cmap = plt.get_cmap("tab20")
            color_map = {cat: cmap(j % cmap.N) for j, cat in enumerate(cat_order)}
            alignment = {cat: None for cat in cat_order}
        else:
            row_ind, col_ind = linear_sum_assignment(-M)

            clusters = list(contingency.index)
            truths = list(contingency.columns)

            alignment = {cl: None for cl in clusters}
            for r, cidx in zip(row_ind, col_ind):
                alignment[clusters[r]] = truths[cidx]

            truth_rank = {t: k for k, t in enumerate(truths)}
            cat_order = sorted(
                clusters,
                key=lambda cl: (truth_rank.get(alignment.get(cl), 10**9), str(cl)),
            )

            if color_map is not None:
                user_pal = dict(color_map)
                fb = plt.get_cmap(truth_cmap) if isinstance(truth_cmap, str) else truth_cmap
                fbN = getattr(fb, "N", 20)
                truth_palette = {
                    t: user_pal.get(t, fb(j % fbN))
                    for j, t in enumerate(truths)
                }
            else:
                tcmap = plt.get_cmap(truth_cmap) if isinstance(truth_cmap, str) else truth_cmap
                tN = getattr(tcmap, "N", 12)
                truth_palette = {t: tcmap(j % tN) for j, t in enumerate(truths)}

            fb_cmap = plt.get_cmap(unmatched_cmap)
            fb_N = getattr(fb_cmap, "N", 20)

            color_map = {}
            k_un = 0
            for cl in cat_order:
                mt = alignment.get(cl)
                if mt is not None and mt in truth_palette:
                    color_map[cl] = truth_palette[mt]
                else:
                    color_map[cl] = fb_cmap(k_un % fb_N)
                    k_un += 1

    # Case D: align_colors=False with truth_key
    else:
        cat_order = cat_order_fallback
        cmap = plt.get_cmap("tab20")
        color_map = {cat: cmap(i % cmap.N) for i, cat in enumerate(cat_order)}

    # Optional user-defined drawing and legend order
    if plot_order is not None:
        plot_order = list(plot_order)
        ordered = [cat for cat in plot_order if cat in cat_order]
        extras = [cat for cat in cat_order if cat not in ordered]
        cat_order = ordered + extras

    if highlight_labels is None:
        highlight_labels = []
    highlight_labels = set(str(x) for x in highlight_labels)

    # ============= plot =============
    n = len(centered)
    if figsize is None:
        figsize = (max(8, 3.0 * n), 6)

    fig, ax = plt.subplots(figsize=figsize)

    xmin, xmax = np.inf, -np.inf
    ymin, ymax = np.inf, -np.inf

    x_shift = 0.0
    handles = {}

    for (xy, c, ad), w in zip(centered, widths):
        x = xy[:, 0] + x_shift
        y = xy[:, 1]

        xmin = min(xmin, np.nanmin(x))
        xmax = max(xmax, np.nanmax(x))
        ymin = min(ymin, np.nanmin(y))
        ymax = max(ymax, np.nanmax(y))

        cats = pd.Categorical(c, categories=cat_order)

        # Important: loop through cat_order, not pd.unique(cats).
        # Later categories are drawn on top.
        for cat in cat_order:
            mask = (cats == cat)
            if not np.any(mask):
                continue

            if str(cat) in highlight_labels:
                s_use = s * highlight_size_factor
                alpha_use = alpha if highlight_alpha is None else highlight_alpha
                zorder_use = highlight_zorder
            else:
                s_use = s
                alpha_use = alpha
                zorder_use = 2

            sc = ax.scatter(
                x[mask],
                y[mask],
                s=s_use,
                alpha=alpha_use,
                c=[color_map[cat]],
                linewidths=0,
                label=str(cat),
                zorder=zorder_use,
                rasterized=True,
            )

            if cat not in handles:
                handles[cat] = sc

        x_shift += (w + x_gap)

    pad_frac = 0.02
    xr = xmax - xmin
    yr = ymax - ymin
    xpad = pad_frac * xr if xr > 0 else 1.0
    ypad = pad_frac * yr if yr > 0 else 1.0

    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    if legend:
        def _is_integer_like(x):
            try:
                # Handles 1, "1", np.int64(1), categorical labels, etc.
                sx = str(x)
                return sx.lstrip("-").isdigit()
            except Exception:
                return False

        handle_keys = [cat for cat in cat_order if cat in handles]

        if plot_order is not None:
            legend_order = handle_keys
        elif len(handle_keys) > 0 and all(_is_integer_like(cat) for cat in handle_keys):
            legend_order = sorted(handle_keys, key=lambda x: int(str(x)))
        else:
            legend_order = handle_keys

        leg_handles = [handles[cat] for cat in legend_order]
        leg_labels = [str(cat) for cat in legend_order]

        n_items = len(leg_labels)
        ncol = min(n_items, 14) if legend_ncol is None else legend_ncol

        fig.legend(
            leg_handles,
            leg_labels,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox,
            ncol=ncol,
            frameon=False,
            fontsize=legend_font,
            handlelength=0.8,
            handletextpad=0.3,
            columnspacing=0.8,
            borderaxespad=0.0,
            markerscale=legend_markerscale,
        )

        fig.subplots_adjust(top=0.88)

    plt.subplots_adjust(top=0.85)
    fig.tight_layout()

    if return_alignment:
        return fig, ax, alignment, contingency

    return fig, ax



def plot_V_dotplot_from_adata(
    adata_list,
    V_list,
    celltype,
    region_key,
    normalize_V=True,
    celltype_names=None,
    times=None,
    agg="mean",
    region_order=None,
    time_order=None,
    min_size=20,
    max_size=400,
    cmap="Reds",
    ax=None,
    title=None,
    rotate_xticks=True,
    show_colorbar=True,
    size_by="cluster_size",
    fontsize=16,
    time_name = 'Time',
    box_threshold=None
):
    """
    Dot plot of V-derived cell-type proportions.

    x-axis:
        region/cluster label from adata.obs[region_key]

    y-axis:
        time

    dot color:
        aggregated V-derived cell-type proportion for the selected cell type,
        e.g. mean proportion within each region/time pair.

    dot size:
        by default, the size of that region/cluster at that time point,
        i.e. number of spots/cells assigned to the region.

    Parameters
    ----------
    adata_list : list of AnnData
        One AnnData object per time point.

    V_list : list of array-like
        V matrices. Each V should have shape (n_obs, n_celltypes).

    celltype : int or str
        If int, interpreted as column index of V.
        If str, matched against celltype_names.

    region_key : str
        Column in adata.obs containing region/cluster labels.

    celltype_names : list-like, optional
        Names of cell types corresponding to columns of V.

    times : list-like, optional
        Time labels. If None, uses 0, 1, ..., len(adata_list)-1.

    agg : {"mean", "median", "sum"}
        Aggregation used for dot color.

    size_by : {"cluster_size", "same_as_color"}
        If "cluster_size", dot size represents number of spots/cells
        in that region/time pair.
        If "same_as_color", dot size represents the same value as color.

    Returns
    -------
    fig, ax, df

    df contains:
        time
        region
        value
        cluster_size
    """


    # ------------------------------------------------------------
    # Resolve cell type index/name
    # ------------------------------------------------------------
    if celltype_names is not None:
        celltype_names = list(celltype_names)

    if isinstance(celltype, str):
        if celltype_names is None:
            raise ValueError("If celltype is a string, celltype_names must be provided.")

        if celltype not in celltype_names:
            raise ValueError(f"{celltype!r} not found in celltype_names.")

        celltype_idx = celltype_names.index(celltype)
        celltype_name = celltype

    else:
        celltype_idx = int(celltype)

        if celltype_names is not None:
            celltype_name = celltype_names[celltype_idx]
        else:
            celltype_name = f"celltype {celltype_idx}"

    # ------------------------------------------------------------
    # Time labels
    # ------------------------------------------------------------
    if times is None:
        times = list(range(len(adata_list)))

    if len(times) != len(adata_list):
        raise ValueError("times must have the same length as adata_list.")

    if len(V_list) != len(adata_list):
        raise ValueError("V_list must have the same length as adata_list.")
    
    if normalize_V:
        V_list=[V_list[i]/np.sum(V_list[i], axis=1, keepdims=True) for i in range(len(V_list))]

    # ------------------------------------------------------------
    # Build region-time summary directly from spot-level V
    # ------------------------------------------------------------
    rows = []

    for adata, V, t in zip(adata_list, V_list, times):

        if region_key not in adata.obs:
            raise KeyError(f"{region_key!r} not found in adata.obs.")

        V = np.asarray(V)

        if V.ndim != 2:
            raise ValueError(
                "Each V must be a 2D array with shape (n_obs, n_celltypes)."
            )

        if V.shape[0] != adata.n_obs:
            raise ValueError(
                f"V has {V.shape[0]} rows but adata has {adata.n_obs} observations."
            )

        if celltype_idx < 0 or celltype_idx >= V.shape[1]:
            raise IndexError(
                f"celltype_idx={celltype_idx} is out of bounds for V with "
                f"{V.shape[1]} columns."
            )

        region = adata.obs[region_key].astype(str).to_numpy()
        prop = V[:, celltype_idx].astype(float)

        tmp = pd.DataFrame(
            {
                "time": t,
                "region": region,
                "prop": prop,
            }
        )

        def aggregate_prop(x):
            x = np.asarray(x, dtype=float)
            x = x[np.isfinite(x)]

            if len(x) == 0:
                return np.nan

            if agg == "mean":
                return float(np.mean(x))
            elif agg == "median":
                return float(np.median(x))
            elif agg == "sum":
                return float(np.sum(x))
            else:
                raise ValueError("agg must be one of {'mean', 'median', 'sum'}.")

        grouped = (
            tmp
            .groupby(["time", "region"], observed=True)
            .agg(
                value=("prop", aggregate_prop),
                cluster_size=("prop", "size"),
            )
            .reset_index()
        )

        rows.append(grouped)

    df = pd.concat(rows, ignore_index=True)

    # ------------------------------------------------------------
    # Region/time order
    # ------------------------------------------------------------
    if region_order is None:
        try:
            region_order = sorted(df["region"].unique(), key=lambda x: int(x))
        except ValueError:
            region_order = sorted(df["region"].unique())

    if time_order is None:
        time_order = list(dict.fromkeys(df["time"].tolist()))

    df = df[df["region"].isin(region_order)].copy()
    df = df[df["time"].isin(time_order)].copy()

    region_to_x = {r: i for i, r in enumerate(region_order)}
    time_to_y = {t: i for i, t in enumerate(time_order)}

    df["x"] = df["region"].map(region_to_x)
    df["y"] = df["time"].map(time_to_y)

    # ------------------------------------------------------------
    # Color scale
    # ------------------------------------------------------------
    if agg == "sum":
        vmax_color = df["value"].max()
        if not np.isfinite(vmax_color) or vmax_color <= 0:
            vmax_color = 1
    else:
        # For mean/median proportions, use fixed 0-1 scale.
        vmax_color = 1

    # ------------------------------------------------------------
    # Size scale
    # ------------------------------------------------------------
    if size_by == "cluster_size":
        size_value = df["cluster_size"].to_numpy(dtype=float)
        vmax_size = np.nanmax(size_value)

    elif size_by == "same_as_color":
        size_value = df["value"].fillna(0).to_numpy(dtype=float)
        vmax_size = vmax_color

    else:
        raise ValueError("size_by must be one of {'cluster_size', 'same_as_color'}.")

    if not np.isfinite(vmax_size) or vmax_size <= 0:
        sizes = np.full(len(df), min_size)
    else:
        sizes = min_size + (max_size - min_size) * size_value / vmax_size

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    if ax is None:
        fig_width = max(6, 0.55 * len(region_order))
        fig_height = max(3.5, 0.6 * len(time_order))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    else:
        fig = ax.figure

    sc = ax.scatter(
        df["x"],
        df["y"],
        s=sizes,
        c=df["value"],
        cmap=cmap,
        vmin=0,
        vmax=vmax_color,
        edgecolor="black",
        linewidth=0.35,
    )
    # Outline clusters whose mean proportion across time clears the threshold
    if box_threshold is not None:
        col_mean = df.groupby("region")["value"].mean()
        for r, m in col_mean.items():
            if m > box_threshold and r in region_to_x:
                ax.add_patch(plt.Rectangle(
                    (region_to_x[r] - 0.5, -0.5), 1, len(time_order),
                    fill=False, edgecolor="black", linewidth=2, zorder=5,
                ))

    ax.set_xticks(range(len(region_order)))
    ax.set_xticklabels(region_order)

    if rotate_xticks:
        ax.tick_params(axis="x", rotation=45)

    ax.set_yticks(range(len(time_order)))
    ax.set_yticklabels(time_order)

    ax.set_xlabel("Cluster", fontsize=fontsize)
    ax.set_ylabel(time_name, fontsize=fontsize)
    ax.invert_yaxis()

    if title is None:
        title = (
                f"{celltype_name}"
            )
        

    ax.set_title(title, fontsize=fontsize)

    ax.grid(True, linewidth=0.3, alpha=0.35)

    if show_colorbar:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label(f"{agg} W")

    return fig, ax, df



def plot_mass_transport_heatmap(
    adata_insamp,
    V,
    pi,
    t_start=0,
    t_end=None,
    y_gap=500,
    y_scale=0.3,
    figsize=(8, 7),
    ax=None,
    fig=None,
    tight_layout=True,
    # source mass
    source_celltype=None,
    celltype_obs_key="bin_annotation",
    source_mass_source="obs",      # {"obs", "V_argmax"}
    labels=None,
    unit_mass="per_spot",          # kept for signature compatibility; the
                                   # displayed quantity is a ratio, so this
                                   # no longer affects the colors
    # edges
    show_edges=True,
    n_sources_show=300,
    sources_rank_by="mass",        # {"mass", "peakedness"}
    edge_mass_threshold=None,      # None -> use background_threshold
    edge_alpha=0.35,
    edge_lw=0.6,
    edge_scale=1.0,
    edge_same_celltype=False,
    edge_celltype_source="obs",    # {"obs", "V_argmax"}
    edge_celltype=None,
    edge_color_by_celltype=False,
    edge_color="gray",
    edge_same_color="tab:blue",
    edge_different_color="gold",
    seed=0,
    # heatmap style
    cmap="inferno",
    s=18,
    alpha=0.95,
    vmin=0.0,
    vmax=1.0,
    show_colorbar=True,
    colorbar_label=None,
    # background cells
    background_color=plt.get_cmap("Reds")(0.0),
    background_alpha=1.0,
    background_size=None,
    background_threshold=1e-3,
    title=None,
    # coordinate keys
    x_key="x",
    y_key="y",
    eps=1e-12,
    time=None,
):
    """
    Plot stacked slices coloured by ANCESTOR PROBABILITY, with optional
    barycentric-projection edges.

    Colour scale
    ------------
    Every spot is coloured by

        P(spot descends from `source_celltype`)
          =  (mass arriving at this spot that originated in the source type)
             / (total mass arriving at this spot)

    which lies in [0, 1] at every time point and is directly comparable
    across slices, so a single colorbar covers the whole figure.

    At t_start this evaluates to exactly 1 on the source cell type and 0
    everywhere else, so the source slice is binary automatically — no special
    casing needed. Later slices show graded probabilities.

    This is the interpretable quantity. Raw transported mass is not: it is
    summed over incoming sources, so it can exceed 1 where many spots
    converge, and its scale drifts between time points.

    Arrow origins are restricted to spots with ancestor probability greater
    than `edge_mass_threshold`. If `edge_mass_threshold=None`, the heatmap's
    `background_threshold` is used.

    Edge coloring
    -------------
    edge_color_by_celltype=False:
        All retained edges use `edge_color`.

    edge_color_by_celltype=True:
        Same-type edges use `edge_same_color`.
        Different-type edges use `edge_different_color`.

    edge_same_celltype=True:
        Only same-type edges are retained.

    edge_celltype:
        If provided, further restrict arrow origins to that cell type.

    Background
    ----------
    Spots with probability <= background_threshold are drawn using
    `background_color`, which defaults to a neutral grey so it cannot be
    confused with the low end of a warm colormap.
    """
    rng = np.random.default_rng(seed)
    created_ax = ax is None

    if source_celltype is None:
        raise ValueError(
            "Pass source_celltype (e.g. 'Somite') to seed mass at t_start."
        )

    T = len(adata_insamp)

    if t_end is None:
        t_end = T - 1

    if not (0 <= t_start <= t_end < T):
        raise ValueError(
            f"Need 0 <= t_start <= t_end < {T}."
        )

    if edge_mass_threshold is None:
        edge_mass_threshold = background_threshold

    adatas = [
        adata_insamp[t]
        for t in range(t_start, t_end + 1)
    ]

    Vs = [
        np.asarray(V[t])
        for t in range(t_start, t_end + 1)
    ]

    # ---------------------------------------------------------
    # Seed the source indicator at the first displayed slice
    # ---------------------------------------------------------
    ad0 = adatas[0]
    V0 = Vs[0]

    if source_mass_source == "obs":
        if celltype_obs_key not in ad0.obs:
            raise KeyError(
                f"'{celltype_obs_key}' not found in "
                f"adata_insamp[{t_start}].obs."
            )

        obs0 = np.asarray(
            ad0.obs[celltype_obs_key].values
        )
        mask0 = obs0 == source_celltype

    elif source_mass_source == "V_argmax":
        types0 = V0.argmax(axis=1)

        if isinstance(source_celltype, str):
            if labels is None:
                raise ValueError(
                    "source_celltype is a string with "
                    "source_mass_source='V_argmax', but labels=None."
                )

            if source_celltype not in labels:
                raise ValueError(
                    f"source_celltype='{source_celltype}' not in labels."
                )

            src_id = labels.index(source_celltype)
        else:
            src_id = int(source_celltype)

        mask0 = types0 == src_id

    else:
        raise ValueError(
            "source_mass_source must be 'obs' or 'V_argmax'."
        )

    if mask0.sum() == 0:
        raise ValueError(
            f"No spots matched source_celltype='{source_celltype}' "
            f"at t_start={t_start}."
        )

    if unit_mass not in ("per_spot", "probability"):
        raise ValueError(
            "unit_mass must be 'per_spot' or 'probability'."
        )

    # u = mass originating in the source cell type
    # w = mass originating anywhere
    # the displayed value is u / w, so any common rescaling cancels
    u = mask0.astype(float)
    w = np.ones(ad0.n_obs, dtype=float)

    fractions = [u / np.maximum(w, eps)]     # exactly the 0/1 indicator

    # ----------------------------------------------
    # Propagate both numerator and denominator
    # ----------------------------------------------
    for t in range(t_start, t_end):
        Pi = np.asarray(pi[t], dtype=float)
        Pi_sum = Pi.sum()

        if Pi_sum <= eps:
            raise ValueError(
                f"pi[{t}] has non-positive total mass."
            )

        Pi = Pi / Pi_sum

        n_src = adata_insamp[t].n_obs
        n_tgt = adata_insamp[t + 1].n_obs

        if Pi.shape != (n_src, n_tgt):
            raise ValueError(
                f"pi[{t}] has shape {Pi.shape}, "
                f"expected {(n_src, n_tgt)}."
            )

        u = np.asarray(u @ Pi).ravel()
        w = np.asarray(w @ Pi).ravel()

        fractions.append(u / np.maximum(w, eps))

    masses = fractions      # returned under the historical name

    # ---------------------
    # Transform coordinates
    # ---------------------
    coords_transformed, x_all, y_all = (
        build_centered_stacked_coords(
            adatas,
            y_gap=y_gap,
            y_scale=y_scale,
            x_key=x_key,
            y_key=y_key,
            shift_dir="x",
        )
    )

    m_all = np.concatenate(
        fractions,
        axis=0,
    )

    if background_size is None:
        background_size = s

    # ----------------
    # Figure and axes
    # ----------------
    if created_ax:
        fig, ax = plt.subplots(
            figsize=figsize
        )

    elif fig is None:
        fig = ax.figure

    # ---------------------------------
    # Background and ancestor probability
    # ---------------------------------
    active = m_all > background_threshold
    inactive = ~active

    if np.any(inactive):
        ax.scatter(
            x_all[inactive],
            y_all[inactive],
            s=background_size,
            color=background_color,
            alpha=background_alpha,
            linewidths=0,
            zorder=1,
        )

    if np.any(active):
        sc = _sorted_scatter(
            x=x_all[active],
            y=y_all[active],
            c=m_all[active],
            s=s,
            ax=ax,
            cmap=cmap,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
            zorder=2,
        )

    else:
        sc = ax.scatter(
            [],
            [],
            c=[],
            s=s,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    # Label each slice
    for idx, XY in enumerate(
        coords_transformed
    ):
        if time is None:
            ax.text(
                float(np.min(XY[:, 0])),
                float(np.min(XY[:, 1])) - 0.05 * y_gap,
                f"t={t_start + idx}",
                fontsize=15,
                ha="left",
                va="top",
            )
        else:
            ax.text(
                float(np.min(XY[:, 0])),
                float(np.min(XY[:, 1])) - 0.05 * y_gap,
                f"t={time[t_start + idx]}",
                fontsize=15,
                ha="left",
                va="top",
            )

    # ---------------------------------
    # Draw barycentric-projection edges
    # ---------------------------------
    if show_edges:

        def _types_for_edges(global_t):
            if edge_celltype_source == "V_argmax":
                return np.asarray(
                    V[global_t]
                ).argmax(axis=1)

            if edge_celltype_source == "obs":
                adata_t = adata_insamp[global_t]

                if celltype_obs_key not in adata_t.obs:
                    raise KeyError(
                        f"'{celltype_obs_key}' not found in "
                        f"adata_insamp[{global_t}].obs."
                    )

                return np.asarray(
                    adata_t.obs[celltype_obs_key].values
                )

            raise ValueError(
                "edge_celltype_source must be "
                "'obs' or 'V_argmax'."
            )

        edge_type = edge_celltype

        if (
            edge_celltype_source == "V_argmax"
            and isinstance(edge_celltype, str)
        ):
            if labels is None:
                raise ValueError(
                    "edge_celltype is a string with "
                    "edge_celltype_source='V_argmax', "
                    "but labels=None."
                )

            if edge_celltype not in labels:
                raise ValueError(
                    f"edge_celltype='{edge_celltype}' "
                    "not in labels."
                )

            edge_type = labels.index(
                edge_celltype
            )

        for local_idx, global_t in enumerate(
            range(t_start, t_end)
        ):
            XY0 = coords_transformed[
                local_idx
            ]
            XY1 = coords_transformed[
                local_idx + 1
            ]

            Pi = np.asarray(
                pi[global_t],
                dtype=float,
            )
            Pi_sum = Pi.sum()

            if Pi_sum <= eps:
                continue

            Pi = Pi / Pi_sum

            P = _row_stochastic_from_pi(
                Pi,
                eps=eps,
            )

            n_src, n_tgt = P.shape

            if (
                XY0.shape[0] != n_src
                or XY1.shape[0] != n_tgt
            ):
                raise ValueError(
                    f"Coordinate mismatch at t={global_t}: "
                    f"P={P.shape}, "
                    f"XY0={XY0.shape}, "
                    f"XY1={XY1.shape}."
                )

            m_t = np.asarray(
                fractions[local_idx]
            ).ravel()

            if m_t.size != n_src:
                raise ValueError(
                    "Mass vector size does not match "
                    "the rows of P."
                )

            src_types = _types_for_edges(
                global_t
            )
            tgt_types = _types_for_edges(
                global_t + 1
            )

            # Only spots with meaningful source ancestry can be arrow origins.
            # At t_start this is exactly the source cell type.
            src_all = np.flatnonzero(
                m_t > edge_mass_threshold
            )

            # Optional additional cell-type restriction
            if edge_type is not None:
                src_all = src_all[
                    src_types[src_all] == edge_type
                ]

            if src_all.size == 0:
                continue

            # Rank the eligible source spots
            if sources_rank_by == "mass":
                scores = m_t[src_all]

            elif sources_rank_by == "peakedness":
                scores = P[src_all].max(
                    axis=1
                )

            else:
                raise ValueError(
                    "sources_rank_by must be "
                    "'mass' or 'peakedness'."
                )

            if n_sources_show is not None:
                n_show = min(
                    int(n_sources_show),
                    src_all.size,
                )

                if n_show <= 0:
                    continue

                jitter = (
                    1e-15
                    * rng.random(src_all.size)
                )

                if n_show < src_all.size:
                    pick = np.argpartition(
                        scores + jitter,
                        -n_show,
                    )[-n_show:]

                    src_sel = src_all[pick]
                else:
                    src_sel = src_all

            else:
                src_sel = src_all

            # Barycentric target coordinates
            Ybar = P[src_sel] @ XY1

            # Representative target cell type:
            # type of target receiving the most mass
            j_star = P[src_sel].argmax(
                axis=1
            )
            target_types_selected = (
                tgt_types[j_star]
            )

            for idx_i, i in enumerate(
                src_sel
            ):
                same_type = (
                    src_types[i]
                    == target_types_selected[idx_i]
                )

                if (
                    edge_same_celltype
                    and not same_type
                ):
                    continue

                if edge_color_by_celltype:
                    current_edge_color = (
                        edge_same_color
                        if same_type
                        else edge_different_color
                    )
                else:
                    current_edge_color = (
                        edge_color
                    )

                x0, y0 = XY0[i]
                x1, y1 = Ybar[idx_i]

                dx = (
                    x1 - x0
                ) * edge_scale
                dy = (
                    y1 - y0
                ) * edge_scale

                ax.annotate(
                    "",
                    xy=(
                        x0 + dx,
                        y0 + dy,
                    ),
                    xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        lw=edge_lw,
                        alpha=edge_alpha,
                        color=current_edge_color,
                        # mutation_scale=15,  # controls arrowhead size 
                    ),
                    zorder=3,
                )

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)

    if show_colorbar:
        sc.set_clim(vmin, vmax)

        cbar = fig.colorbar(
            sc,
            ax=ax,
            fraction=0.03,
            pad=0.01,
        )

        if colorbar_label is None:
            colorbar_label = f"Transported mass"

        cbar.set_label(colorbar_label)
        cbar.set_ticks(
            np.linspace(vmin, vmax, 6)
        )

    if tight_layout and created_ax:
        fig.tight_layout()

    return fig, ax, masses


def build_centered_stacked_coords(adatas, y_gap=500, y_scale=0.3, x_key="x", y_key="y", shift_dir='y'):
    """
    Implements exactly the coordinate transform you posted.

    Returns
    -------
    coords_transformed : list[np.ndarray]
        List of XY arrays, one per time, each of shape (n_spots_t, 2).
    x_all, y_all : np.ndarray
        Concatenated coordinates across all times.
    """
    n_time = len(adatas)

    all_xy_raw = np.vstack([
        ad.obs[[x_key, y_key]].to_numpy()
        - ad.obs[[x_key, y_key]].to_numpy().mean(axis=0, keepdims=True)
        for ad in adatas
    ])
    center = all_xy_raw.mean(axis=0)

    coords_transformed = []
    for idx, ad in enumerate(adatas):
        xy = ad.obs[[x_key, y_key]].to_numpy() - ad.obs[[x_key, y_key]].to_numpy().mean(axis=0, keepdims=True)
        xy_centered = xy - center
        xy_squash = np.column_stack([xy_centered[:, 0], y_scale * xy_centered[:, 1]])
        offset_y = ((n_time - 1) / 2.0 - idx) * y_gap
        if shift_dir == 'y':
            xy_ellip = xy_squash + np.array([0.0, offset_y])
        else:
            xy_ellip = xy_squash + np.array([-offset_y,0.0])
        coords_transformed.append(xy_ellip)

    x_all = np.concatenate([XY[:, 0] for XY in coords_transformed])
    y_all = np.concatenate([XY[:, 1] for XY in coords_transformed])
    return coords_transformed, x_all, y_all


def _row_stochastic_from_pi(pi, eps=1e-12):
    pi = np.asarray(pi, dtype=float)
    den = pi.sum(axis=1, keepdims=True)
    den = np.maximum(den, eps)
    return pi / den


def _sorted_scatter(ax, x, y, c, s=18, cmap="inferno", alpha=0.95, vmin=None, vmax=None, **kwargs):
    # low first, high last => hotspots drawn on top
    x = np.asarray(x); y = np.asarray(y); c = np.asarray(c, float)
    order = np.argsort(c)
    sc = ax.scatter(
        x[order], y[order],
        c=c[order],
        s=s,
        cmap=cmap,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        **kwargs
    )
    return sc

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def get_xy(a):
    if hasattr(a, "obsm") and "spatial" in a.obsm:
        return np.asarray(a.obsm["spatial"], float)
    return a.obs[["x", "y"]].to_numpy(float)


def panel(slice_obj, gene, coords=None, basis=None, target_sum=1e4):
    """(centered coords, expression vector) for one gene in one slice."""
    X = slice_obj.X if coords is None else slice_obj
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X, float)
    names = list(basis.columns) if basis is not None else list(slice_obj.var_names)
    if basis is not None:
        X = X @ basis.to_numpy(float)

    X = X * target_sum / np.maximum(X.sum(1, keepdims=True), 1e-12)
    xy = np.asarray(coords, float) if coords is not None else get_xy(slice_obj)
    return xy - xy.mean(0), X[:, names.index(gene)]


def plot_gene_panels(gene, rows, spot_size=5, cmap="Reds", q=0.99,
                     color_mode="global", labels=None):
    """rows: {method -> [(coords, values), ...]}, one entry per time point."""
    methods = list(rows)
    n = len(rows[methods[0]])
    flat = [p for m in methods for p in rows[m]]

    lim = lambda i: 1.03 * np.abs(np.concatenate([X[:, i] for X, _ in flat])).max()
    vmax = lambda v: max(np.quantile(v, q), 1e-9)

    if color_mode == "global":
        g = vmax(np.concatenate([v for _, v in flat]))
        scale = {(m, c): g for m in methods for c in range(n)}
    elif color_mode == "column":
        col = [vmax(np.concatenate([rows[m][c][1] for m in methods])) for c in range(n)]
        scale = {(m, c): col[c] for m in methods for c in range(n)}
    else:
        scale = {(m, c): vmax(rows[m][c][1]) for m in methods for c in range(n)}

    fig, axes = plt.subplots(len(methods), n, figsize=(2 * n, 1.7 * len(methods)),
                             sharex=True, sharey=True, squeeze=False,
                             constrained_layout=True)

    for r, m in enumerate(methods):
        for c, (X, v) in enumerate(rows[m]):
            ax = axes[r, c]
            v = np.clip(v / scale[(m, c)], 0, 1)
            o = np.argsort(v)                      # hot spots on top
            sc = ax.scatter(X[o, 0], X[o, 1], c=v[o], s=spot_size, cmap=cmap,
                            norm=Normalize(0, 1), edgecolors="none")
            ax.set(xlim=(-lim(0), lim(0)), ylim=(-lim(1), lim(1)),
                   xticks=[], yticks=[], aspect="equal")
            ax.axis("off")
            if r == 0:
                ax.set_title((labels or [f"$t={c}$" for c in range(n)])[c], fontsize=14)
            if c == 0:
                ax.set_ylabel(m, fontsize=13, fontweight="bold", rotation=0,
                              ha="right", va="center", labelpad=12)

    cbar = fig.colorbar(sc, ax=axes, shrink=0.75, label="Expression")
    cbar.set_ticks([0, 1]); cbar.set_ticklabels(["Low", "High"])
    fig.suptitle(f"{gene}: spatial expression", fontsize=16, fontweight="bold")
    return fig