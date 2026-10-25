import os
import argparse
import traceback
from pathlib import Path
import pickle
import json
import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd

import sys
sys.path.append('../Utils')
from objective import solve_V_all_slices
from preprocessing import find_B_music, select_informative_genes, from_anndata



def _stratified_subsample_adata(adata, n_obs=10000, group_key="celltype", random_state=42):
    """
    Subsample AnnData while approximately preserving the proportion of `group_key`.

    Parameters
    ----------
    adata : AnnData
    n_obs : int
        Target number of cells.
    group_key : str
        Column in adata.obs used for stratification.
    random_state : int

    Returns
    -------
    AnnData
        Subsampled AnnData object.
    """
    if adata.n_obs <= n_obs:
        return adata.copy()

    if group_key not in adata.obs.columns:
        raise ValueError(f"`{group_key}` not found in adata.obs")

    rng = np.random.default_rng(random_state)

    group_sizes = adata.obs[group_key].value_counts().sort_index()
    props = group_sizes / group_sizes.sum()

    # Initial proportional allocation
    raw_alloc = props * n_obs
    alloc = np.floor(raw_alloc).astype(int)

    # Distribute leftover cells by largest remainder
    remainder = raw_alloc - alloc
    leftover = n_obs - alloc.sum()
    if leftover > 0:
        extra_groups = remainder.sort_values(ascending=False).index[:leftover]
        alloc.loc[extra_groups] += 1

    # Safety: cannot sample more than available in any group
    alloc = np.minimum(alloc, group_sizes)

    # If clipping caused us to fall below n_obs, redistribute again
    deficit = n_obs - alloc.sum()
    if deficit > 0:
        capacity = group_sizes - alloc
        expandable = capacity[capacity > 0].sort_values(ascending=False).index.tolist()
        i = 0
        while deficit > 0 and len(expandable) > 0:
            g = expandable[i % len(expandable)]
            if alloc[g] < group_sizes[g]:
                alloc[g] += 1
                deficit -= 1
            i += 1

    sampled_idx = []
    for g, k in alloc.items():
        if k <= 0:
            continue
        group_idx = np.where(adata.obs[group_key].values == g)[0]
        chosen = rng.choice(group_idx, size=k, replace=False)
        sampled_idx.append(chosen)

    sampled_idx = np.concatenate(sampled_idx)
    rng.shuffle(sampled_idx)

    return adata[sampled_idx].copy()
# # -----------------------------
# # Data prep (yours, minimal edits)
# # -----------------------------
# def prepare_data(threshold=1.25, return_sc=False, data_dir=None, reference="agg", renormalize=False):
#     if data_dir is None:
#         data_dir = "../../"

#     adata = sc.read_h5ad(os.path.join(data_dir, "aging_coronal.h5ad"))
#     sc.pp.filter_genes(adata, min_counts=1)

#     ages = pd.to_numeric(adata.obs["age"], errors="coerce")
#     timepoints = np.sort(ages.dropna().unique())

#     adata_all = []
#     for t in timepoints:
#         a = adata[adata.obs["age"] == t].copy()
#         if a.n_obs > 10000:
#             a = _stratified_subsample_adata(a, n_obs=10000, random_state=42)
#         a.obs["x"], a.obs["y"] = a.obs["center_x"], a.obs["center_y"]
#         # a.obsm["spatial"] = a.obs[["x", "y"]].to_numpy(float)
#         a.layers["counts"] = a.X.copy()
#         sc.pp.normalize_total(a, target_sum=1e4)
#         adata_all.append(a)

#     spot_genes = adata_all[0].var_names

#     # Loom is read either way: for expression under "agg", for the
#     # ClusterName -> TaxonomyRank4 lookup under "cells".
    

#     if reference == "agg":
#         ref_obj = sc.read_loom(os.path.join(data_dir, "l5_all.agg.loom"))
#         X = ref_obj.X if isinstance(ref_obj.X, np.ndarray) else ref_obj.X.toarray()
#         labels = ref_obj.obs["TaxonomyRank4"].astype(str).values
#         w = pd.to_numeric(ref_obj.obs["NCells"], errors="coerce").fillna(0).values
#         keep = (labels != "nan") & (w > 0)

#         df = pd.DataFrame(X[keep], index=labels[keep],
#                           columns=ref_obj.var_names.astype(str))
#         df = df.T.groupby(level=0).sum().T          # collapse duplicate genes
#         pd_B = df.mul(w[keep], axis=0).groupby(level=0).sum().div(
#             pd.Series(w[keep], index=df.index).groupby(level=0).sum(), axis=0)
#         pd_B = pd_B.div(pd_B.sum(axis=1), axis=0) * 1e4

#     elif reference == "cells":
#         # Normalize per cell *before* averaging by type — the aggregate above
#         # can't do this, its averaging already happened.
#         ref_obj = sc.read_loom(os.path.join(data_dir, "l5_all.agg.loom"))
#         sc_data = sc.read_h5ad(
#             os.path.join(data_dir, "adult_mouse_nervous_system.h5ad"))
#         sc_data.obs_names_make_unique()
#         sc_data.var_names = sc_data.var["Gene"].astype(str).values
#         sc_data.var_names_make_unique()

#         lut = (ref_obj.obs[["ClusterName", "TaxonomyRank4"]].astype(str)
#                .drop_duplicates().set_index("ClusterName")["TaxonomyRank4"])
#         sc_data.obs["Rank4"] = sc_data.obs["CellType"].astype(str).map(lut)
#         sc_data = sc_data[sc_data.obs["Rank4"].notna()].copy()
#         # drop = ["Spinal cord excitatory neurons", "Sympathetic cholinergic neurons"]
#         # sc_data = sc_data[~sc_data.obs["Rank4"].isin(drop)].copy()

#         sc.pp.filter_genes(sc_data, min_counts=1)
#         sc.pp.normalize_total(sc_data, target_sum=1e4)
#         pd_B = find_B_music(sc_data, ct_varname="Rank4").T
#         ref_obj = sc_data


#     else:
#         raise ValueError("reference must be 'agg' or 'cells'.")

#     pd_B = pd_B.loc[:, pd_B.columns.intersection(spot_genes)]
#     pd_B = pd_B.loc[:, select_informative_genes(pd_B, fc_thresh_ln=threshold)]
#     pd_B = pd_B.div(pd_B.sum(axis=1), axis=0) * 1e4          # <-- add
#     genes_subset = list(pd_B.columns)
    
#     if renormalize:
#         common = [g for g in genes_subset if g in set(adata_all[0].var_names)]
#         pd_B = pd_B.loc[:, common]
#         adata_all = [a[:, common].copy() for a in adata_all]
#         for a in adata_all:
#             sc.pp.normalize_total(a, target_sum=1e4)
#         pd_B = pd_B.div(pd_B.sum(axis=1), axis=0) * 1e4

#     if return_sc:
#         return adata_all, pd_B, ref_obj, timepoints
#     return adata_all, pd_B, timepoints


# def prepare_data_free(data_dir=None):
#     if data_dir is None:
#         data_dir = "../../"

#     adata = sc.read_h5ad(os.path.join(data_dir, "aging_coronal.h5ad"))
#     sc.pp.filter_genes(adata, min_counts=1)

#     ages = pd.to_numeric(adata.obs["age"], errors="coerce")
#     timepoints = np.sort(ages.dropna().unique())

#     adata_all = []
#     for t in timepoints:
#         a = adata[adata.obs["age"] == t].copy()
#         if a.n_obs > 10000:
#             a = _stratified_subsample_adata(a, n_obs=10000, random_state=42)
#         a.obs["x"], a.obs["y"] = a.obs["center_x"], a.obs["center_y"]
#         # a.obsm["spatial"] = a.obs[["x", "y"]].to_numpy(float)
#         a.layers["counts"] = a.X.copy()
#         sc.pp.normalize_total(a, target_sum=1e4)
#         adata_all.append(a)


   
#     return adata_all, timepoints


def prepare_data(threshold=1.25, return_sc=False, data_dir=None, reference="agg", renormalize=False):
    if data_dir is None:
        data_dir = "../../"

    adata = sc.read_h5ad(os.path.join(data_dir, "aging_coronal.h5ad"))
    sc.pp.filter_genes(adata, min_counts=1)

    ages = pd.to_numeric(adata.obs["age"], errors="coerce")
    timepoints = np.sort(ages.dropna().unique())

    adata_all = []
    for t in timepoints:
        a = adata[adata.obs["age"] == t].copy()
        if a.n_obs > 10000:
            a = _stratified_subsample_adata(a, n_obs=10000, random_state=42)
        a.obs["x"], a.obs["y"] = a.obs["center_x"], a.obs["center_y"]
        a.layers["counts"] = a.X.copy()
        # NOTE: normalize_total deferred until after gene-scoping to
        # genes_subset below, so spatial and pd_B end up on the same scale.
        adata_all.append(a)

    # All adata_all[i] share var_names (they're subsets of one loaded h5ad),
    # so index 0 is a valid stand-in for the full spot panel.
    spot_genes = adata_all[0].var_names

    if reference == "agg":
        ref_obj = sc.read_loom(os.path.join(data_dir, "l5_all.agg.loom"))
        X = ref_obj.X if isinstance(ref_obj.X, np.ndarray) else ref_obj.X.toarray()
        labels = ref_obj.obs["TaxonomyRank4"].astype(str).values
        w = pd.to_numeric(ref_obj.obs["NCells"], errors="coerce").fillna(0).values
        keep = (labels != "nan") & (w > 0)

        # NOTE: this reference is pre-aggregated (cluster-level mean expression,
        # NCells-weighted) — there is no raw per-cell data here to re-scope
        # before averaging. Gene-subsetting and rescaling can only happen
        # after the fact, which is a structural limitation of this loom file,
        # not something fixable by reordering these steps.
        df = pd.DataFrame(X[keep], index=labels[keep],
                          columns=ref_obj.var_names.astype(str))
        df = df.T.groupby(level=0).sum().T          # collapse duplicate genes
        pd_B = df.mul(w[keep], axis=0).groupby(level=0).sum().div(
            pd.Series(w[keep], index=df.index).groupby(level=0).sum(), axis=0)
        pd_B = pd_B.div(pd_B.sum(axis=1), axis=0) * 1e4

    elif reference == "cells":
        ref_obj = sc.read_loom(os.path.join(data_dir, "l5_all.agg.loom"))
        sc_data = sc.read_h5ad(
            os.path.join(data_dir, "adult_mouse_nervous_system.h5ad"))
        sc_data.obs_names_make_unique()
        sc_data.var_names = sc_data.var["Gene"].astype(str).values
        sc_data.var_names_make_unique()

        lut = (ref_obj.obs[["ClusterName", "TaxonomyRank4"]].astype(str)
               .drop_duplicates().set_index("ClusterName")["TaxonomyRank4"])
        sc_data.obs["Rank4"] = sc_data.obs["CellType"].astype(str).map(lut)
        sc_data = sc_data[sc_data.obs["Rank4"].notna()].copy()

        sc.pp.filter_genes(sc_data, min_counts=1)

        # --- scope to spot_genes BEFORE normalizing, so per-cell size
        #     factors are computed only over genes the spatial panel has ---
        common_ref_genes = sc_data.var_names.intersection(spot_genes)
        sc_data = sc_data[:, common_ref_genes].copy()

        sc.pp.normalize_total(sc_data, target_sum=1e4)
        pd_B = find_B_music(sc_data, ct_varname="Rank4").T
        ref_obj = sc_data

    else:
        raise ValueError("reference must be 'agg' or 'cells'.")

    pd_B = pd_B.loc[:, pd_B.columns.intersection(spot_genes)]
    # pd_B = pd_B.loc[:, select_informative_genes(pd_B, fc_thresh_ln=threshold)]
    pd_B = pd_B.div(pd_B.sum(axis=1), axis=0) * 1e4
    genes_subset = list(pd_B.columns)

    # --- scope + normalize spatial data to genes_subset unconditionally,
    #     so adata_all and pd_B are always on a matched scale, regardless
    #     of the renormalize flag ---
    adata_all = [a[:, genes_subset].copy() for a in adata_all]
    for a in adata_all:
        sc.pp.normalize_total(a, target_sum=1e4)

    if renormalize:
        # pd_B and adata_all are already scoped/normalized above; this
        # re-application is a no-op given the current genes_subset, but is
        # kept for parity with the other prepare_data variants.
        pd_B = pd_B.div(pd_B.sum(axis=1), axis=0) * 1e4

    if return_sc:
        return adata_all, pd_B, ref_obj, timepoints
    return adata_all, pd_B, timepoints


def prepare_data_free(data_dir=None):
    if data_dir is None:
        data_dir = "../../"

    adata = sc.read_h5ad(os.path.join(data_dir, "aging_coronal.h5ad"))
    sc.pp.filter_genes(adata, min_counts=1)

    ages = pd.to_numeric(adata.obs["age"], errors="coerce")
    timepoints = np.sort(ages.dropna().unique())

    adata_all = []
    for t in timepoints:
        a = adata[adata.obs["age"] == t].copy()
        if a.n_obs > 10000:
            a = _stratified_subsample_adata(a, n_obs=10000, random_state=42)
        a.obs["x"], a.obs["y"] = a.obs["center_x"], a.obs["center_y"]
        a.layers["counts"] = a.X.copy()
        sc.pp.normalize_total(a, target_sum=1e4)
        adata_all.append(a)

    return adata_all, timepoints