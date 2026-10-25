import os
import argparse
import traceback
from pathlib import Path
import pickle
import json
import numpy as np
import scanpy as sc
import anndata as ad


import sys
sys.path.append('../Utils')
from objective import solve_V_all_slices
from preprocessing import find_B_music, select_informative_genes, from_anndata
import re

# def prepare_data(fname, threshold=1.25, return_sc=False, renormalize=False):

#     adata = sc.read_h5ad(fname)
#     adata.X = adata.layers["counts"].copy()

#     sc.pp.filter_genes(adata, min_counts=1)

#     # Drop mitochondrial and ribosomal genes before anything downstream.
#     # drop = (
#     #     adata.var_names.str.match(r'^Hb[ab]-', case=False)
#     #     | adata.var_names.str.fullmatch(r'Hbb-bs|Hbb-bt|Hbb-y|Hbb-bh1|Alas2', case=False)
#     #     # adata.var_names.str.match(r'^mt-', case=False)
#     #     # | adata.var_names.str.match(r'^Rp[ls]\d', case=False)
#     #     # | adata.var_names.str.match(r'^Mrp[ls]\d', case=False)
#     # )
#     # print(f"dropping {drop.sum()} mt/ribosomal genes")
#     # adata = adata[:, ~drop].copy()

#     batches_out = ['SS200000131BL_C3C4', 'SS200000131BL_C5C6']
#     batches = ['FP200000600TR_E3', 'SS200000108BR_B1B2', 'SS200000108BR_A3A4']

#     adata_all = []
#     for t in batches:
#         print(t)
#         idx = adata.obs["Batch"] == t
#         print(adata[idx].obs['Time point'][0])
#         adata_temp = adata[idx].copy()
#         adata_temp.obs['x'] = adata_temp.obsm['spatial'][:, 0]
#         adata_temp.obs['y'] = adata_temp.obsm['spatial'][:, 1]
#         adata_all.append(adata_temp)

#     adata_all_out = []
#     for t in batches_out:
#         print(t)
#         idx = adata.obs["Batch"] == t
#         print(adata[idx].obs['Time point'][0])
#         adata_temp = adata[idx].copy()
#         sc.pp.normalize_total(adata_temp, target_sum=1e4)
#         adata_temp.obs['x'] = adata_temp.obsm['spatial'][:, 0]
#         adata_temp.obs['y'] = adata_temp.obsm['spatial'][:, 1]
#         adata_all_out.append(adata_temp)

#     adata_out = ad.concat(adata_all_out)
#     adata_out.obs['cellType'] = adata_out.obs['annotation']

#     adata_out.X = adata_out.layers["counts"].copy()
#     B = find_B_music(adata_out)
#     pd_B = B.T

    
#     counts, cell_types = from_anndata(adata_out, celltype_key="cellType")
#     genes_subset = select_informative_genes(pd_B, counts, cell_types, fc_thresh_ln=threshold)
#     print(len(genes_subset))
#     pd_B = pd_B.loc[:, genes_subset]
#     common = [g for g in genes_subset if g in set(adata_all[0].var_names)]
#     pd_B = pd_B.loc[:, common]
#     adata_all = [a[:, common].copy() for a in adata_all]



#     if renormalize:
        
#         for a in adata_all:
#             sc.pp.normalize_total(a, target_sum=1e4)
#         pd_B = pd_B.div(pd_B.sum(axis=1), axis=0) * 1e4

#     if return_sc == False:
#         return adata_all, pd_B
#     else:
#         return adata_all, pd_B, adata_out

def prepare_data(fname, threshold=1.25, return_sc=False, reference_subset=True, renormalize=False):

    adata = sc.read_h5ad(fname)
    adata.X = adata.layers["counts"].copy()
    sc.pp.filter_genes(adata, min_counts=1)

    batches_out = ['SS200000131BL_C3C4', 'SS200000131BL_C5C6']
    batches = ['FP200000600TR_E3', 'SS200000108BR_B1B2', 'SS200000108BR_A3A4']

    adata_all = []
    for t in batches:
        idx = adata.obs["Batch"] == t
        adata_temp = adata[idx].copy()
        adata_temp.obs['x'] = adata_temp.obsm['spatial'][:, 0]
        adata_temp.obs['y'] = adata_temp.obsm['spatial'][:, 1]
        adata_all.append(adata_temp)

    adata_all_out = []
    for t in batches_out:
        idx = adata.obs["Batch"] == t
        adata_temp = adata[idx].copy()
        adata_temp.obs['x'] = adata_temp.obsm['spatial'][:, 0]
        adata_temp.obs['y'] = adata_temp.obsm['spatial'][:, 1]
        adata_all_out.append(adata_temp)

    adata_out = ad.concat(adata_all_out)
    adata_out.obs['cellType'] = adata_out.obs['annotation']
    adata_out.X = adata_out.layers["counts"].copy()

    # --- Step 1: subset reference + spatial to common (panel) genes ---
    common_panel = set(adata_out.var_names)
    for a in adata_all:
        common_panel &= set(a.var_names)
    common_panel = sorted(common_panel)

    adata_out = adata_out[:, common_panel].copy()
    if reference_subset:
        adata_all = [a[:, common_panel].copy() for a in adata_all]

    # --- Step 2: normalize reference AND spatial together, same gene scope ---
    sc.pp.normalize_total(adata_out, target_sum=1e4)
    for a in adata_all:
        sc.pp.normalize_total(a, target_sum=1e4)

    # --- Step 3: compute B from normalized, panel-scoped reference ---
    B = find_B_music(adata_out)
    pd_B = B.T

    # --- Step 4: select informative genes on this scoped B ---
    # from_anndata / select_informative_genes presumably want counts;
    # decide whether they need raw counts or normalized — use adata_out.layers["counts"]
    # restricted to common_panel if raw counts are required internally.
    counts, cell_types = from_anndata(adata_out, celltype_key="cellType")
    genes_subset = select_informative_genes(pd_B, counts, cell_types, fc_thresh_ln=threshold)
    print(len(genes_subset))

    # --- Step 5: subset B and spatial to informative genes ---
    pd_B = pd_B.loc[:, genes_subset]
    adata_all = [a[:, genes_subset].copy() for a in adata_all]

    # --- Step 6: renormalize on the final, smaller gene set ---
    if renormalize:
        for a in adata_all:
            sc.pp.normalize_total(a, target_sum=1e4)
        pd_B = pd_B.div(pd_B.sum(axis=1), axis=0) * 1e4

    if return_sc == False:
        return adata_all, pd_B
    else:
        return adata_all, pd_B, adata_out