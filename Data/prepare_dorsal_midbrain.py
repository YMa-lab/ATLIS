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



# -----------------------------
# Data prep (yours, minimal edits)
# -----------------------------
def prepare_data(fname, threshold=1.25, return_sc=False):
    
    adata = sc.read_h5ad(fname)
    adata.X = adata.layers["counts"].copy()

    
    sc.pp.filter_genes(adata, min_counts=1)

    batches_out = ['SS200000131BL_C3C4', 'SS200000131BL_C5C6']
    batches = ['FP200000600TR_E3', 'SS200000108BR_B1B2', 'SS200000108BR_A3A4']

    adata_all = []
    for t in batches:
        print(t)
        idx = adata.obs["Batch"] == t
        print(adata[idx].obs['Time point'][0])
        adata_temp = adata[idx].copy()
        adata_temp.obs['x'] = adata_temp.obsm['spatial'][:, 0]
        adata_temp.obs['y'] = adata_temp.obsm['spatial'][:, 1]
        adata_all.append(adata_temp)

    adata_all_out = []
    for t in batches_out:
        print(t)
        idx = adata.obs["Batch"] == t
        print(adata[idx].obs['Time point'][0])
        adata_temp = adata[idx].copy()
        sc.pp.normalize_total(adata_temp, target_sum=1e4)
        adata_temp.obs['x'] = adata_temp.obsm['spatial'][:, 0]
        adata_temp.obs['y'] = adata_temp.obsm['spatial'][:, 1]
        adata_all_out.append(adata_temp)

    adata_out = ad.concat(adata_all_out)
    adata_out.obs['cellType'] = adata_out.obs['annotation']

    B = find_B_music(adata_out)
    pd_B = B.T

    adata_out.X = adata_out.layers["counts"].copy()
    counts, cell_types = from_anndata(adata_out, celltype_key="cellType")
    genes_subset = select_informative_genes(pd_B, counts, cell_types, fc_thresh_ln=threshold)
    print(len(genes_subset))
    pd_B = pd_B.loc[:, genes_subset]

    # sc.pp.normalize_total(adata_out, target_sum=1e4)
    if return_sc == False:
        return adata_all, pd_B
    else:
        return adata_all, pd_B, adata_out