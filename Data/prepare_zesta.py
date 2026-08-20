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
# Your data preparation (as-is)
# -----------------------------
def prepare_data(threshold=1.25,data_dir=None,return_sc=False):
    print(data_dir)
    if data_dir == None:
        data_dir = "../../"
    adata_scRNA3  = sc.read_h5ad(os.path.join(data_dir, "zf3_scRNA.h5ad"))
    adata_scRNA3.X = adata_scRNA3.layers["counts"].copy()

    adata_scRNA5  = sc.read_h5ad(os.path.join(data_dir, "zf5_scRNA.h5ad"))
    adata_scRNA5.X = adata_scRNA5.layers["counts"].copy()

    adata_scRNA10 = sc.read_h5ad(os.path.join(data_dir, "zf10_scRNA.h5ad"))
    adata_scRNA10.X = adata_scRNA10.layers["counts"].copy()

    adata_scRNA12 = sc.read_h5ad(os.path.join(data_dir, "zf12_scRNA.h5ad"))
    adata_scRNA12.X = adata_scRNA12.layers["counts"].copy()

    adata_scRNA18 = sc.read_h5ad(os.path.join(data_dir, "zf18_scRNA.h5ad"))
    adata_scRNA18.X = adata_scRNA18.layers["counts"].copy()

    adata_scRNA24 = sc.read_h5ad(os.path.join(data_dir, "zf24_scRNA.h5ad"))
    adata_scRNA24.X = adata_scRNA24.layers["counts"].copy()

    def fix_ann(adata):
        for k, v in adata.obsm.items():
            if hasattr(v, "values"):
                adata.obsm[k] = v.values
        for k, v in adata.varm.items():
            if hasattr(v, "values"):
                adata.varm[k] = v.values
        return adata

    adata_list = [adata_scRNA3, adata_scRNA5, adata_scRNA10, adata_scRNA12, adata_scRNA18, adata_scRNA24]
    adata_list = [fix_ann(a) for a in adata_list]
    adata_scRNA = sc.concat(adata_list)
    adata_scRNA.obs_names_make_unique()

    

    sc.pp.normalize_total(adata_scRNA, target_sum=1e4)
    sc.pp.filter_genes(adata_scRNA, min_counts=1)

    del adata_scRNA3, adata_scRNA5, adata_scRNA10, adata_scRNA12, adata_scRNA18, adata_scRNA24

    zf3 = sc.read_h5ad(os.path.join(data_dir, "zf3_stereoseq.h5ad"))
    zf3 = zf3[zf3.obs["slice"] == 1]
    zf3.X = zf3.layers["counts"].copy()

    zf5 = sc.read_h5ad(os.path.join(data_dir, "zf5_stereoseq.h5ad"))
    zf5 = zf5[zf5.obs["slice"] == 10]
    zf5.X = zf5.layers["counts"].copy()

    zf10 = sc.read_h5ad(os.path.join(data_dir, "zf10_stereoseq.h5ad"))
    zf10 = zf10[(zf10.obs["slice"] == 11) | (zf10.obs["slice"] == 17)]
    zf10.X = zf10.layers["counts"].copy()

    zf12 = sc.read_h5ad(os.path.join(data_dir, "zf12_stereoseq.h5ad"))
    zf12 = zf12[(zf12.obs["slice"] == 8) | (zf12.obs["slice"] == 5)]
    zf12.X = zf12.layers["counts"].copy()

    zf18 = sc.read_h5ad(os.path.join(data_dir, "zf18_stereoseq.h5ad"))
    zf18 = zf18[(zf18.obs["slice"] == 8) | (zf18.obs["slice"] == 11)]
    zf18.X = zf18.layers["counts"].copy()

    zf24 = sc.read_h5ad(os.path.join(data_dir, "zf24_stereoseq.h5ad"))
    zf24 = zf24[zf24.obs["slice"] == 4]
    zf24.X = zf24.layers["counts"].copy()

    adata = sc.concat((zf3, zf5, zf10, zf12, zf18, zf24))
    adata.obs_names_make_unique()
    del zf3, zf5, zf10, zf12, zf18, zf24

    adata.obsm["spatial"] = adata.obs[["spatial_x", "spatial_y"]].values
    adata.obs["Batch"] = adata.obs["time_point"].astype(str) + " slice " + adata.obs["slice"].astype(str)

    adata.obsm["spatial"][adata.obs["Batch"] == "18hpf slice 8"]  = adata.obsm["spatial"][adata.obs["Batch"] == "18hpf slice 8"]  @ np.array([[0, 1], [1, 0]])
    adata.obsm["spatial"][adata.obs["Batch"] == "10hpf slice 17"] = adata.obsm["spatial"][adata.obs["Batch"] == "10hpf slice 17"] @ np.array([[0, 1], [1, 0]])

    theta = -7 * np.pi / 4
    adata.obsm["spatial"][adata.obs["Batch"] == "12hpf slice 5"] *= [1, -1]
    adata.obsm["spatial"][adata.obs["Batch"] == "12hpf slice 5"] = adata.obsm["spatial"][adata.obs["Batch"] == "12hpf slice 5"] @ np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    theta = -np.pi / 8
    adata.obsm["spatial"][adata.obs["Batch"] == "12hpf slice 8"] = adata.obsm["spatial"][adata.obs["Batch"] == "12hpf slice 8"] @ np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    adata.obsm["spatial"][adata.obs["Batch"] == "10hpf slice 11"] *= [-1, 1]
    adata.obsm["spatial"][adata.obs["Batch"] == "12hpf slice 8"] *= [-1, 1]
    adata.obsm["spatial"][adata.obs["Batch"] == "24hpf slice 4"] *= [-1, 1]
    adata.obsm["spatial"][adata.obs["Batch"] == "18hpf slice 11"] *= [1, -1]

    adata.obs["x"] = adata.obsm["spatial"][:, 0]
    adata.obs["y"] = adata.obsm["spatial"][:, 1]
    adata_scRNA.obs = adata_scRNA.obs.rename(columns={"celltype_new": "cellType"})

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.filter_genes(adata, min_counts=1)

    adata_all = []
    for b in adata.obs["Batch"].unique():
        idx = adata.obs["Batch"] == b
        adata_temp = adata[idx].copy()
        adata_temp.obs["x"] = adata_temp.obsm["spatial"][:, 0]
        adata_temp.obs["y"] = adata_temp.obsm["spatial"][:, 1]
        adata_all.append(adata_temp)

    slices = [0, 1, 2, 5, 6, 8]
    adata_insamp = [adata_all[i] for i in slices]

    B = find_B_music(adata_scRNA)
    pd_B = B.T
    counts, cell_types = from_anndata(adata_scRNA, celltype_key="cellType")
    adata_scRNA.X = adata_scRNA.layers["counts"].copy()
    genes_subset = select_informative_genes(pd_B, counts, cell_types, fc_thresh_ln=threshold)
    pd_B = pd_B.loc[:, genes_subset]

    if return_sc == False:
        return adata_insamp, pd_B
    else:
        return adata_insamp, pd_B, adata_scRNA