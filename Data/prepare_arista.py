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


def prepare_reference(data_dir, threshold=1.25):
    # Load the dataset.
    if data_dir == None:
        data_dir = "../../"

    dpi2_rep2 = sc.read_h5ad(
        os.path.join(data_dir, "2DPI_2.h5ad") 
    )
    dpi2_rep2.obsm["spatial"] = dpi2_rep2.obsm["spatial"] * np.array([-1, -1])
    dpi2_rep2.X = dpi2_rep2.layers["counts"].copy()
    print("loaded DPI2")

    dpi2_rep2 = dpi2_rep2[dpi2_rep2.obs["inj_uninj"] == "inj"]
    dpi2_rep2 = dpi2_rep2[dpi2_rep2.obs["D_V"] == "D"]

    dpi5_rep2 = sc.read_h5ad(
         os.path.join(data_dir, "5DPI_2.h5ad") 
    )
    dpi5_rep2.obsm["spatial"] = dpi5_rep2.obsm["spatial"] * np.array([-1, -1])
    dpi5_rep2.X = dpi5_rep2.layers["counts"].copy()
    print("loaded DPI5")

    dpi5_rep2 = dpi5_rep2[dpi5_rep2.obs["inj_uninj"] == "inj"]
    dpi5_rep2 = dpi5_rep2[dpi5_rep2.obs["D_V"] == "D"]

    dpi10_rep2 = sc.read_h5ad(
         os.path.join(data_dir, "10DPI_2.h5ad") 
    )
    dpi10_rep2.obsm["spatial"] = dpi10_rep2.obsm["spatial"] * np.array([-1, -1])
    dpi10_rep2.X = dpi10_rep2.layers["counts"].copy()
    dpi10_rep2 = dpi10_rep2[
        np.arange(dpi10_rep2.n_obs) != np.argmin(dpi10_rep2.obsm["spatial"][:, 0])
    ]
    
    print("loaded DPI10")

    dpi10_rep2 = dpi10_rep2[dpi10_rep2.obs["inj_uninj"] == "inj"]
    dpi10_rep2 = dpi10_rep2[dpi10_rep2.obs["D_V"] == "D"]


    dpi15_rep4 = sc.read_h5ad(
         os.path.join(data_dir, "15DPI_4.h5ad") 
    )
    dpi15_rep4.obsm["spatial"] = dpi15_rep4.obsm["spatial"] * np.array([-1, -1])
    dpi15_rep4 = dpi15_rep4[
        np.arange(dpi15_rep4.n_obs) != np.argmax(dpi15_rep4.obsm["spatial"][:, 1])
    ]
    dpi15_rep4.X = dpi15_rep4.layers["counts"].copy()
    print("loaded DPI15")

    dpi15_rep4 = dpi15_rep4[dpi15_rep4.obs["inj_uninj"] == "inj"]
    dpi15_rep4 = dpi15_rep4[dpi15_rep4.obs["D_V"] == "D"]


    dpi20_rep3 = sc.read_h5ad(
         os.path.join(data_dir, "20DPI_3.h5ad") 
    )
    dpi20_rep3.obsm["spatial"] = dpi20_rep3.obsm["spatial"] * np.array([-1, -1])
    dpi20_rep3.X = dpi20_rep3.layers["counts"].copy()
    dpi20_rep3 = dpi20_rep3[
        np.arange(dpi20_rep3.n_obs) != np.argmin(dpi20_rep3.obsm["spatial"][:, 0])
    ]
   
    print("loaded DPI20")

    dpi20_rep3 = dpi20_rep3[dpi20_rep3.obs["inj_uninj"] == "inj"]
    dpi20_rep3 = dpi20_rep3[dpi20_rep3.obs["D_V"] == "D"]

    

    adata = sc.concat((dpi2_rep2, dpi5_rep2, dpi10_rep2, dpi15_rep4, dpi20_rep3))
    adata.obs_names_make_unique()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.calculate_qc_metrics(adata, inplace=True)

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_cells(adata, max_genes=adata.obs["n_genes_by_counts"].quantile(0.999))
    sc.pp.filter_genes(adata, min_cells=3)

    adata.obs["x"] = adata.obsm["spatial"][:, 0]
    adata.obs["y"] = adata.obsm["spatial"][:, 1]

    adata.obs = adata.obs.rename(columns={"Annotation": "cellType"})
    
    return adata



def prepare_data(data_dir=None, threshold=1.25, return_sc=False):
    if data_dir == None:
        data_dir = "../../"

    # Load the dataset.
    dpi2_rep1 = sc.read_h5ad(
         os.path.join(data_dir, "2DPI_1.h5ad")
    )
    dpi2_rep1.obsm["spatial"] = dpi2_rep1.obsm["spatial"] * np.array([-1, -1])
    dpi2_rep1.X = dpi2_rep1.layers["counts"].copy()
    print("loaded DPI2")

    dpi2_rep1 = dpi2_rep1[dpi2_rep1.obs["inj_uninj"] == "inj"]
    dpi2_rep1 = dpi2_rep1[dpi2_rep1.obs["D_V"] == "D"]

    dpi5_rep1 = sc.read_h5ad(
       os.path.join(data_dir, "5DPI_1.h5ad")
    )
    dpi5_rep1.obsm["spatial"] = dpi5_rep1.obsm["spatial"] * np.array([-1, -1])
    dpi5_rep1.X = dpi5_rep1.layers["counts"].copy()
    print("loaded DPI5")

    dpi5_rep1 = dpi5_rep1[dpi5_rep1.obs["inj_uninj"] == "inj"]
    dpi5_rep1 = dpi5_rep1[dpi5_rep1.obs["D_V"] == "D"]

    dpi10_rep1 = sc.read_h5ad(
        os.path.join(data_dir, "10DPI_1.h5ad")
    )
    dpi10_rep1.obsm["spatial"] = dpi10_rep1.obsm["spatial"] * np.array([-1, -1])
    dpi10_rep1.X = dpi10_rep1.layers["counts"].copy()
    dpi10_rep1 = dpi10_rep1[
        np.arange(dpi10_rep1.n_obs) != np.argmin(dpi10_rep1.obsm["spatial"][:, 0])
    ]
    dpi10_rep1 = dpi10_rep1[
        np.arange(dpi10_rep1.n_obs) != np.argmin(dpi10_rep1.obsm["spatial"][:, 0])
    ]
    dpi10_rep1 = dpi10_rep1[
        np.arange(dpi10_rep1.n_obs) != np.argmin(dpi10_rep1.obsm["spatial"][:, 0])
    ]
    dpi10_rep1 = dpi10_rep1[
        np.arange(dpi10_rep1.n_obs) != np.argmin(dpi10_rep1.obsm["spatial"][:, 0])
    ]
    print("loaded DPI10")

    dpi10_rep1 = dpi10_rep1[dpi10_rep1.obs["inj_uninj"] == "inj"]
    dpi10_rep1 = dpi10_rep1[dpi10_rep1.obs["D_V"] == "D"]

    dpi15_rep3 = sc.read_h5ad(
        os.path.join(data_dir, "15DPI_3.h5ad")
    )
    dpi15_rep3.obsm["spatial"] = dpi15_rep3.obsm["spatial"] * np.array([-1, -1])
    dpi15_rep3 = dpi15_rep3[
        np.arange(dpi15_rep3.n_obs) != np.argmax(dpi15_rep3.obsm["spatial"][:, 1])
    ]
    dpi15_rep3.X = dpi15_rep3.layers["counts"].copy()
    print("loaded DPI15")

    dpi15_rep3 = dpi15_rep3[dpi15_rep3.obs["inj_uninj"] == "inj"]
    dpi15_rep3 = dpi15_rep3[dpi15_rep3.obs["D_V"] == "D"]

    dpi20_rep2 = sc.read_h5ad(
       os.path.join(data_dir, "20DPI_2.h5ad")
    )
    dpi20_rep2.obsm["spatial"] = dpi20_rep2.obsm["spatial"] * np.array([-1, -1])
    dpi20_rep2.X = dpi20_rep2.layers["counts"].copy()
    dpi20_rep2 = dpi20_rep2[
        np.arange(dpi20_rep2.n_obs) != np.argmin(dpi20_rep2.obsm["spatial"][:, 0])
    ]
    dpi20_rep2 = dpi20_rep2[
        np.arange(dpi20_rep2.n_obs) != np.argmin(dpi20_rep2.obsm["spatial"][:, 0])
    ]
    print("loaded DPI20")

    dpi20_rep2 = dpi20_rep2[dpi20_rep2.obs["inj_uninj"] == "inj"]
    dpi20_rep2 = dpi20_rep2[dpi20_rep2.obs["D_V"] == "D"]

    dpi30 = sc.read_h5ad(
        os.path.join(data_dir, "30DPI.h5ad")
    )
    dpi30.obsm["spatial"] = dpi30.obsm["spatial"] * np.array([-1, -1])
    dpi30.X = dpi30.layers["counts"].copy()
    print("loaded DPI30")

    dpi30 = dpi30[dpi30.obs["inj_uninj"] == "inj"]
    dpi30 = dpi30[dpi30.obs["D_V"] == "D"]

    adata = sc.concat((dpi2_rep1, dpi5_rep1, dpi10_rep1, dpi15_rep3, dpi20_rep2, dpi30))
    adata.obs_names_make_unique()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.calculate_qc_metrics(adata, inplace=True)

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_cells(adata, max_genes=adata.obs["n_genes_by_counts"].quantile(0.999))
    sc.pp.filter_genes(adata, min_cells=3)

    adata.obs["x"] = adata.obsm["spatial"][:, 0]
    adata.obs["y"] = adata.obsm["spatial"][:, 1]

    adata.obs = adata.obs.rename(columns={"Annotation": "cellType"})


    adata_all = []
    for b in adata.obs["Batch"].unique():
        idx = adata.obs["Batch"] == b
        adata_temp = adata[idx].copy()
        adata_temp.obs["x"] = adata_temp.obsm["spatial"][:, 0]
        adata_temp.obs["y"] = adata_temp.obsm["spatial"][:, 1]
        adata_all.append(adata_temp)

    adata_sc = prepare_reference(data_dir = data_dir, threshold=threshold)
    

    B = find_B_music(adata_sc)
    pd_B = B.T


    adata_sc.X = adata_sc.layers["counts"].copy()
    counts, cell_types = from_anndata(adata_sc, celltype_key="cellType")
    genes_subset = select_informative_genes(pd_B, counts, cell_types, fc_thresh_ln=threshold)
    pd_B = pd_B.loc[:, genes_subset]
    if return_sc == False:
        return adata_all, pd_B
    else:
        return adata_all, pd_B, adata_sc