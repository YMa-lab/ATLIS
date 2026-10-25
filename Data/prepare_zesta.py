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


def merge_spatial_celltypes(labels):
    MERGE_MAP = {
        # Blastomere / blastoderm
        "deep blastomere": "Blastomere",
        "superficial blastomere": "Blastomere",
        "marginal blastomere": "Blastomere",

        # Mesoderm / tail bud
        "presumptive mesoderm": "Mesoderm",
        "hypoblast": "Mesoderm",
        "lateral plate mesoderm": "Mesoderm",
        "segmental plate, tail bud": "Mesoderm",

        # Surface ectoderm
        "presumptive ectoderm": "Epidermis",
        "evl": "Epidermis",
        "periderm": "Epidermis",
        "epidermis": "Epidermis",
        "yolk syncytial layer, periderm": "Epidermis",

        # Neural rod / CNS
        "neural rod": "Neural",
        "anterior neural keel": "Neural",
        "posterior neural keel": "Neural",
        "nervous system": "Neural",
        "forebrain": "Neural",
        "spinal cord ventral region": "Neural",
        "spinal cord dorsal region": "Neural",
        "spinal cord anterior region": "Neural",

        # Muscle / somite
        "adaxial cell": "Muscle",
        "somite": "Muscle",
        "myotome": "Muscle",
        "musculature system": "Muscle",
        "fast muscle cell": "Muscle",
        "slow muscle cell": "Muscle",

        # Eye
        "optic vesicle": "Eye",
        "immature eye, midbrain": "Eye",
        "eye": "Eye",

        # Erythroid
        "erythroid lineage cell": "Erythroid",
        "anterior erythroid lineage cell": "Erythroid",
        "posterior erythroid lineage cell": "Erythroid",

        # Neural crest / otic
        "neural crest": "Neural Crest",
        "neural crest, otic vesicle": "Neural Crest",
        "otic vesicle": "Neural Crest",
    }

    return pd.Series([MERGE_MAP.get(str(x).strip().lower(), str(x).strip()) for x in labels], index=labels.index if hasattr(labels, "index") else None)


def merge_celltypes(labels, drop_multitissue=False):
    """Collapse reference annotations into broader cell types."""
    import re

    MERGE_RULES = [
        # Blastoderm
        ("Blastomere", [r"^early blastodisc$", r"^late blastodisc$"]),

        # Epidermis / surface ectoderm
        ("Epidermis", [r"^periderm", r"^epidermis", r"^integument", r"^evl"]),

        # Neural
        ("Neural", [
            r"^central nervous system",
            r"^brain, spinal cord$",
            r"^neural plate, neural keel$",
            r"^anterior neural keel$",
            r"^neural rod$",
            r"^anterior neural rod$",
            r"^posterior neural rod$",
            r"^neural stem cells$",
            r"^primary neuron$",
            r"^gabaergic neuron$",
        ]),

        # Mesoderm / tail bud
        ("Mesoderm", [
            r"^presumptive mesoderm$",
            r"^segmental plate, tail bud$",
            r"^paraxial mesoderm, tail bud$",
            r"^lateral plate mesoderm$",
            r"^tail bud$",
        ]),

        # Muscle / somite
        ("Muscle", [
            r"^musculature system$",
            r"^pharyngeal arch, somite, cardiovascular system$",
        ]),

        # Eye
        ("Eye", [r"^optic cup$", r"^solid lens vesicle$"]),

        # Erythroid
        ("Erythroid", [r"^erythroid lineage cell$"]),

        # Neural crest
        ("Neural Crest", [r"^neural crest$"]),
    ]

    def canon(label):
        if pd.isna(label):
            return None

        s = str(label).strip()
        low = s.lower()

        if drop_multitissue and "," in s:
            return None

        for name, patterns in MERGE_RULES:
            if any(re.search(p, low) for p in patterns):
                return name

        return s

    return pd.Series([canon(l) for l in labels], index=labels.index if hasattr(labels, "index") else None)



def prepare_data(threshold=1.25, data_dir=None, return_sc=False, reference_subset=True, renormalize=False, filter_celltypes=True, return_full=False):
    print(data_dir)
    if data_dir is None:
        data_dir = "../../"

    # --- load & concat scRNA (unchanged) ---
    adata_scRNA3  = sc.read_h5ad(os.path.join(data_dir, "zf3_scRNA.h5ad"));  adata_scRNA3.X  = adata_scRNA3.layers["counts"].copy()
    adata_scRNA5  = sc.read_h5ad(os.path.join(data_dir, "zf5_scRNA.h5ad"));  adata_scRNA5.X  = adata_scRNA5.layers["counts"].copy()
    adata_scRNA10 = sc.read_h5ad(os.path.join(data_dir, "zf10_scRNA.h5ad")); adata_scRNA10.X = adata_scRNA10.layers["counts"].copy()
    adata_scRNA12 = sc.read_h5ad(os.path.join(data_dir, "zf12_scRNA.h5ad")); adata_scRNA12.X = adata_scRNA12.layers["counts"].copy()
    adata_scRNA18 = sc.read_h5ad(os.path.join(data_dir, "zf18_scRNA.h5ad")); adata_scRNA18.X = adata_scRNA18.layers["counts"].copy()
    adata_scRNA24 = sc.read_h5ad(os.path.join(data_dir, "zf24_scRNA.h5ad")); adata_scRNA24.X = adata_scRNA24.layers["counts"].copy()

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
    adata_scRNA.obs = adata_scRNA.obs.rename(columns={"celltype_new": "cellType"})
    if filter_celltypes:
        # --- collapse subtypes / drop ambiguous multi-tissue labels ---
        adata_scRNA.obs["cellType"] = merge_celltypes(adata_scRNA.obs["cellType"])
        adata_scRNA = adata_scRNA[adata_scRNA.obs["cellType"].notna()].copy()
        print('cell types filtered')


    sc.pp.filter_genes(adata_scRNA, min_counts=1)
    # NOTE: normalize_total deliberately deferred until after gene-scoping below.
    del adata_scRNA3, adata_scRNA5, adata_scRNA10, adata_scRNA12, adata_scRNA18, adata_scRNA24

    # --- load & assemble spatial (unchanged geometry/transform logic) ---
    zf3  = sc.read_h5ad(os.path.join(data_dir, "zf3_stereoseq.h5ad"));  zf3  = zf3[zf3.obs["slice"] == 1].copy();  zf3.X  = zf3.layers["counts"].copy()
    zf5  = sc.read_h5ad(os.path.join(data_dir, "zf5_stereoseq.h5ad"));  zf5  = zf5[zf5.obs["slice"] == 10].copy(); zf5.X  = zf5.layers["counts"].copy()
    zf10 = sc.read_h5ad(os.path.join(data_dir, "zf10_stereoseq.h5ad")); zf10 = zf10[(zf10.obs["slice"] == 11) | (zf10.obs["slice"] == 17)].copy(); zf10.X = zf10.layers["counts"].copy()
    zf12 = sc.read_h5ad(os.path.join(data_dir, "zf12_stereoseq.h5ad")); zf12 = zf12[(zf12.obs["slice"] == 8)  | (zf12.obs["slice"] == 5)].copy();  zf12.X = zf12.layers["counts"].copy()
    zf18 = sc.read_h5ad(os.path.join(data_dir, "zf18_stereoseq.h5ad")); zf18 = zf18[(zf18.obs["slice"] == 8)  | (zf18.obs["slice"] == 11)].copy(); zf18.X = zf18.layers["counts"].copy()
    zf24 = sc.read_h5ad(os.path.join(data_dir, "zf24_stereoseq.h5ad")); zf24 = zf24[zf24.obs["slice"] == 4].copy(); zf24.X = zf24.layers["counts"].copy()

    adata = sc.concat((zf3, zf5, zf10, zf12, zf18, zf24))
    adata.obs_names_make_unique()
    if filter_celltypes:
        adata.obs["cellType"] = merge_spatial_celltypes(adata.obs["bin_annotation"])
    
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
    sc.pp.filter_genes(adata, min_counts=1)

    # if filter_celltypes:
    #     annotation_col = "bin_annotation"  # <-- confirm this is the right column name
    #     adata.obs[annotation_col] = merge_spatial_celltypes(
    #         adata.obs[annotation_col],
    #     )

    # NOTE: normalize_total deferred until after gene-scoping below.

    adata_all = []
    for b in adata.obs["Batch"].unique():
        idx = adata.obs["Batch"] == b
        adata_temp = adata[idx].copy()
        adata_temp.obs["x"] = adata_temp.obsm["spatial"][:, 0]
        adata_temp.obs["y"] = adata_temp.obsm["spatial"][:, 1]
        adata_all.append(adata_temp)

    slices = [0, 1, 2, 5, 6, 8]
    if return_full:
        adata_full = [adata_all[i] for i in slices]

    # --- Step 1: subset reference + all spatial slices to common panel genes ---
    common_panel = set(adata_scRNA.var_names)
    for a in adata_all:
        common_panel &= set(a.var_names)
    common_panel = sorted(common_panel)

    adata_scRNA = adata_scRNA[:, common_panel].copy()
    if reference_subset:
        adata_all = [a[:, common_panel].copy() for a in adata_all]

    # --- Step 2: normalize reference AND spatial together, same gene scope ---
    sc.pp.normalize_total(adata_scRNA, target_sum=1e4)
    for a in adata_all:
        sc.pp.normalize_total(a, target_sum=1e4)

    # --- Step 3: compute B from normalized, panel-scoped reference ---
    B = find_B_music(adata_scRNA)
    pd_B = B.T

    # --- Step 4: select informative genes ---
    # confirm whether from_anndata/select_informative_genes want raw counts;
    # if so, pass adata_scRNA.layers["counts"][:, common_panel] instead.
    counts, cell_types = from_anndata(adata_scRNA, celltype_key="cellType")
    genes_subset = select_informative_genes(pd_B, counts, cell_types, fc_thresh_ln=threshold)
    print(len(genes_subset))

    # --- Step 5: subset B and ALL spatial slices to informative genes ---
    pd_B = pd_B.loc[:, genes_subset]
    adata_all = [a[:, genes_subset].copy() for a in adata_all]

    

    # slice AFTER gene-subsetting, not before, so adata_insamp matches pd_B
    adata_insamp = [adata_all[i] for i in slices]
    

    # --- Step 6: renormalize on the final, smaller gene set ---
    if renormalize:
        for a in adata_insamp:
            sc.pp.normalize_total(a, target_sum=1e4)
        pd_B = pd_B.div(pd_B.sum(axis=1), axis=0) * 1e4

    if (return_sc == False) and (return_full == False):
        return adata_insamp, pd_B
    elif (return_sc == True) and  (return_full == False):
        return adata_insamp, pd_B, adata_scRNA
    elif (return_sc == False) and  (return_full == True):
        return adata_insamp, pd_B, adata_full
    else:
        return adata_insamp, pd_B, adata_scRNA, adata_full