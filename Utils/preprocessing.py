import anndata as ad
import numpy as np
import scipy.sparse as sp
import scipy.spatial
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from sklearn.neighbors import NearestNeighbors
import re
_MONTHS = "(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
_DATE_LIKE_RE = re.compile(rf"^\d{{1,2}}-{_MONTHS}$")



def from_anndata(adata_scRNA, celltype_key="cellType", layer_key="counts"):
    X = adata_scRNA.layers[layer_key] if layer_key in adata_scRNA.layers else adata_scRNA.X
    if sp.issparse(X):
        X = X.toarray()

    counts = pd.DataFrame(
        X.T, index=adata_scRNA.var_names, columns=adata_scRNA.obs_names
    )
    cell_types = adata_scRNA.obs[celltype_key].astype(str).to_numpy()
    return counts, cell_types



def select_informative_genes(
    B,
    counts=None,
    cell_types=None,
    common_genes=None,
    fc_thresh_ln=1.25,
    dispersion_quantile=1.0,
    drop_date_like=True,
    cell_type_key=None
):
    eps = 1e-6
    gene_set = set()

    # Step 1: fold-change filter
    for ict in B.index:
        rest = B.drop(index=ict).mean(axis=0)
        FC = np.log(B.loc[ict] + eps) - np.log(rest + eps)
        keep = (FC > fc_thresh_ln) & (B.loc[ict] > 0)
        gene_set.update(B.columns[keep])

    genes = sorted(gene_set)

    if common_genes is not None:
        genes = [g for g in genes if g in common_genes]

    if len(genes) == 0:
        return []

    # If no dispersion filtering, skip counts entirely
    if dispersion_quantile >= 1.0:
        genes_final = genes
    else:
        if counts is None or cell_types is None:
            raise ValueError("counts and cell_types are required when dispersion_quantile < 1.0")

        counts_sub = counts.loc[counts.index.intersection(genes)]

        ct_series = pd.Series(cell_types, index=counts.columns)
        valid_cts = ct_series.value_counts()
        valid_cts = valid_cts[valid_cts > 1].index

        disp = []
        for ct in valid_cts:
            cols = ct_series[ct_series == ct].index
            X = counts_sub[cols]

            mean = X.mean(axis=1)
            var = X.var(axis=1)
            disp.append(var / (mean + eps))

        if len(disp) == 0:
            genes_final = genes
        else:
            disp = pd.concat(disp, axis=1)
            mean_disp = disp.mean(axis=1, skipna=True)
            cutoff = mean_disp.quantile(dispersion_quantile)
            genes_final = mean_disp[mean_disp < cutoff].index.tolist()

    if drop_date_like:
        genes_final = [
            g for g in genes_final
            if not _DATE_LIKE_RE.fullmatch(str(g))
        ]

    return sorted(genes_final)

def find_B_music(adata, ct_varname="cellType", return_parts=False, eps=1e-12):
    """Build a MuSiC-style reference: Theta_gk = m_k * mu_gk.

    mu_gk is the relative expression of gene g within cell type k (columns
    sum to 1), and m_k the average library size of cells of type k — so Theta
    carries both the expression profile and the cell-size factor that MuSiC
    uses to weight types.

    `adata.X` should be raw counts; log1p breaks the library-size term.

    Returns Theta (genes x cell types), or (Theta, parts) when `return_parts`,
    where parts holds mu_df and m_series.
    """
    X = adata.X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)

    genes = np.asarray(adata.var_names)
    cell_types = np.asarray(adata.obs[ct_varname])
    cts_unique = pd.unique(cell_types)

    libsizes = X.sum(axis=1)

    mu_cols, m_vals = [], []

    for k in cts_unique:
        mask = cell_types == k
        if not mask.any():
            mu_cols.append(pd.Series(0.0, index=genes, name=k))
            m_vals.append(0.0)
            continue

        mean_counts = X[mask].mean(axis=0)
        total = mean_counts.sum()
        mu_k = (
            mean_counts / (total + eps) if total > eps
            else np.zeros_like(mean_counts, dtype=float)
        )

        mu_cols.append(pd.Series(mu_k, index=genes, name=k))
        m_vals.append(float(libsizes[mask].mean()))

    mu_df = pd.concat(mu_cols, axis=1)
    m_series = pd.Series(m_vals, index=cts_unique)

    Theta_df = mu_df * m_series.values

    if return_parts:
        return Theta_df, dict(mu_df=mu_df, m_series=m_series)
    return Theta_df