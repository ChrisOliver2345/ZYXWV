import sys
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import get_RP_score
import re
import os
from utils import Preprocessor, Binning
import numpy as np

def load_data(args):
    modal_a_path = "./Datasets/" + args.dateset + "/" + args.modal_a_file
    modal_b_path = "./Datasets/" + args.dateset + "/" + args.modal_b_file

    modal_a_adata = sc.read_h5ad(modal_a_path)
    modal_b_adata = sc.read_h5ad(modal_b_path)

    return modal_a_adata, modal_b_adata


class SCDataset(Dataset):
    def __init__(self, modal_a_matrix, modal_a_label, modal_b_matrix, modal_b_label, device):
        super(SCDataset, self).__init__()
        self.mt_a = modal_a_matrix
        self.l_a = modal_a_label
        self.mt_b = modal_b_matrix
        self.l_b = modal_b_label

        self.length = len(self.l_a)
        self.device = device

    def __len__(self):
        return self.length

def preprocess_data(modal_a_adata, modal_b_adata, args):

    expr_matrix_b = modal_b_adata.X.toarray()
    cell_names_b = modal_b_adata.obs.index
    gene_names_b = modal_b_adata.var.index

    modal_b_matrix = pd.DataFrame(
        data=expr_matrix_b,
        index=cell_names_b,
        columns=gene_names_b
    )

    if args.modal_a == 'scRNA-seq':
        pass

    if args.modal_b == 'scATAC-seq':

        def is_underscore_format(peak):
            return re.match(r'^chr[\w\d]+_\d+_\d+$', peak) is not None

        if not all(is_underscore_format(col) for col in modal_b_matrix.columns):
            formatted_columns = [peak.replace(":", "_").replace("-", "_") for peak in modal_b_matrix.columns]
            modal_b_matrix.columns = formatted_columns

        cache_dir = f"./TEMP/{args.dateset}"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"rp_score.h5")

        if os.path.exists(cache_file):
            print(f"Loading cached RP score matrix from {cache_file}")
            rp_score_matrix = pd.read_hdf(cache_file, key='rp_score')
        else:
            print(f"Computing RP score matrix for {args.dateset}")
            rp_score_matrix = get_RP_score(
                atac_df=modal_b_matrix,
                organism="GRCh38",
                decaydistance=10000,
                model="Simple"
            )
            rp_score_matrix.to_hdf(cache_file, key='rp_score', mode='w')
            print(f"Saved RP score matrix to {cache_file}")

    adata_a_lm = modal_a_adata

    adata_b_lm = ad.AnnData(
        X=rp_score_matrix.values,
        obs=modal_b_adata.obs.loc[rp_score_matrix.index],
        var=pd.DataFrame(index=rp_score_matrix.columns)
    )

    data_is_raw = True
    preprocessor_a = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=50,  # step 1
        filter_cell_by_counts=50,  # step 2
        normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=True,  # 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        subset_hvg=2000,  # 5. whether to subset the raw data to highly variable genes
        hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger"
    )

    data_is_raw = False
    preprocessor_b = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=False,  # step 1
        filter_cell_by_counts=False,  # step 2
        normalize_total=False,  # 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=True,  # 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        subset_hvg=2000,  # 5. whether to subset the raw data to highly variable genes
        hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger"
    )

    adata_a_lm = preprocessor_a(adata_a_lm, batch_key=None)

    common_genes = np.intersect1d(adata_a_lm.var_names, adata_b_lm.var_names)
    adata_a_lm = adata_a_lm[:, common_genes].copy()
    adata_b_lm = adata_b_lm[:, common_genes].copy()

    adata_b_lm = preprocessor_b(adata_b_lm, batch_key=None)

    hvg_a = adata_a_lm.var[adata_a_lm.var['highly_variable']].index
    hvg_b = adata_b_lm.var[adata_b_lm.var['highly_variable']].index
    hvg_union = np.union1d(hvg_a, hvg_b)

    adata_a_lm = adata_a_lm[:, hvg_union].copy()
    adata_b_lm = adata_b_lm[:, hvg_union].copy()

    binning = Binning(binning=args.n_bins, result_binned_key="X_binned")

    adata_a_bin, adata_b_bin = binning(adata_a_lm, adata_b_lm)

    return adata_a_lm, adata_b_lm
