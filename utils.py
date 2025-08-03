import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse
import random
import torch
from anndata import AnnData
import scanpy as sc
from scanpy.get import _get_obs_rep, _set_obs_rep
from scipy.sparse import issparse


def get_RP_score(atac_df, organism="GRCh38", decaydistance=10000, model="Simple"):
    # Simple：简单模型，基于峰值和基因的基本信息计算调控潜力
    # Enhanced：增强模型，考虑了基因的TSS（转录起始位点）以及外显子区域，同时移除启动子区域的影响

    if model == "Simple":
        if organism == "GRCh38":
            refgenes_file = "./Data/gene_annotation/GRCh38.refgenes.genescore.simple.csv"
        else:
            refgenes_file = "./Data/gene_annotation/GRCm38.refgenes.genescore.simple.csv"
        gene_bed_df = pd.read_csv(refgenes_file, index_col=False, header=0)
        genes_list = gene_bed_df.iloc[:, 3].tolist()
        gene_bed_df = gene_bed_df.drop(columns=gene_bed_df.columns[3])
        rp_score_mt = calculate_RP_score(peakmatrix=atac_df, gene_bed_df=gene_bed_df, genes_list=genes_list, decay=decaydistance, model=model)

    else:
        if organism == "GRCh38":
            refgenes_file = "./Data/gene_annotation/GRCh38.refgenes.genescore.adjusted.csv"
        else:
            refgenes_file = "./Data/gene_annotation/GRCm38.refgenes.genescore.adjusted.csv"
        gene_bed_df = pd.read_csv(refgenes_file, index_col=0, header=0)
        rp_score_mt = calculate_RP_score(peakmatrix=atac_df, gene_bed_df=gene_bed_df, genes_list=None, decay=decaydistance, model=model)

    return rp_score_mt


def calculate_RP_score(peakmatrix, gene_bed_df, genes_list, decay, model):
    """Calculate regulatery potential for each gene based on the single-cell peaks."""

    # genes_info = []
    # genes_list = []
    # peaks_info = []

    # 初始化和数据预处理
    cell_peaks = peakmatrix.T
    # peaks_list = features
    # cells_list = barcodes

    # cell_peaks = pd.read_csv(peak_file, sep="\t", header=0, index_col=0)
    cell_peaks[cell_peaks>1] = 1
    cells_list = list(cell_peaks.columns)
    peaks_list = [peak for peak in cell_peaks.index if peak.split("_")[1].isdigit()]
    cell_peaks = sp_sparse.csc_matrix(cell_peaks.loc[peaks_list, :].values)

    if model == "Simple":
        peaks_info = []
        genes_info = gene_bed_df.values.tolist()
        genes = list(set([i.split("@")[0] for i in genes_list]))
        for ipeak, peak in enumerate(peaks_list):
            peaks_tmp = peak.rsplit("_", maxsplit=2)
            peaks_info.append([peaks_tmp[0], (int(peaks_tmp[1]) + int(peaks_tmp[2])) / 2.0, 0, ipeak])

        genes_peaks_score_dok = RP_Simple(peaks_info, genes_info, decay)
    else:
        genes_list = []
        genes_info = ExtractGeneInfo(gene_bed_df)
        genes_info_tss = list()
        genes_info_full = list()  ### [chrom, tss, start, end, 1, unique_id]

        for igene in range(len(genes_info)):
            tmp_gene = genes_info[igene]
            genes_list.append(tmp_gene[-1])
            genes_info_full.append(tmp_gene + [igene])
            genes_info_tss.append([tmp_gene[0], tmp_gene[3], tmp_gene[1], tmp_gene[2]] + tmp_gene[4:] + [igene])
            ### add index at the end of gene symbol
        genes = list(set([i.split("@")[0] for i in genes_list]))

        # print("genes_info_full", genes_info_full[:2])
        # print("genes_info_tss", genes_info_tss[:2])

        peaks_info = []

        for ipeak, peak in enumerate(peaks_list):
            peaks_tmp = peak.rsplit("_", maxsplit=2)
            peaks_info.append([peaks_tmp[0], (int(peaks_tmp[1]) + int(peaks_tmp[2])) / 2.0, int(peaks_tmp[1]), int(peaks_tmp[2]), 0, peak, ipeak])
            # peaks_info [chrom, center, start, end, 0, uid, ipeak]
        # print("peaks_info", peaks_info[:2])
        if model == "Enhanced":
            genes_peaks_score_dok = RP_AddExonRemovePromoter(peaks_info, genes_info_full, genes_info_tss, decay)

    # 基因-细胞分数矩阵计算
    genes_peaks_score_csr = genes_peaks_score_dok.tocsr()
    genes_cells_score_csr = genes_peaks_score_csr.dot(cell_peaks.tocsr())

    # 去重和分数汇总
    score_cells_dict = {}
    score_cells_sum_dict = {}

    for igene, gene in enumerate(genes_list):
        score_cells_dict[gene] = igene
        score_cells_sum_dict[gene] = genes_cells_score_csr[igene, :].sum()

    score_cells_dict_dedup = {}
    score_cells_dict_max = {}
    for gene in genes:
        score_cells_dict_max[gene] = float("-inf")

    for gene in genes_list:
        symbol = gene.split("@")[0]
        if score_cells_sum_dict[gene] > score_cells_dict_max[symbol]:
            score_cells_dict_dedup[symbol] = score_cells_dict[gene]
            score_cells_dict_max[symbol] = score_cells_sum_dict[gene]

    # 生成最终矩阵
    gene_symbol = sorted(score_cells_dict_dedup.keys())
    matrix_row = []
    for gene in gene_symbol:
        matrix_row.append(score_cells_dict_dedup[gene])

    score_cells_matrix = genes_cells_score_csr[matrix_row, :]

    rp_score_df = pd.DataFrame(
        score_cells_matrix.T.toarray(),
        index=cells_list,
        columns=gene_symbol
    )

    return rp_score_df


def RP_Simple(peaks_info, genes_info, decay):
    """Multiple processing function to calculate regulation potential."""

    Sg = lambda x: 2 ** (-x)
    gene_distance = 15 * decay
    genes_peaks_score_array = sp_sparse.dok_matrix((len(genes_info), len(peaks_info)), dtype=np.float64)

    w = genes_info + peaks_info

    A = {}

    w.sort()
    for elem in w:
        if int(elem[2]) == 1:
            A[int(elem[-1])] = [elem[0], elem[1]]
        else:
            dlist = []
            for gene_name in list(A.keys()):
                g = A[gene_name]
                tmp_distance = elem[1] - g[1]
                if (g[0] != elem[0]) or (tmp_distance > gene_distance):
                    dlist.append(gene_name)
                else:
                    genes_peaks_score_array[gene_name, int(elem[-1])] = Sg(tmp_distance / decay)
            for gene_name in dlist:
                del A[gene_name]

    w.reverse()
    for elem in w:
        if int(elem[2]) == 1:
            A[int(elem[-1])] = [elem[0], elem[1]]
        else:
            dlist = []
            for gene_name in list(A.keys()):
                g = A[gene_name]
                tmp_distance = g[1] - elem[1]
                if (g[0] != elem[0]) or tmp_distance > gene_distance:
                    dlist.append(gene_name)
                else:
                    genes_peaks_score_array[gene_name, int(elem[-1])] = Sg(tmp_distance / decay)
            for gene_name in dlist:
                del A[gene_name]

    return (genes_peaks_score_array)

def RP_AddExonRemovePromoter(peaks_info, genes_info_full, genes_info_tss, decay):
    """Multiple processing function to calculate regulation potential."""

    Sg = lambda x: 2 ** (-x)
    checkInclude = lambda x, y: all([x >= y[0], x <= y[1]])
    gene_distance = 15 * decay
    genes_peaks_score_array = sp_sparse.dok_matrix((len(genes_info_full), len(peaks_info)), dtype=np.float64)
    peaks_info_inbody = []
    peaks_info_outbody = []

    w = genes_info_full + peaks_info
    A = {}

    w.sort()
    #     print(w[:100])
    for elem in w:
        if elem[-3] == 1:
            A[elem[-1]] = elem
        else:
            dlist = []
            for gene_name in list(A.keys()):
                g = A[gene_name]
                ### NOTE: main change here
                ### if peak center in the gene area
                if all([g[0] == elem[0], elem[1] >= g[1], elem[1] <= g[2]]):
                    ### if peak center in the exons
                    if any(list(map(checkInclude, [elem[1]] * len(g[5]), list(g[5])))):
                        genes_peaks_score_array[gene_name, elem[-1]] = 1.0 / g[-4]
                        peaks_info_inbody.append(elem)
                    ### if peak cencer in the promoter
                    elif checkInclude(elem[1], g[4]):
                        tmp_distance = abs(elem[1] - g[3])
                        genes_peaks_score_array[gene_name, elem[-1]] = Sg(tmp_distance / decay)
                        peaks_info_inbody.append(elem)
                    ### intron regions
                    else:
                        continue
                else:
                    dlist.append(gene_name)
            for gene_name in dlist:
                del A[gene_name]

    ### remove genes in promoters and exons
    peaks_info_set = [tuple(i) for i in peaks_info]
    peaks_info_inbody_set = [tuple(i) for i in peaks_info_inbody]
    peaks_info_outbody_set = list(set(peaks_info_set) - set(peaks_info_inbody_set))
    peaks_info_outbody = [list(i) for i in peaks_info_outbody_set]

    # print("peaks number: ", len(peaks_info_set))
    # print("peaks number in gene promoters and exons: ", len(set(peaks_info_inbody_set)))
    # print("peaks number out gene promoters and exons:", len(peaks_info_outbody_set))

    w = genes_info_tss + peaks_info_outbody
    A = {}

    w.sort()
    for elem in w:
        if elem[-3] == 1:
            A[elem[-1]] = elem
        else:
            dlist = []
            for gene_name in list(A.keys()):
                g = A[gene_name]
                tmp_distance = elem[1] - g[1]
                if all([g[0] == elem[0], tmp_distance <= gene_distance]):
                    genes_peaks_score_array[gene_name, elem[-1]] = Sg(tmp_distance / decay)
                else:
                    dlist.append(gene_name)
            for gene_name in dlist:
                del A[gene_name]

    w.reverse()
    for elem in w:
        if elem[-3] == 1:
            A[elem[-1]] = elem
        else:
            dlist = []
            for gene_name in list(A.keys()):
                g = A[gene_name]
                tmp_distance = g[1] - elem[1]
                if all([g[0] == elem[0], tmp_distance <= gene_distance]):
                    genes_peaks_score_array[gene_name, elem[-1]] = Sg(tmp_distance / decay)
                else:
                    dlist.append(gene_name)
            for gene_name in dlist:
                del A[gene_name]

    return (genes_peaks_score_array)

def ExtractGeneInfo(bed):
    """Extract gene information from gene bed data frame."""

    bed['transcript'] = [x.strip().split(".")[0] for x in bed['name'].tolist()]
    bed['tss'] = bed.apply(lambda x: x['txStart'] if x['strand'] == '+' else x['txEnd'], axis=1)

    ### adjacent P+GB
    bed["start"] = bed.apply(lambda x: x['txStart'] - 2000 if x['strand'] == '+' else x['txStart'], axis=1)
    bed["end"] = bed.apply(lambda x: x['txEnd'] + 2000 if x['strand'] == '-' else x['txEnd'], axis=1)

    bed['promoter'] = bed.apply(lambda x: tuple([x['tss'] - 2000, x['tss'] + 2000]), axis=1)
    bed['exons'] = bed.apply(lambda x: tuple(
        [(int(i), int(j)) for i, j in zip(x['exonStarts'].strip(',').split(','), x['exonEnds'].strip(',').split(','))]),
                             axis=1)

    ### exon length
    bed['length'] = bed.apply(lambda x: sum(list(map(lambda i: (i[1] - i[0]) / 1000.0, x['exons']))), axis=1)
    bed['uid'] = bed.apply(lambda x: "%s@%s@%s" % (x['name2'], x['start'], x['end']), axis=1)
    bed = bed.drop_duplicates(subset='uid', keep="first")
    gene_info = []
    for irow, x in bed.iterrows():
        gene_info.append(
            [x['chrom'], x['start'], x['end'], x['tss'], x['promoter'], x['exons'], x['length'], 1, x['uid']])
    ### [chrom_0, start_1, end_2, tss_3, promoter_4, exons_5, length_6, 1_7, uid_8]
    return (gene_info)


def set_seed(seed):
    """set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from typing import Dict, Optional, Union

class Preprocessor:
    """
    Prepare data into training, valid and test split. Normalize raw expression
    values, binning or using other transform into the preset model input format.
    """

    def __init__(
        self,
        use_key: Optional[str] = None,
        filter_gene_by_counts: Union[int, bool] = False,
        filter_cell_by_counts: Union[int, bool] = False,
        normalize_total: Union[float, bool] = 1e4,
        result_normed_key: Optional[str] = "X_normed",
        log1p: bool = False,
        result_log1p_key: str = "X_log1p",
        subset_hvg: Union[int, bool] = False,
        hvg_use_key: Optional[str] = None,
        hvg_flavor: str = "seurat_v3"
    ):
        r"""
        Set up the preprocessor, use the args to config the workflow steps.

        Args:

        use_key (:class:`str`, optional):
            The key of :class:`~anndata.AnnData` to use for preprocessing.
        filter_gene_by_counts (:class:`int` or :class:`bool`, default: ``False``):
            Whther to filter genes by counts, if :class:`int`, filter genes with counts
        filter_cell_by_counts (:class:`int` or :class:`bool`, default: ``False``):
            Whther to filter cells by counts, if :class:`int`, filter cells with counts
        normalize_total (:class:`float` or :class:`bool`, default: ``1e4``):
            Whether to normalize the total counts of each cell to a specific value.
        result_normed_key (:class:`str`, default: ``"X_normed"``):
            The key of :class:`~anndata.AnnData` to store the normalized data. If
            :class:`None`, will use normed data to replce the :attr:`use_key`.
        log1p (:class:`bool`, default: ``True``):
            Whether to apply log1p transform to the normalized data.
        result_log1p_key (:class:`str`, default: ``"X_log1p"``):
            The key of :class:`~anndata.AnnData` to store the log1p transformed data.
        subset_hvg (:class:`int` or :class:`bool`, default: ``False``):
            Whether to subset highly variable genes.
        hvg_use_key (:class:`str`, optional):
            The key of :class:`~anndata.AnnData` to use for calculating highly variable
            genes. If :class:`None`, will use :attr:`adata.X`.
        hvg_flavor (:class:`str`, default: ``"seurat_v3"``):
            The flavor of highly variable genes selection. See
            :func:`scanpy.pp.highly_variable_genes` for more details.
        binning (:class:`int`, optional):
            Whether to bin the data into discrete values of number of bins provided.
        result_binned_key (:class:`str`, default: ``"X_binned"``):
            The key of :class:`~anndata.AnnData` to store the binned data.
        """
        self.use_key = use_key
        self.filter_gene_by_counts = filter_gene_by_counts
        self.filter_cell_by_counts = filter_cell_by_counts
        self.normalize_total = normalize_total
        self.result_normed_key = result_normed_key
        self.log1p = log1p
        self.result_log1p_key = result_log1p_key
        self.subset_hvg = subset_hvg
        self.hvg_use_key = hvg_use_key
        self.hvg_flavor = hvg_flavor


    def __call__(self, adata: AnnData, batch_key: Optional[str] = None) -> Dict:
        """
        format controls the different input value wrapping, including categorical
        binned style, fixed-sum normalized counts, log1p fixed-sum normalized counts, etc.

        Args:

        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        batch_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information. This arg
            is used in the highly variable gene selection step.
        """
        key_to_process = self.use_key
        # preliminary checks, will use later
        if key_to_process == "X":
            key_to_process = None  # the following scanpy apis use arg None to use X
        is_logged = self.check_logged(adata, obs_key=key_to_process)

        # step 1: filter genes
        if self.filter_gene_by_counts:
            print("Filtering genes by counts ...")
            sc.pp.filter_genes(
                adata,
                min_counts=self.filter_gene_by_counts
                if isinstance(self.filter_gene_by_counts, int)
                else None,
            )

        # step 2: filter cells
        if (
            isinstance(self.filter_cell_by_counts, int)
            and self.filter_cell_by_counts > 0
        ):
            print("Filtering cells by counts ...")
            sc.pp.filter_cells(
                adata,
                min_counts=self.filter_cell_by_counts
                if isinstance(self.filter_cell_by_counts, int)
                else None,
            )

        # step 3: normalize total
        if self.normalize_total:
            print("Normalizing total counts ...")
            normed_ = sc.pp.normalize_total(
                adata,
                target_sum=self.normalize_total
                if isinstance(self.normalize_total, float)
                else None,
                layer=key_to_process,
                inplace=False,
            )["X"]
            key_to_process = self.result_normed_key or key_to_process
            _set_obs_rep(adata, normed_, layer=key_to_process)

        # step 4: log1p
        if self.log1p:
            print("Log1p transforming ...")
            if is_logged:
                print(
                    "The input data seems to be already log1p transformed. "
                    "Set `log1p=False` to avoid double log1p transform."
                )
            if self.result_log1p_key:
                _set_obs_rep(
                    adata,
                    _get_obs_rep(adata, layer=key_to_process),
                    layer=self.result_log1p_key,
                )
                key_to_process = self.result_log1p_key
            sc.pp.log1p(adata, layer=key_to_process)

        # step 5: subset hvg
        if self.subset_hvg:
            print("Subsetting highly variable genes ...")
            if batch_key is None:
                print(
                    "No batch_key is provided, will use all cells for HVG selection."
                )
            sc.pp.highly_variable_genes(
                adata,
                layer=self.hvg_use_key,
                n_top_genes=self.subset_hvg if isinstance(self.subset_hvg, int) else None,
                batch_key=batch_key,
                flavor=self.hvg_flavor,
                subset=False,
            )

        return adata

    def check_logged(self, adata: AnnData, obs_key: Optional[str] = None) -> bool:
        """
        Check if the data is already log1p transformed.

        Args:

        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        obs_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information. This arg
            is used in the highly variable gene selection step.
        """
        data = _get_obs_rep(adata, layer=obs_key)
        max_, min_ = data.max(), data.min()
        if max_ > 30:
            return False
        if min_ < 0:
            return False

        non_zero_min = data[data > 0].min()
        if non_zero_min >= 1:
            return False

        return True

def _digitize(x: np.ndarray, bins: np.ndarray, side="both") -> np.ndarray:
    """
    Digitize the data into bins. This method spreads data uniformly when bins
    have same values.

    Args:

    x (:class:`np.ndarray`):
        The data to digitize.
    bins (:class:`np.ndarray`):
        The bins to use for digitization, in increasing order.
    side (:class:`str`, optional):
        The side to use for digitization. If "one", the left side is used. If
        "both", the left and right side are used. Default to "one".

    Returns:

    :class:`np.ndarray`:
        The digitized data.
    """
    assert x.ndim == 1 and bins.ndim == 1

    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits

    right_difits = np.digitize(x, bins, right=True)

    rands = np.random.rand(len(x))  # uniform random numbers

    digits = rands * (right_difits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits


def binning(
    row: Union[np.ndarray, torch.Tensor], n_bins: int
) -> Union[np.ndarray, torch.Tensor]:
    """Binning the row into n_bins."""
    dtype = row.dtype
    return_np = False if isinstance(row, torch.Tensor) else True
    row = row.cpu().numpy() if isinstance(row, torch.Tensor) else row
    # TODO: use torch.quantile and torch.bucketize

    if row.max() == 0:
        print(
            "The input data contains row of zeros. Please make sure this is expected."
        )
        return (
            np.zeros_like(row, dtype=dtype)
            if return_np
            else torch.zeros_like(row, dtype=dtype)
        )

    if row.min() <= 0:
        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
        non_zero_digits = _digitize(non_zero_row, bins)
        binned_row = np.zeros_like(row, dtype=np.int64)
        binned_row[non_zero_ids] = non_zero_digits
    else:
        bins = np.quantile(row, np.linspace(0, 1, n_bins - 1))
        binned_row = _digitize(row, bins)
    return torch.from_numpy(binned_row) if not return_np else binned_row.astype(dtype)


class Binning:

    def __init__(
        self,
        binning: Optional[int] = None,
        result_binned_key: str = "X_binned",
    ):
        self.binning = binning
        self.result_binned_key = result_binned_key
        self.use_key = 'X'

    def __call__(self, adata_a, adata_b):

        adata_a = self.run_binning(adata_a)
        adata_b = self.run_binning(adata_b)

        return adata_a, adata_b


    def run_binning(self, adata):

        key_to_process = self.use_key
        # preliminary checks, will use later
        if key_to_process == "X":
            key_to_process = None

        if self.binning:
            print("Binning data ...")
            if not isinstance(self.binning, int):
                raise ValueError(
                    "Binning arg must be an integer, but got {}.".format(self.binning)
                )
            n_bins = self.binning  # NOTE: the first bin is always a spectial for zero
            binned_rows = []
            bin_edges = []
            layer_data = _get_obs_rep(adata, layer=key_to_process)
            layer_data = layer_data.toarray() if issparse(layer_data) else layer_data
            if layer_data.min() < 0:
                raise ValueError(
                    f"Assuming non-negative data, but got min value {layer_data.min()}."
                )
            for row in layer_data:
                if row.max() == 0:
                    print(
                        "The input data contains all zero rows. Please make sure "
                        "this is expected. You can use the `filter_cell_by_counts` "
                        "arg to filter out all zero rows."
                    )
                    binned_rows.append(np.zeros_like(row, dtype=np.int64))
                    bin_edges.append(np.array([0] * n_bins))
                    continue
                non_zero_ids = row.nonzero()
                non_zero_row = row[non_zero_ids]
                bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
                # bins = np.sort(np.unique(bins))
                # NOTE: comment this line for now, since this will make the each category
                # has different relative meaning across datasets
                non_zero_digits = _digitize(non_zero_row, bins)
                assert non_zero_digits.min() >= 1
                assert non_zero_digits.max() <= n_bins - 1
                binned_row = np.zeros_like(row, dtype=np.int64)
                binned_row[non_zero_ids] = non_zero_digits
                binned_rows.append(binned_row)
                bin_edges.append(np.concatenate([[0], bins]))
            adata.layers[self.result_binned_key] = np.stack(binned_rows)
            adata.obsm["bin_edges"] = np.stack(bin_edges)

        return adata