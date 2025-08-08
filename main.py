import sys
import torch
import argparse
from tqdm import tqdm
from torch import optim
from datasets import load_data, SCDataset, preprocess_data, prepare_data, prepare_dataloader
from sklearn.model_selection import KFold
from utils import set_seed
from torchtext.vocab import Vocab
from torchtext._torchtext import (Vocab as VocabPybind,)
import numpy as np
from tokenizer import tokenize_and_pad_batch
import time
from model import Gene_Transformer
import torch.nn as nn

def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=10, help='')
    parser.add_argument('--train_batch_size', type=int, default=32, help='')
    parser.add_argument('--test_batch_size', type=int, default=64, help='')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='')
    parser.add_argument('--dateset', type=str, help='')
    parser.add_argument('--modal_a', type=str, default='scRNA-seq', help='')
    parser.add_argument('--modal_b', type=str, default='scATAC-seq', help='')
    parser.add_argument('--modal_a_file', type=str, help='')
    parser.add_argument('--modal_b_file', type=str, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='')
    parser.add_argument('--folds', type=int, default=5, help='')
    parser.add_argument('--d_model', type=int, default=256, help='')
    parser.add_argument('--num_heads', type=int, default=2, help='')
    parser.add_argument('--d_embd', type=int, default=128, help='')
    parser.add_argument('--d_ff', type=int, default=128, help='')
    parser.add_argument("--n_bins", type=int, default=5, help='')
    parser.add_argument("--seed", type=int, default=2025, help='')
    parser.add_argument('--max_seq_len', type=int, default=2048, help='')
    parser.add_argument('--include_zero_gene', type=bool, default=True, help='')
    parser.add_argument('--dropout', type=float, default=0.2, help='')


    args = parser.parse_args()

    return args

def run():
    args = prepare()

    set_seed(args.seed)

    modal_a_adata, modal_b_adata = load_data(args)

    adata_a, adata_b, num_types = preprocess_data(modal_a_adata, modal_b_adata, args)

    count_a = adata_a.layers["X_binned"]
    count_b = adata_b.layers["X_binned"]

    genes = adata_a.var_names.tolist()
    celltypes_a = adata_a.obs["celltype_id"].to_numpy()
    celltypes_b = adata_b.obs["celltype_id"].to_numpy()

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=25)

    splits = []
    for train_idx, test_idx in kf.split(count_a):
        count_a_train = count_a[train_idx]
        count_a_test = count_a[test_idx]

        count_b_train = count_b[train_idx]
        count_b_test = count_b[test_idx]

        celltypes_a_train = celltypes_a[train_idx]
        celltypes_a_test = celltypes_a[test_idx]

        celltypes_b_train = celltypes_b[train_idx]
        celltypes_b_test = celltypes_b[test_idx]

        splits.append({
            'count_a_train': count_a_train,
            'count_a_test': count_a_test,
            'count_b_train': count_b_train,
            'count_b_test': count_b_test,
            'celltypes_a_train': celltypes_a_train,
            'celltypes_a_test': celltypes_a_test,
            'celltypes_b_train': celltypes_b_train,
            'celltypes_b_test': celltypes_b_test
        })

    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>"]
    vocab = Vocab(VocabPybind(genes + special_tokens, None))
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)
    pad_value = args.n_bins
    n_input_bins = args.n_bins + 1

    for i, split in enumerate(splits):
        print(f"Fold {i + 1}:")
        print(f"  Train cells: {len(split['count_a_train'])}")
        print(f"  Test cells: {len(split['count_a_test'])}")

        train_data_a = split['count_a_train']
        train_data_b = split['count_b_train']

        train_data = np.vstack([train_data_a, train_data_b])

        train_celltypes_a = split['celltypes_a_train']
        train_celltypes_b = split['celltypes_b_train']

        train_celltypes = np.vstack([train_celltypes_a, train_celltypes_b])

        tokenized_train = tokenize_and_pad_batch(
            train_data,
            gene_ids,
            max_len=args.max_seq_len,
            vocab=vocab,
            pad_token=pad_token,
            pad_value=pad_value,
            append_cls=True,  # append <cls> token at the beginning
            include_zero_gene=args.include_zero_gene,
        )

        test_data_a = split['count_a_test']
        test_data_b = split['count_b_test']

        test_data = np.vstack([test_data_a, test_data_b])

        test_celltypes_a = split['celltypes_a_test']
        test_celltypes_b = split['celltypes_b_test']

        test_celltypes = np.vstack([test_celltypes_a, test_celltypes_b])

        tokenized_test = tokenize_and_pad_batch(
            test_data,
            gene_ids,
            max_len=args.max_seq_len,
            vocab=vocab,
            pad_token=pad_token,
            pad_value=pad_value,
            append_cls=True,  # append <cls> token at the beginning
            include_zero_gene=args.include_zero_gene,
        )

        model = Gene_Transformer(
            args=args,
            ntoken=len(vocab),
            d_model=args.d_embd,
            nhead=args.num_heads,
            d_hid=args.d_ff,
            vocab=vocab,
            dropout=args.dropout,
            pad_token=pad_token,
            pad_value=pad_value,
            n_input_bins=n_input_bins,
            nlayers=2,
            n_cls=num_types
        )

        model.to(args.device)
        criterion_cls = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-4)


        for epoch in range(1, args.n_epochs + 1):
            epoch_start_time = time.time()
            train_data_pt, valid_data_pt = prepare_data(tokenized_train=tokenized_train,
                                                        tokenized_test=tokenized_test,
                                                        train_celltype_labels=train_celltypes,
                                                        test_celltype_labels=test_celltypes
                                                        )

            train_loader = prepare_dataloader(
                train_data_pt,
                batch_size=args.train_batch_size,
                shuffle=False,
                intra_domain_shuffle=True,
                drop_last=False,
            )
            test_loader = prepare_dataloader(
                valid_data_pt,
                batch_size=args.test_batch_size,
                shuffle=False,
                intra_domain_shuffle=False,
                drop_last=False,
            )



            sys.exit()




if __name__ == '__main__':
    run()