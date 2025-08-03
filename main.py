import sys

import torch
import argparse
from tqdm import tqdm
from torch import optim
from datasets import load_data, SCDataset, preprocess_data
from sklearn.model_selection import KFold
from utils import set_seed

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
    parser.add_argument('--d_ff', type=int, default=1024, help='')
    parser.add_argument('--vocab_size', type=int, default=18628, help='')
    parser.add_argument("--n_bins", type=int, default=5, help='')
    parser.add_argument("--seed", type=int, default=2025, help='')
    parser.add_argument('--max_seq_len', type=int, default=2048, help='')

    args = parser.parse_args()

    return args

def run():
    args = prepare()

    set_seed(args.seed)

    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>"]

    modal_a_adata, modal_b_adata = load_data(args)

    mt_a, l_a, mt_b, l_b, rp_b = preprocess_data(modal_a_adata, modal_b_adata, args)




    sys.exit()

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=25)
    train_dataset_list = []
    test_dataset_list = []

    for train_indices, val_indices in kf.split(modal_a_matrix):
        train_modal_a_matrix = modal_a_matrix.iloc[train_indices]
        train_modal_a_label = modal_a_label.iloc[train_indices]
        train_modal_b_matrix = modal_b_matrix.iloc[train_indices]
        train_modal_b_label = modal_b_label.iloc[train_indices]

        test_modal_a_matrix = modal_a_matrix.iloc[val_indices]
        test_modal_a_label = modal_a_label.iloc[val_indices]
        test_modal_b_matrix = modal_b_matrix.iloc[val_indices]
        test_modal_b_label = modal_b_label.iloc[val_indices]

        train_dataset = SCDataset(
            train_modal_a_matrix, train_modal_a_label,
            train_modal_b_matrix, train_modal_b_label,
            device=args.device
        )
        test_dataset = SCDataset(
            test_modal_a_matrix, test_modal_a_label,
            test_modal_b_matrix, test_modal_b_label,
            device=args.device
        )

        train_dataset_list.append(train_dataset)
        test_dataset_list.append(test_dataset)


    for fold, (train_dataset, test_dataset) in enumerate(zip(train_dataset_list, test_dataset_list)):
        print(f"Fold {fold + 1}/{args.folds}")



if __name__ == '__main__':
    run()