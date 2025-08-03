import subprocess

command = [
    'python', 'main.py',
    '--n_epochs', '10',
    '--train_batch_size', '32',
    '--test_batch_size', '64',
    '--dateset', '10x_Multiome_Pbmc10k',
    '--lr', '0.0001',
    '--folds', '5',
    '--modal_a_file', '10x-Multiome-Pbmc10k-RNA.h5ad',
    '--modal_b_file', '10x-Multiome-Pbmc10k-ATAC.h5ad'
]

result = subprocess.run(command)