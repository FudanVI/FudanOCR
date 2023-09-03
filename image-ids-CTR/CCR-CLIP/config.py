import os
root = 'Font_Datasets'
datasets = os.listdir(root)
train_data = ','.join([os.path.join(root, item) for item in datasets])
config = {
    'epoch': 100000,
    'train_dataset': train_data,
    'test_dataset': '',
    'batch': 128,
    'imageW': 128,
    'imageH': 128,
    'alphabet_path': './data/radical_alphabet_27533_benchmark.txt',
    'decompose_path': './data/decompose_27533_benchmark.txt',
    'max_len': 30,
    'lr': 1e-4,
    'exp_name': '',
}