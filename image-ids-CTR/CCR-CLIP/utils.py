import torch
import time
import shutil
import os

from config import config
from dataset import lmdbDataset, resizeNormalize
from shutil import copyfile

alp2num = {}
alphabet_character = []
alphabet_character.append('PAD')
lines = open(config['alphabet_path'], 'r').readlines()
for line in lines:
    alphabet_character.append(line.strip('\n'))
alphabet_character.append('$')
for index, char in enumerate(alphabet_character):
    alp2num[char] = index

dict_file = open(config['decompose_path'], 'r').readlines()
char_radical_dict = {}
for line in dict_file:
    line = line.strip('\n')
    try:
        char, r_s = line.split(':')
    except:
        char, r_s = ':', ':'
    if 'rsst' not in config['decompose_path']:
        char_radical_dict[char] = r_s.split(' ')
    else:
        char_radical_dict[char] = list(''.join(r_s.split(' ')))

def get_data_package():
    train_dataset = []
    for dataset_root in config['train_dataset'].split(','):
        dataset = lmdbDataset(dataset_root, resizeNormalize((config['imageW'], config['imageH']), test=False))
        train_dataset.append(dataset)
    train_dataset_total = torch.utils.data.ConcatDataset(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_total, batch_size=config['batch'], shuffle=True, num_workers=8,
    )

    test_dataset = []
    for dataset_root in config['test_dataset'].split(','):
        dataset = lmdbDataset(dataset_root, resizeNormalize((config['imageW'], config['imageH']), test=True))
        test_dataset.append(dataset)
    test_dataset_total = torch.utils.data.ConcatDataset(test_dataset)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset_total, batch_size=config['batch'], shuffle=False, num_workers=8,
    )

    return train_dataloader, test_dataloader

import copy
def convert(label):
    r_label = []
    batch = len(label)
    for i in range(batch):
        r_tmp = copy.deepcopy(char_radical_dict[label[i]])
        r_tmp.append('$')
        r_label.append(r_tmp)

    text_tensor = torch.zeros(batch, config['max_len']).long().cuda()
    for i in range(batch):
        tmp = r_label[i]
        for j in range(len(tmp)):
            text_tensor[i][j] = alp2num[tmp[j]]
    return text_tensor

def saver():
    try:
        shutil.rmtree('./history/{}'.format(config['exp_name']))
    except:
        pass
    os.mkdir('./history/{}'.format(config['exp_name']))
    print('**** Experiment Name: {} ****'.format(config['exp_name']))

    localtime = time.asctime(time.localtime(time.time()))
    f = open(os.path.join('./history', config['exp_name'], str(localtime)),'w+')
    f.close()

    src_folder = './'
    exp_name = config['exp_name']
    dst_folder = os.path.join('./history', exp_name)

    file_list = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    for file_name in file_list:
        src = os.path.join(src_folder, file_name)
        dst = os.path.join(dst_folder, file_name)
        copyfile(src, dst)


def get_alphabet():
    return alphabet_character
