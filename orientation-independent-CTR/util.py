import torch
import os
import shutil
import torch.nn as nn

from shutil import copyfile
from data.lmdbReader import lmdbDataset, alignCollate
from config import config

mse_loss = nn.MSELoss()

alphabet_character_file = open(config['alpha_path'])
alphabet_character = list(alphabet_character_file.read().strip())
alphabet_character_raw = ['START', '\xad']

for item in alphabet_character:
    alphabet_character_raw.append(item)

alphabet_character_raw.append('END')
alphabet_character = alphabet_character_raw

alp2num_character = {}

for index, char in enumerate(alphabet_character):
    alp2num_character[char] = index

def get_data_package():
    train_dataset = []
    for dataset_root in config['train_dataset'].split(','):
        dataset = lmdbDataset(dataset_root)
        train_dataset.append(dataset)
    train_dataset_total = torch.utils.data.ConcatDataset(train_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_total, batch_size=config['batch'], shuffle=True, num_workers=8,
        collate_fn = alignCollate(imgH=config['imageH'], imgW=config['imageW'])
    )

    test_dataset = []
    for dataset_root in config['test_dataset'].split(','):
        dataset = lmdbDataset(dataset_root)
        test_dataset.append(dataset)
    test_dataset_total = torch.utils.data.ConcatDataset(test_dataset)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset_total, batch_size=config['batch'], shuffle=False, num_workers=8,
        collate_fn=alignCollate(imgH=config['imageH'], imgW=config['imageW'])
    )

    return train_dataloader, test_dataloader

def converter(label):

    string_label = label
    label = [i for i in label]
    alp2num = alp2num_character

    batch = len(label)
    length = torch.Tensor([len(i) for i in label]).long().cuda()
    max_length = max(length)

    text_input = torch.zeros(batch, max_length).long().cuda()
    for i in range(batch):
        for j in range(len(label[i]) - 1):
            try:
                text_input[i][j + 1] = alp2num[label[i][j]]
            except:
                text_input[i][j + 1] = alp2num['-']

    sum_length = sum(length)
    text_all = torch.zeros(sum_length).long().cuda()
    start = 0
    for i in range(batch):
        for j in range(len(label[i])):
            if j == (len(label[i])-1):
                text_all[start + j] = alp2num['END']
            else:
                try:
                    text_all[start + j] = alp2num[label[i][j]]
                except:
                    text_all[start + j] = alp2num['-']
        start += len(label[i])

    return length, text_input, text_all, string_label


def get_alphabet():
    return alphabet_character

def get_sample():
    char2img_dict, char2rotimg_dict = {}, {}
    test_dataset = []
    for dataset_root in '/home/yuhaiyang/dataset/SIMSUN_Benchmark_Char'.split(','):
        dataset = lmdbDataset(dataset_root)
        test_dataset.append(dataset)
    test_dataset_total = torch.utils.data.ConcatDataset(test_dataset)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset_total, batch_size=1, shuffle=False, num_workers=8,
        collate_fn=alignCollate(imgH=32, imgW=32)
    )
    test_dataloader = iter(test_dataloader)
    for iteration in range(len(test_dataloader)):
        data = test_dataloader.next()
        image, _, label, _ = data
        image = torch.nn.functional.interpolate(image, size=(32, 32))
        char2img_dict[label[0][0]] = image
        char2rotimg_dict[label[0][0]] = torch.rot90(image, 1, [2, 3])
    return char2img_dict, char2rotimg_dict

def tensor2str(tensor):
    alphabet = get_alphabet()
    string = ""
    for i in tensor:
        if i == (len(alphabet)-1):
            continue
        string += alphabet[i]
    return string

def must_in_screen():
    text = os.popen('echo $STY').readlines()
    string = ''
    for line in text:
        string += line
    if len(string.strip()) == 0:
        print("run the program in the screen!")
        exit(0)

def saver():
    try:
        shutil.rmtree('./history/{}'.format(config['exp_name']))
    except:
        pass
    os.mkdir('./history/{}'.format(config['exp_name']))

    import time
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


if __name__ == '__main__':
    length, text_input, text_all = converter(['我$', '你$','他$'])
    print(length)
    print(text_input)
    print(text_all)