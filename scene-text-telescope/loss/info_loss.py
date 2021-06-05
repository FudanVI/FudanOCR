# 数据不对齐问题出现在各个超分环节中，严格对齐对于文本来说是没有什么意义的，
# 在这种情况下就要选择更为精密的loss，而不是简单的L1，L2对齐Loss
import torch
import torch.nn as nn
import cv2
import numpy as np
# from model.crnn import CRNN
import sys
# import fasttext
import string
import random
import torch.optim as optim
import time

# embedding_bin = "/home/chenjingye/cc.en.300.bin"
# emb_encoder = fasttext.load_model(embedding_bin)
# print('Success loading!')
# from loss.GAN.discriminator import discriminator

ce_loss = torch.nn.CrossEntropyLoss()

# from loss.GAN.pool_data import tensorDataset
from loss.weight_ce_loss import weight_cross_entropy
# from loss.transformer import Transformer

# 计算flops
from thop import profile

mode = 'plain'  # character / stroke / plain
if mode == 'stroke':
    from Transformer_pretrain_split_english.model import Transformer
    transformer = Transformer().cuda()
    transformer = nn.DataParallel(transformer)
    transformer.load_state_dict(torch.load('/home/chenjingye/new_textzoom/TextZoom/src/Transformer_pretrain/checkpoint_拆英文/56000.pth'))
    transformer.eval()
    lines = open('/home/chenjingye/TextZoom/src/Transformer_pretrain_split_english/english_digit_split.txt','r').readlines()
    dic = {}
    for line in lines:
        line = line.strip()
        character, sequence = line.split()
        dic[character] = sequence
    english_alphabet = '0123456789'
    english_dict = {}
    for index in range(len(english_alphabet)):
        english_dict[english_alphabet[index]] = index
elif mode == 'character':
    from loss.transformer import Transformer

    transformer = Transformer().cuda()
    transformer = nn.DataParallel(transformer)
    transformer.load_state_dict(torch.load('/home/chenjingye/TextZoom/src/Transformer_pretrain/lower_74000_0.73.pth'))
    transformer.eval()

    english_alphabet = '-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    english_dict = {}
    for index in range(len(english_alphabet)):
        english_dict[english_alphabet[index]] = index
# english_alphabet = '-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


def to_gray_tensor(tensor):
    R = tensor[:, 0:1, :, :]
    G = tensor[:, 1:2, :, :]
    B = tensor[:, 2:3, :, :]
    tensor = 0.299 * R + 0.587 * G + 0.114 * B
    return tensor


def get_word_embedding(word_batch):
    # 去除最后一个占位符
    # word_batch = [i[:-1] for i in word_batch]
    lis = []
    for word in word_batch:
         lis.append(torch.from_numpy(emb_encoder.get_word_vector(word)).view(1,300,1,1))

    word_tensor = torch.cat([i for i in lis],0)
    # word_tensor = word_tensor.repeat(1,1,16,64).cuda()
    return word_tensor.cuda()


def str_filt(str_, voc_type):
    alpha_dict = {
        'digit': string.digits,
        'lower': string.digits + string.ascii_lowercase,
        'upper': string.digits + string.ascii_letters,
        'all': string.digits + string.ascii_letters + string.punctuation
    }
    if voc_type == 'lower':
        str_ = str_.lower()
    for char in str_:
        if char not in alpha_dict[voc_type]:
            str_ = str_.replace(char, '')
    str_ = str_.lower()
    return str_

def negative_sample_rebatch(word_decoder_result, label):
    negative_pool = []
    for label_index, one_label in enumerate(label):
        for i in range(len(one_label)-1):
            negative_pool.append((word_decoder_result[label_index][i],english_dict[label[label_index][i]]))

    # print(negative_pool)
    negative_data = torch.cat([i[0].view(1,-1) for i in negative_pool],0).cuda()
    negative_gt = torch.Tensor([i[1] for i in negative_pool]).long().cuda()

    return negative_data, negative_gt

def get_positive_sample(dataset, number):
    length = len(dataset)
    lis = list(range(length))
    sample_list = random.sample(lis, number)

    data, gt = dataset.list_sample(sample_list)
    # print(type(data))
    # print(type(gt))
    return data, gt

def discriminator_process(vector, gt):
    length = int(vector.shape[0]/2)

    d_optimizer.zero_grad()
    pred = d(vector)
    loss = ce_loss(pred, gt)
    loss.backward(retain_graph=True)
    d_optimizer.step()
    return loss

memory = {}


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()

        # L1损失与L2损失也要做实验的

        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()

    def delete_useless(self, map_tensor, length_tensor):
        batch, head_num, max_length, map_length = map_tensor.shape
        useful_map_tensor = torch.zeros(length_tensor.sum() - length_tensor.shape[0], head_num, map_length).cuda()
        start = 0
        for batch_index, one_length in enumerate(length_tensor):
            useful_map_tensor[start: start+one_length-1] = map_tensor[batch_index, :,:one_length-1,:].permute(1,0,2).contiguous()
            start += one_length-1
        return useful_map_tensor

    def forward(self,sr_img, hr_img, label):

        #######################拆解英文的配套设施############################
        # label保留大小写，剩余中文英文，然后变成序列
        # label = [str_filt(i, 'lower')+'-' for i in label]
        if mode == 'stroke':
            original_label = label
            new_label_list = []
            for one_label in label:
                stroke_sequence = ''
                for character in one_label:
                    if character not in dic:
                        continue
                    else:
                        stroke_sequence += dic[character]
                stroke_sequence += '0'
                new_label_list.append(stroke_sequence)
            label = new_label_list
            length_tensor, input_tensor, text_gt = label_encoder(label)
            # print('label的长度为',len(label))
            # print('sr图片的尺寸为',sr_img.shape)
        elif mode == 'character':
            label = [str_filt(i, 'lower')+'-' for i in label]
            length_tensor, input_tensor, text_gt = label_encoder(label)
        # print('label为',label)
        ####################################################################
        # print('label', label)
        # 图像L2损失
        mse_loss = self.mse_loss(sr_img, hr_img)
        if mode == 'plain':
            attention_loss = 0
            recognition_loss = 0
        else:
            hr_pred, word_attention_map_gt, hr_correct_list = transformer(to_gray_tensor(hr_img), length_tensor, input_tensor, test=False)
            sr_pred, word_attention_map_pred, sr_correct_list = transformer(to_gray_tensor(sr_img), length_tensor, input_tensor, test=False)
            attention_loss = self.l1_loss(word_attention_map_gt, word_attention_map_pred)

            # flops, params = profile(transformer.cuda(), inputs=(to_gray_tensor(sr_img).cuda(), length_tensor.cuda(), input_tensor.cuda(),))
            # print('FLOPS : {} | PARAMS : {}'.format(flops, params))
            # exit(0)
            # attention_loss = 0

        # recognition_loss = 0
        # recognition_loss = weight_cross_entropy(sr_pred, text_gt)
            recognition_loss = self.l1_loss(hr_pred, sr_pred)
        # recognition_loss = self.ce_loss(sr_pred, text_gt)

        if mode == 'character':
            # loss = mse_loss
            # loss = mse_loss + recognition_loss * 0.1
            loss = mse_loss + attention_loss * 10 + recognition_loss * 0.0005
            return loss, mse_loss, attention_loss, recognition_loss
        elif mode == 'stroke':
            loss = mse_loss + attention_loss * 50 + recognition_loss * 0.0005
            return loss, mse_loss, attention_loss, recognition_loss
        elif mode == 'plain':
            loss = mse_loss
            return loss, mse_loss, attention_loss, recognition_loss

def label_encoder(label):
    batch = len(label)

    length = [len(i) for i in label]
    length_tensor = torch.Tensor(length).long().cuda()

    max_length = max(length)
    input_tensor = np.zeros((batch, max_length))
    for i in range(batch):
        for j in range(length[i]-1):
            input_tensor[i][j+1] = english_dict[label[i][j]]

    text_gt = []
    for i in label:
        for j in i:
            text_gt.append(english_dict[j])
    text_gt = torch.Tensor(text_gt).long().cuda()

    input_tensor = torch.from_numpy(input_tensor).long().cuda()
    return length_tensor, input_tensor, text_gt

