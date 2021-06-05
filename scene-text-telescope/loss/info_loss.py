import torch
import torch.nn as nn
import cv2
import numpy as np
import sys
import string
import random
import torch.optim as optim
import time

ce_loss = torch.nn.CrossEntropyLoss()
from loss.weight_ce_loss import weight_cross_entropy


mode = 'character'  # character / plain
if mode == 'character':
    from loss.transformer import Transformer

    transformer = Transformer().cuda()
    transformer = nn.DataParallel(transformer)
    transformer.load_state_dict(torch.load('./dataset/mydata/pretrain_transformer.pth'))
    transformer.eval()

    english_alphabet = '-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    english_dict = {}
    for index in range(len(english_alphabet)):
        english_dict[english_alphabet[index]] = index


def to_gray_tensor(tensor):
    R = tensor[:, 0:1, :, :]
    G = tensor[:, 1:2, :, :]
    B = tensor[:, 2:3, :, :]
    tensor = 0.299 * R + 0.587 * G + 0.114 * B
    return tensor



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




class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
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

        if mode == 'character':
            label = [str_filt(i, 'lower')+'-' for i in label]
            length_tensor, input_tensor, text_gt = label_encoder(label)
        mse_loss = self.mse_loss(sr_img, hr_img)
        if mode == 'plain':
            attention_loss = 0
            recognition_loss = 0
        else:
            hr_pred, word_attention_map_gt, hr_correct_list = transformer(to_gray_tensor(hr_img), length_tensor, input_tensor, test=False)
            sr_pred, word_attention_map_pred, sr_correct_list = transformer(to_gray_tensor(sr_img), length_tensor, input_tensor, test=False)
            attention_loss = self.l1_loss(word_attention_map_gt, word_attention_map_pred)
            recognition_loss = self.l1_loss(hr_pred, sr_pred)

        if mode == 'character':
            loss = mse_loss + attention_loss * 10 + recognition_loss * 0.0005
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

