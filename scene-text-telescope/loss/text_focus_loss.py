import cv2
import sys
import time
import torch
import string
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from loss.transformer import Transformer

# ce_loss = torch.nn.CrossEntropyLoss()
from loss.weight_ce_loss import weight_cross_entropy


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


class TextFocusLoss(nn.Module):
    def __init__(self, args):
        super(TextFocusLoss, self).__init__()
        self.args = args
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        self.english_alphabet = '-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.english_dict = {}
        for index in range(len(self.english_alphabet)):
            self.english_dict[self.english_alphabet[index]] = index

        self.build_up_transformer()

    def build_up_transformer(self):

        transformer = Transformer().cuda()
        transformer = nn.DataParallel(transformer)
        transformer.load_state_dict(torch.load('./dataset/mydata/pretrain_transformer.pth'))
        transformer.eval()
        self.transformer = transformer

    def label_encoder(self, label):
        batch = len(label)

        length = [len(i) for i in label]
        length_tensor = torch.Tensor(length).long().cuda()

        max_length = max(length)
        input_tensor = np.zeros((batch, max_length))
        for i in range(batch):
            for j in range(length[i] - 1):
                input_tensor[i][j + 1] = self.english_dict[label[i][j]]

        text_gt = []
        for i in label:
            for j in i:
                text_gt.append(self.english_dict[j])
        text_gt = torch.Tensor(text_gt).long().cuda()

        input_tensor = torch.from_numpy(input_tensor).long().cuda()
        return length_tensor, input_tensor, text_gt


    def forward(self,sr_img, hr_img, label):

        mse_loss = self.mse_loss(sr_img, hr_img)

        if self.args.text_focus:
            label = [str_filt(i, 'lower')+'-' for i in label]
            length_tensor, input_tensor, text_gt = self.label_encoder(label)
            hr_pred, word_attention_map_gt, hr_correct_list = self.transformer(to_gray_tensor(hr_img), length_tensor,
                                                                          input_tensor, test=False)
            sr_pred, word_attention_map_pred, sr_correct_list = self.transformer(to_gray_tensor(sr_img), length_tensor,
                                                                            input_tensor, test=False)
            attention_loss = self.l1_loss(word_attention_map_gt, word_attention_map_pred)
            # recognition_loss = self.l1_loss(hr_pred, sr_pred)
            recognition_loss = weight_cross_entropy(sr_pred, text_gt)
            loss = mse_loss + attention_loss * 10 + recognition_loss * 0.0005
            return loss, mse_loss, attention_loss, recognition_loss
        else:
            attention_loss = -1
            recognition_loss = -1
            loss = mse_loss
            return loss, mse_loss, attention_loss, recognition_loss