import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from PIL import Image
from IPython import embed
from torchvision import transforms
import string

english_alphabet = '-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
english_dict = {}
for index in range(len(english_alphabet)):
    english_dict[english_alphabet[index]] = index

from loss.transformer import Transformer
transformer = Transformer().cuda()
transformer = nn.DataParallel(transformer)
transformer.load_state_dict(torch.load('/home/db/TextZoom/src/4000_0.73.pth'))
transformer.eval()

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

class ImageLoss(nn.Module):
    def __init__(self, gradient=True, loss_weight=[20, 1e-4]):
        super(ImageLoss, self).__init__()
        self.mse = nn.MSELoss()

        self.GPLoss = GradientPriorLoss()
        self.gradient = gradient
        self.loss_weight = loss_weight

        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, out_images, target_images, label):

        label = [str_filt(i, 'upper') + '-' for i in label]
        length_tensor, input_tensor, text_gt = label_encoder(label)


        mse_loss = self.loss_weight[0] * self.mse(out_images, target_images) + \
               self.loss_weight[1] * self.GPLoss(out_images[:, :3, :, :], target_images[:, :3, :, :])

        # 得到注意力损失
        # 从HR和SR中获得Attention map，再做一次l1 loss
        _, word_attention_map_gt = transformer(to_gray_tensor(target_images), length_tensor, input_tensor, test=False)
        _, word_attention_map_pred = transformer(to_gray_tensor(out_images), length_tensor, input_tensor, test=False)
        attention_loss = self.l1_loss(word_attention_map_gt, word_attention_map_pred)

        # 内容损失
        pred, _ = transformer(to_gray_tensor(out_images), length_tensor, input_tensor, test=False,
                              attention_map=word_attention_map_gt)
        recognition_loss = self.ce_loss(pred, text_gt)

        loss = mse_loss + attention_loss + 0.001 * recognition_loss



        return loss, mse_loss , attention_loss ,recognition_loss

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


class GradientPriorLoss(nn.Module):
    def __init__(self, ):
        super(GradientPriorLoss, self).__init__()
        self.func = nn.L1Loss()

    def forward(self, out_images, target_images):
        map_out = self.gradient_map(out_images)
        map_target = self.gradient_map(target_images)
        return self.func(map_out, map_target)

    @staticmethod
    def gradient_map(x):
        batch_size, channel, h_x, w_x = x.size()
        r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
        l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
        t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
        b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
        xgrad = torch.pow(torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2)+1e-6, 0.5)
        return xgrad


if __name__ == '__main__':
    im1=Image.open('../tt.jpg')
    im1=transforms.ToTensor()(im1)
    im1=im1.unsqueeze(0)
    im2 = Image.open('../tt1.jpg')
    im2 = transforms.ToTensor()(im2)
    im2 = im2.unsqueeze(0)
    embed()
