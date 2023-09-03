import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
import numpy as np

from torch.autograd import Variable
from .reconstruct import GeneratorIMG_H_1
from util import get_alphabet

torch.set_printoptions(precision=None, threshold=1000000, edgeitems=None, linewidth=None, profile=None)
alphabet = get_alphabet()


def getWordAlphabetLen():
    return len(alphabet)

class CRNN_Backbone(nn.Module):
    def __init__(self, nc, LeakyRelu=False):
        super(CRNN_Backbone, self).__init__()

        kernal_size = [3, 3, 3, 3, 3, 3, 3]
        padding_size = [1, 1, 1, 1, 1, 1, 1]
        stride_size = [1, 1, 1, 1, 1, 1, 1]
        channels = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, BatchNormalize=False):
            if i == 0:
                nIn = nc
            else:
                nIn = channels[i-1]
            nOut = channels[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, kernal_size[i], stride_size[i], padding_size[i]))
            if BatchNormalize:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if LeakyRelu:
                cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
        convRelu(0)
        convRelu(1)
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2,2), (2,2), (0,0)))
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2,2), (2,2), (0,0)))
        convRelu(6, True)

        self.cnn = cnn

    def forward(self, input):
        conv = self.cnn(input)
        return conv


class Bottleneck(nn.Module):

    def __init__(self, input_dim):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, input_dim, 1)
        self.bn1 = nn.BatchNorm2d(input_dim)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(input_dim, input_dim, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(input_dim)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, downsample):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample != None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, num_in, block, layers):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(num_in, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d((2, 2), (2, 2))

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.layer1_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer1 = self._make_layer(block, 128, 256, layers[0])
        self.layer1_conv = nn.Conv2d(256, 256, 3, 1, 1)
        self.layer1_bn = nn.BatchNorm2d(256)
        self.layer1_relu = nn.ReLU(inplace=True)

        self.layer2_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer2 = self._make_layer(block, 256, 512, layers[1])
        self.layer2_conv = nn.Conv2d(512, 512, 3, 1, 1)
        self.layer2_bn = nn.BatchNorm2d(512)
        self.layer2_relu = nn.ReLU(inplace=True)

        self.layer3_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer3 = self._make_layer(block, 512, 1024, layers[2])
        self.layer3_conv = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.layer3_bn = nn.BatchNorm2d(1024)
        self.layer3_relu = nn.ReLU(inplace=True)

        self.layer4_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer4 = self._make_layer(block, 512, 512, layers[3])
        self.layer4_conv2 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.layer4_conv2_bn = nn.BatchNorm2d(1024)
        self.layer4_conv2_relu = nn.ReLU(inplace=True)

    def _make_layer(self, block, inplanes, planes, blocks):

        if inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 3, 1, 1),
                nn.BatchNorm2d(planes), )
        else:
            downsample = None
        layers = []
        layers.append(block(inplanes, planes, downsample))
        for i in range(1, blocks):
            layers.append(block(planes, planes, downsample=None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.layer1_pool(x)
        x = self.layer1(x)
        x = self.layer1_conv(x)
        x = self.layer1_bn(x)
        x = self.layer1_relu(x)

        x = self.layer2_pool(x)
        x = self.layer2(x)
        x = self.layer2_conv(x)
        x = self.layer2_bn(x)
        x = self.layer2_relu(x)

        x = self.layer3(x)
        x = self.layer3_conv(x)
        x = self.layer3_bn(x)
        x = self.layer3_relu(x)

        return x


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=7000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, compress_attention=True):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.compress_attention = compress_attention
        self.compress_attention_linear = nn.Linear(h, 1)

    def forward(self, query, key, value, mask=None, align=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attention_map = attention(query, key, value, mask=mask,
                                     dropout=self.dropout, align=align)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        if self.compress_attention:
            batch, head, s1, s2 = attention_map.shape
            attention_map = attention_map.permute(0, 2, 3, 1).contiguous()
            attention_map = self.compress_attention_linear(attention_map).permute(0, 3, 1, 2).contiguous()
        # print('transformer file:', attention_map)

        return self.linears[-1](x), attention_map


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None, align=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    else:
        pass

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.proj(x)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        embed = self.lut(x) * math.sqrt(self.d_model)
        return embed


class Decoder(nn.Module):

    def __init__(self, channel):
        super(Decoder, self).__init__()

        self.mask_multihead = MultiHeadedAttention(h=4, d_model=channel, dropout=0.1)
        self.mul_layernorm1 = LayerNorm(features=channel)

        self.multihead = MultiHeadedAttention(h=4, d_model=channel, dropout=0.1, compress_attention=True)
        self.mul_layernorm2 = LayerNorm(features=channel)

        self.pff = PositionwiseFeedForward(channel, channel*2)
        self.mul_layernorm3 = LayerNorm(features=channel)

    def forward(self, text, conv_feature):
        text_max_length = text.shape[1]
        mask = subsequent_mask(text_max_length).cuda()

        result = text
        result = self.mul_layernorm1(result + self.mask_multihead(result, result, result, mask=mask)[0])

        b, c, h, w = conv_feature.shape
        conv_feature = conv_feature.view(b, c, h * w).permute(0, 2, 1).contiguous()
        word_image_align, attention_map = self.multihead(result, conv_feature, conv_feature, mask=None)
        result = self.mul_layernorm2(result + word_image_align)

        result = self.mul_layernorm3(result + self.pff(result))

        return result, attention_map


class DirectionExtractor(nn.Module):
    def __init__(self):
        super(DirectionExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1024, 512, 1)
        self.global_pool = nn.AvgPool2d((4, 32), stride=1)
        self.linear = nn.Linear(512, 512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.global_pool(x).squeeze()
        x = self.linear(x)
        return x

class DirectionCrossAttn(nn.Module):
    def __init__(self):
        super(DirectionCrossAttn, self).__init__()
        self.multihead = MultiHeadedAttention(h=4, d_model=1024, dropout=0.1, compress_attention=False)
        self.mul_layernorm1 = LayerNorm(features=1024)

    def forward(self, q, k, v):
        b, l, c = k.size()
        q = q.unsqueeze(1).repeat(1, l, 1)
        result = self.multihead(q, k, v, mask=None)[0]
        result = self.mul_layernorm1(result)
        return result


class Transformer(nn.Module):

    def __init__(self):
        super(Transformer, self).__init__()

        self.word_n_class = len(alphabet)
        self.embedding_word = Embeddings(256, self.word_n_class)
        self.pe = PositionalEncoding(d_model=256, dropout=0.1, max_len=7000)
        self.encoder = ResNet(num_in=3, block=BasicBlock, layers=[3,4,6,3]).cuda()
        self.features_compress = nn.Conv2d(in_channels=128, out_channels=4, kernel_size=1, stride=1, padding=0)

        self.content_extractor = nn.Conv2d(1024, 512, 1)
        self.direction_extractor = DirectionExtractor()
        self.direction_cls = nn.Linear(512, 2)
        self.query_direction = DirectionCrossAttn()

        self.conv_feature_vq = nn.Conv2d(512, 3, 1)
        self.dir_feature_vq = nn.Conv2d(512, 3, 1)
        self.all_feat_vq_decode = nn.Conv2d(7, 1024, 1)

        self.construct = GeneratorIMG_H_1()
        self.decoder = Decoder(channel=512)
        self.generator_word = Generator(512, self.word_n_class)


    def forward(self, image, text_length, text_input, is_v_char=None, conv_feature=None, test=False, att_map=None):

        if conv_feature is None:
            conv_feature_raw = self.encoder(image)
            conv_feature = self.content_extractor(conv_feature_raw)

        if text_length is None:
            return {
                'conv': conv_feature,
            }

        text_embedding = self.embedding_word(text_input)
        postion_embedding = self.pe(torch.zeros(text_embedding.shape).cuda()).cuda()
        text_input_with_pe = torch.cat([text_embedding, postion_embedding], 2)
        batch, seq_len, _ = text_input_with_pe.shape

        text_input_with_pe, attention_map = self.decoder(text_input_with_pe, conv_feature)

        if not test:
            b, c, h, w = conv_feature.size()
            _, l, _ = text_input_with_pe.size()
            attention_map = attention_map.squeeze().view(b, l, h*w)
            conv_feature = conv_feature.view(b, c, h*w)
            char_maps = torch.mul(conv_feature.unsqueeze(1), attention_map.unsqueeze(2))
            char_maps = self.features_compress(char_maps.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

            direction_feat_raw = self.direction_extractor(conv_feature_raw)
            direction_res = self.direction_cls(direction_feat_raw)

            total_len = torch.sum(text_length).item() - len(text_length)
            char_maps_all = torch.zeros(total_len, c, 4).type_as(char_maps)
            direction_feat = torch.zeros(total_len, c).type_as(direction_feat_raw)

            start = 0
            for index, length in enumerate(text_length):
                length = length.data
                char_maps_all[start:start + length - 1, :] = char_maps[index, 0:0 + length - 1, :, :]
                direction_feat[start:start + length - 1, :] = direction_feat_raw[index]
                start = start + length - 1

            raw_imgs = self.construct(torch.cat([char_maps_all.view(total_len, c, 2, 2), direction_feat.unsqueeze(2).unsqueeze(3).repeat(1, 1, 2, 2)], dim=1))

            dir_feat_hor, dir_feat_ver = [], []
            for i in range(total_len):
                if is_v_char[i] == 0:
                    dir_feat_hor.append(direction_feat[i].unsqueeze(0))
                else:
                    dir_feat_ver.append(direction_feat[i].unsqueeze(0))
            len_dir_feat_hor = len(dir_feat_hor)
            len_dir_feat_ver = len(dir_feat_ver)
            if len_dir_feat_ver == 0 or len_dir_feat_hor == 0:
                new_imgs = None
            else:
                dir_feat_hor = torch.cat(dir_feat_hor, dim=0)
                dir_feat_ver = torch.cat(dir_feat_ver, dim=0)

                dir_feat_ex = []
                for i in range(total_len):
                    if is_v_char[i] == 0:
                        dir_feat_ex.append(dir_feat_ver[i % len_dir_feat_ver].unsqueeze(0))
                    else:
                        dir_feat_ex.append(dir_feat_hor[i % len_dir_feat_hor].unsqueeze(0))
                dir_feat_ex = torch.cat(dir_feat_ex, dim=0)
                new_imgs = self.construct(torch.cat([char_maps_all.view(total_len, c, 2, 2), dir_feat_ex.unsqueeze(2).unsqueeze(3).repeat(1, 1, 2, 2)], dim=1))

        word_decoder_result = self.generator_word(text_input_with_pe)

        if test:
            return {
                'pred': word_decoder_result,
                'map': attention_map,
                'conv': conv_feature,
            }

        else:
            total_length = torch.sum(text_length).data
            probs_res = torch.zeros(total_length, self.word_n_class).type_as(word_decoder_result.data)

            start = 0
            for index, length in enumerate(text_length):
                length = length.data
                probs_res[start:start + length, :] = word_decoder_result[index, 0:0 + length, :]
                start = start + length

            return {
                'pred': probs_res,
                'map': attention_map,
                'conv': conv_feature,
                'direction_res': direction_res,
                'raw_imgs': raw_imgs,
                'new_imgs': new_imgs,
            }

if __name__ == '__main__':
    net = ResNet(num_in=3, block=BasicBlock, layers=[3, 4, 6, 3]).cuda()
    image = torch.Tensor(8, 3, 64, 64).cuda()
    result = net(image)
    print(result.shape)
    pass


