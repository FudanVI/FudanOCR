import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
import numpy as np
from torch.autograd import Variable

alphabet = '-0123456789abcdefghijklmnopqrstuvwxyz'

def get_alphabet_len():
    return len(alphabet)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    '''
    这里使用偏移k=2是因为前面补位embedding
    '''
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None, attention_map=None):
    "Compute 'Scaled Dot Product Attention'"

    if attention_map is not None:
        return torch.matmul(attention_map, value), attention_map

    # print(mask)
    # print("在attention模块,q_{0}".format(query.shape))
    # print("在attention模块,k_{0}".format(key.shape))
    # print("在attention模块,v_{0}".format(key.shape))
    # print("mask :",mask)
    # print("mask的尺寸为",mask.shape)

    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        # print(mask)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    else:
        pass
        # print("scores", scores.shape)
    '''
    工程
    这里的scores需要再乘上一个prob
    这个prob的序号要和word_index对应！
    '''
    # if align is not None:
    #     scores = scores * align.unsqueeze(1)

    # mask在这里，可以改成可以直接传入的方式
    p_attn = F.softmax(scores, dim=-1)

    # if mask is not None:
    #     print("p_attn",p_attn)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # print("p_attn", p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, compress_attention=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.compress_attention = compress_attention
        self.compress_attention_linear = nn.Linear(h, 1)

    def forward(self, query, key, value, mask=None, attention_map=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # cnt = 0
        # for l , x in zip(self.linears, (query, key, value)):
        #     print(cnt,l,x)
        #     cnt += 1

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # print("在Multi中，query的尺寸为", query.shape)
        # print("在Multi中，key的尺寸为", key.shape)
        # print("在Multi中，value的尺寸为", value.shape)

        # 2) Apply attention on all the projected vectors in batch.
        # 这一步要决定是否直接传入attention mask进去
        x, attention_map = attention(query, key, value, mask=mask,
                                     dropout=self.dropout, attention_map=attention_map)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        # 因为multi-head可能会产生若干张注意力map，应该将其合并为一张
        # if self.compress_attention:
        #     batch, head, s1, s2 = attention_map.shape
        #     attention_map = attention_map.permute(0, 2, 3, 1).contiguous()
        #     attention_map = self.compress_attention_linear(attention_map).permute(0, 3, 1, 2).contiguous()

        return self.linears[-1](x), attention_map


class ResNet(nn.Module):
    '''特征抽取网络'''

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
        self.layer2 = self._make_layer(block, 256, 256, layers[1])
        self.layer2_conv = nn.Conv2d(256, 256, 3, 1, 1)
        self.layer2_bn = nn.BatchNorm2d(256)
        self.layer2_relu = nn.ReLU(inplace=True)

        self.layer3_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer3 = self._make_layer(block, 256, 512, layers[2])
        self.layer3_conv = nn.Conv2d(512, 512, 3, 1, 1)
        self.layer3_bn = nn.BatchNorm2d(512)
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
        # self.inplanes = planes * block.expansion
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

        # x = self.layer2_pool(x)
        x = self.layer2(x)
        x = self.layer2_conv(x)
        x = self.layer2_bn(x)
        x = self.layer2_relu(x)

        # x = self.layer3_pool(x)
        x = self.layer3(x)
        x = self.layer3_conv(x)
        x = self.layer3_bn(x)
        x = self.layer3_relu(x)

        # x = self.layer4_pool(x)
        x = self.layer4(x)
        x = self.layer4_conv2(x)
        x = self.layer4_conv2_bn(x)
        x = self.layer4_conv2_relu(x)

        return x


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
        # out = self.se(out)

        out += residual
        out = self.relu(out)

        return out


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        '''这个地方去掉，做pe的对照实验'''
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # print(features)
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
        # return F.softmax(self.proj(x))
        return self.proj(x)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        embed = self.lut(x) * math.sqrt(self.d_model)
        # print("embed",embed)
        # embed = self.lut(x)
        # print(embed.requires_grad)
        return embed


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.mask_multihead = MultiHeadedAttention(h=16, d_model=1024, dropout=0.1)
        self.mul_layernorm1 = LayerNorm(features=1024)

        self.multihead = MultiHeadedAttention(h=16, d_model=1024, dropout=0.1, compress_attention=True)
        self.mul_layernorm2 = LayerNorm(features=1024)

        self.pff = PositionwiseFeedForward(1024, 2048)
        self.mul_layernorm3 = LayerNorm(features=1024)

    def forward(self, text, conv_feature, attention_map=None):
        text_max_length = text.shape[1]
        mask = subsequent_mask(text_max_length).cuda()

        result = text
        # 自注意力照样进行
        result = self.mul_layernorm1(result + self.mask_multihead(result, result, result, mask=mask)[0])

        b, c, h, w = conv_feature.shape
        conv_feature = conv_feature.view(b, c, h * w).permute(0, 2, 1).contiguous()
        # 这一步需要适当改一下
        # print("特征图",conv_feature.shape, h, w)

        word_image_align, attention_map = self.multihead(result, conv_feature, conv_feature, mask=None, attention_map=attention_map)
        # print("注意力图",attention_map.shape)
        result = self.mul_layernorm2(result + word_image_align)

        result = self.mul_layernorm3(result + self.pff(result))

        return result, attention_map


class BasicBlock(nn.Module):
    '''
    构成ResNet的残差块
    '''

    def __init__(self, inplanes, planes, downsample):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        # self.se = SELayer(planes, 16)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.se(out)

        if self.downsample != None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        # CNN架构，这里选择带预训练的Resnet34
        self.cnn = ResNet(num_in=1, block=BasicBlock, layers=[1, 2, 5, 3])

    def forward(self, input):
        conv_result = self.cnn(input)
        return conv_result


class Transformer(nn.Module):

    def __init__(self):
        super(Transformer, self).__init__()

        # word part
        word_n_class = get_alphabet_len()
        self.embedding_word = Embeddings(512, word_n_class)
        self.pe = PositionalEncoding(d_model=512, dropout=0.1, max_len=5000)

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.generator_word = Generator(1024, word_n_class)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, image, text_length, text_input, test=False, attention_map=None):

        conv_feature = self.encoder(image) # batch, 1024, 8, 32
        text_embedding = self.embedding_word(text_input) # batch, text_max_length, 512
        postion_embedding = self.pe(torch.zeros(text_embedding.shape).cuda()).cuda() # batch, text_max_length, 512
        text_input_with_pe = torch.cat([text_embedding, postion_embedding], 2) # batch, text_max_length, 1024
        batch, seq_len, _ = text_input_with_pe.shape

        # 单层decoder
        text_input_with_pe, word_attention_map = self.decoder(text_input_with_pe, conv_feature, attention_map=attention_map)

        # 解码
        word_decoder_result = self.generator_word(text_input_with_pe)

        # 训练阶段
        total_length = torch.sum(text_length).data
        probs_res = torch.zeros(total_length, get_alphabet_len()).type_as(word_decoder_result.data)

        start = 0
        for index, length in enumerate(text_length):
            # print("index, length", index,length)
            length = length.data
            probs_res[start:start + length, :] = word_decoder_result[index, 0:0 + length, :]
            start = start + length

        # 训练阶段
        if not test:
            return probs_res, word_attention_map, None
        # 测试阶段
        else:
            # return word_decoder_result, word_attention_map, text_input_with_pe
            return word_decoder_result


        # return {
        #     'word_result': word_decoder_result,
        #     'word_attention_map': word_attention_map,
        # }


if __name__ == '__main__':
    image = torch.Tensor(32,1,32,128).cuda()
    model = Encoder().cuda()
    output = model(image)
    print(output.shape)

    word_n_class = get_alphabet_len()
    embedding_word = Embeddings(512, word_n_class).cuda()
    word = torch.Tensor([[1,2,3],[4,5,6]]).long().cuda() # 2,3 -> 2, 3, 512
    embedding = embedding_word(word)
    print(embedding.shape)

    text_input = torch.Tensor(32,10,1024).cuda()
    image_input = torch.Tensor(32,1024,8,32).cuda()
    decoder = Decoder().cuda()
    decoder_output = decoder(text_input, image_input)
    print(decoder_output[0].shape)
    print(decoder_output[1].shape)

    image = torch.Tensor(4,1,32,128).cuda()
    text_length = torch.Tensor([3,2,2,4]).long().cuda()
    text_input = torch.Tensor([[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]]).long().cuda()
    transformer = Transformer().cuda()
    output = transformer(image, text_length, text_input)
    print('build success!')
    print(output['word_result'].shape)
