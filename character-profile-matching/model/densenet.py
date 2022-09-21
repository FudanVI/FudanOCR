import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from collections import OrderedDict

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                   kernel_size=1, stride=1, bias=False)),
        self.add_module('norm1', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu1', nn.ReLU(inplace=True)),

        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                   kernel_size=3, stride=1, padding=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),

        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, iblock):
        super(_Transition, self).__init__()
        assert iblock < 4, "There are maximal 4 blocks."
        self.ks = [2, 2, 2]
        self.h_ss = [2, 2, 2]
        self.w_ss = [1, 1, 1]
        self.w_pad = [1, 1, 1]
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        # self.add_module('dropout', nn.Dropout(p=0.2))
        self.add_module('norm', nn.BatchNorm2d(num_output_features))
        self.add_module('relu', nn.ReLU(inplace=True))

#         self.add_module('pool', nn.AvgPool2d((self.ks[iblock], self.ks[iblock]),
#                                              (self.h_ss[iblock], self.w_ss[iblock]),
#                                              (0, self.w_pad[iblock])))


class DenseNet(nn.Module):
    def __init__(self, num_in, growth_rate=32, block_config=(16,16,16),
                 num_init_features=64, bn_size=4, drop_rate=0):
        super(DenseNet, self).__init__()

        # 进入特征抽取层之前的卷积操作
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_in, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=2, stride=2)),
            # ('dropout0', nn.Dropout(p=0.2))
        ]))

        num_features = num_init_features

        # Each denseblock
        # 进入每一个DenseBlock，进行特征抽取
        for i, num_layers in enumerate(block_config):
            # 注册每一个block
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)

            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, iblock=i)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        # self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        self.conv_last = nn.Conv2d(1808,1024,1,1)
        self.bn_last = nn.BatchNorm2d(1024)
        self.relu_last = nn.ReLU(inplace=True)

        # Official init from torch repo
        print("Initializing Dense Net weights...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        # out = F.relu(features, inplace=True)
        # out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        features = self.relu_last(self.bn_last(self.conv_last(features)))

        return features

def DenseNet51(**kwargs):
    model = DenseNet(num_in=3, num_init_features=64, growth_rate=64, block_config=(16,16,16),
                     **kwargs)
    return model