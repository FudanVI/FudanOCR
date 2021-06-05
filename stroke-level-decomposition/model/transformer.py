import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math, copy
import numpy as np
import time
from torch.autograd import Variable

from model.tps_spatial_transformer import TPSSpatialTransformer
from model.stn_head import STNHead

import torchvision.models as models

torch.set_printoptions(precision=None, threshold=1000000, edgeitems=None, linewidth=None, profile=None)

from config import config

from model.densenet import DenseNet51
from model.vgg import VGG

from util import get_alphabet
alphabet = get_alphabet(config['mode'])
# 下次跑代码的时候记得改


def getWordAlphabetLen():
    return len(alphabet)


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

        # x = self.layer1_pool(x)
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


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=7000):
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

    def forward(self, query, key, value, mask=None, align=None):
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
        x, attention_map = attention(query, key, value, mask=mask,
                                     dropout=self.dropout, align=align)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        # 因为multi-head可能会产生若干张注意力map，应该将其合并为一张
        # if self.compress_attention:
        #     batch, head, s1, s2 = attention_map.shape
        #     attention_map = attention_map.permute(0, 2, 3, 1).contiguous()
        #     attention_map = self.compress_attention_linear(attention_map).permute(0, 3, 1, 2).contiguous()

        return self.linears[-1](x), attention_map


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    '''
    这里使用偏移k=2是因为前面补位embedding
    '''
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None, align=None):
    "Compute 'Scaled Dot Product Attention'"

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
    #     # print("score", scores.shape)
    #     # print("align", align.shape)
    #
    #     scores = scores * align.unsqueeze(1)

    p_attn = F.softmax(scores, dim=-1)

    # if mask is not None:
    #     print("p_attn",p_attn)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # print("p_attn", p_attn)
    return torch.matmul(p_attn, value), p_attn


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

        self.mask_multihead = MultiHeadedAttention(h=4, d_model=1024, dropout=0.1)
        self.mul_layernorm1 = LayerNorm(features=1024)

        '''
        工程：这里把head修改为1
        '''
        self.multihead = MultiHeadedAttention(h=4, d_model=1024, dropout=0.1, compress_attention=True)
        self.mul_layernorm2 = LayerNorm(features=1024)

        self.pff = PositionwiseFeedForward(1024, 2048)
        self.mul_layernorm3 = LayerNorm(features=1024)

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


class RNN_Decoder(nn.Module):

    def __init__(self):
        super(RNN_Decoder, self).__init__()
        pass

    def forward(self, text, conv_feature):
        # text: batch, max_length, embedding
        # conv_feature: batch, channel, height, weight

        # result: batch, max_length, dim
        # attention_map: ...

        return result, attention_map


class Transformer(nn.Module):

    def __init__(self, mode):
        super(Transformer, self).__init__()

        self.mode = mode

        # if mode == 'character':
        #     alphabet = '<啊阿埃挨哎唉哀皑癌蔼矮艾碍爱隘鞍氨安俺按暗岸胺案肮昂盎凹敖熬翱袄傲奥懊澳芭捌扒叭吧笆八疤巴拔跋靶把耙坝霸罢爸白柏百摆佰败拜稗斑班搬扳般颁板版扮拌伴瓣半办绊邦帮梆榜膀绑棒磅蚌镑傍谤苞胞包褒剥薄雹保堡饱宝抱报暴豹鲍爆杯碑悲卑北辈背贝钡倍狈备惫焙被奔苯本笨崩绷甭泵蹦迸逼鼻比鄙笔彼碧蓖蔽毕毙毖币庇痹闭敝弊必辟壁臂避陛鞭边编贬扁便变卞辨辩辫遍标彪膘表鳖憋别瘪彬斌濒滨宾摈兵冰柄丙秉饼炳病并玻菠播拨钵波博勃搏铂箔伯帛舶脖膊渤泊驳捕卜哺补埠不布步簿部怖擦猜裁材才财睬踩采彩菜蔡餐参蚕残惭惨灿苍舱仓沧藏操糙槽曹草厕策侧册测层蹭插叉茬茶查碴搽察岔差诧拆柴豺搀掺蝉馋谗缠铲产阐颤昌猖场尝常长偿肠厂敞畅唱倡超抄钞朝嘲潮巢吵炒车扯撤掣彻澈郴臣辰尘晨忱沉陈趁衬撑称城橙成呈乘程惩澄诚承逞骋秤吃痴持匙池迟弛驰耻齿侈尺赤翅斥炽充冲虫崇宠抽酬畴踌稠愁筹仇绸瞅丑臭初出橱厨躇锄雏滁除楚础储矗搐触处揣川穿椽传船喘串疮窗幢床闯创吹炊捶锤垂春椿醇唇淳纯蠢戳绰疵茨磁雌辞慈瓷词此刺赐次聪葱囱匆从丛凑粗醋簇促蹿篡窜摧崔催脆瘁粹淬翠村存寸磋撮搓措挫错搭达答瘩打大呆歹傣戴带殆代贷袋待逮怠耽担丹单郸掸胆旦氮但惮淡诞弹蛋当挡党荡档刀捣蹈倒岛祷导到稻悼道盗德得的蹬灯登等瞪凳邓堤低滴迪敌笛狄涤翟嫡抵底地蒂第帝弟递缔颠掂滇碘点典靛垫电佃甸店惦奠淀殿碉叼雕凋刁掉吊钓调跌爹碟蝶迭谍叠丁盯叮钉顶鼎锭定订丢东冬董懂动栋侗恫冻洞兜抖斗陡豆逗痘都督毒犊独读堵睹赌杜镀肚度渡妒端短锻段断缎堆兑队对墩吨蹲敦顿囤钝盾遁掇哆多夺垛躲朵跺舵剁惰堕蛾峨鹅俄额讹娥恶厄扼遏鄂饿恩而儿耳尔饵洱二贰发罚筏伐乏阀法珐藩帆番翻樊矾钒繁凡烦反返范贩犯饭泛坊芳方肪房防妨仿访纺放菲非啡飞肥匪诽吠肺废沸费芬酚吩氛分纷坟焚汾粉奋份忿愤粪丰封枫蜂峰锋风疯烽逢冯缝讽奉凤佛否夫敷肤孵扶拂辐幅氟符伏俘服浮涪福袱弗甫抚辅俯釜斧脯腑府腐赴副覆赋复傅付阜父腹负富讣附妇缚咐噶嘎该改概钙盖溉干甘杆柑竿肝赶感秆敢赣冈刚钢缸肛纲岗港杠篙皋高膏羔糕搞镐稿告哥歌搁戈鸽胳疙割革葛格蛤阁隔铬个各给根跟耕更庚羹埂耿梗工攻功恭龚供躬公宫弓巩汞拱贡共钩勾沟苟狗垢构购够辜菇咕箍估沽孤姑鼓古蛊骨谷股故顾固雇刮瓜剐寡挂褂乖拐怪棺关官冠观管馆罐惯灌贯光广逛瑰规圭硅归龟闺轨鬼诡癸桂柜跪贵刽辊滚棍锅郭国果裹过哈骸孩海氦亥害骇酣憨邯韩含涵寒函喊罕翰撼捍旱憾悍焊汗汉夯杭航壕嚎豪毫郝好耗号浩呵喝荷菏核禾和何合盒貉阂河涸赫褐鹤贺嘿黑痕很狠恨哼亨横衡恒轰哄烘虹鸿洪宏弘红喉侯猴吼厚候后呼乎忽瑚壶葫胡蝴狐糊湖弧虎唬护互沪户花哗华猾滑画划化话槐徊怀淮坏欢环桓还缓换患唤痪豢焕涣宦幻荒慌黄磺蝗簧皇凰惶煌晃幌恍谎灰挥辉徽恢蛔回毁悔慧卉惠晦贿秽会烩汇讳诲绘荤昏婚魂浑混豁活伙火获或惑霍货祸击圾基机畸稽积箕肌饥迹激讥鸡姬绩缉吉极棘辑籍集及急疾汲即嫉级挤几脊己蓟技冀季伎祭剂悸济寄寂计记既忌际妓继纪嘉枷夹佳家加荚颊贾甲钾假稼价架驾嫁歼监坚尖笺间煎兼肩艰奸缄茧检柬碱硷拣捡简俭剪减荐槛鉴践贱见键箭件健舰剑饯渐溅涧建僵姜将浆江疆蒋桨奖讲匠酱降蕉椒礁焦胶交郊浇骄娇嚼搅铰矫侥脚狡角饺缴绞剿教酵轿较叫窖揭接皆秸街阶截劫节桔杰捷睫竭洁结解姐戒藉芥界借介疥诫届巾筋斤金今津襟紧锦仅谨进靳晋禁近烬浸尽劲荆兢茎睛晶鲸京惊精粳经井警景颈静境敬镜径痉靖竟竞净炯窘揪究纠玖韭久灸九酒厩救旧臼舅咎就疚鞠拘狙疽居驹菊局咀矩举沮聚拒据巨具距踞锯俱句惧炬剧捐鹃娟倦眷卷绢撅攫抉掘倔爵觉决诀绝均菌钧军君峻俊竣浚郡骏喀咖卡咯开揩楷凯慨刊堪勘坎砍看康慷糠扛抗亢炕考拷烤靠坷苛柯棵磕颗科壳咳可渴克刻客课肯啃垦恳坑吭空恐孔控抠口扣寇枯哭窟苦酷库裤夸垮挎跨胯块筷侩快宽款匡筐狂框矿眶旷况亏盔岿窥葵奎魁傀馈愧溃坤昆捆困括扩廓阔垃拉喇蜡腊辣啦莱来赖蓝婪栏拦篮阑兰澜谰揽览懒缆烂滥琅榔狼廊郎朗浪捞劳牢老佬姥酪烙涝勒乐雷镭蕾磊累儡垒擂肋类泪棱楞冷厘梨犁黎篱狸离漓理李里鲤礼莉荔吏栗丽厉励砾历利傈例俐痢立粒沥隶力璃哩俩联莲连镰廉怜涟帘敛脸链恋炼练粮凉梁粱良两辆量晾亮谅撩聊僚疗燎寥辽潦了撂镣廖料列裂烈劣猎琳林磷霖临邻鳞淋凛赁吝拎玲菱零龄铃伶羚凌灵陵岭领另令溜琉榴硫馏留刘瘤流柳六龙聋咙笼窿隆垄拢陇楼娄搂篓漏陋芦卢颅庐炉掳卤虏鲁麓碌露路赂鹿潞禄录陆戮驴吕铝侣旅履屡缕虑氯律率滤绿峦挛孪滦卵乱掠略抡轮伦仑沦纶论萝螺罗逻锣箩骡裸落洛骆络妈麻玛码蚂马骂嘛吗埋买麦卖迈脉瞒馒蛮满蔓曼慢漫谩芒茫盲氓忙莽猫茅锚毛矛铆卯茂冒帽貌贸么玫枚梅酶霉煤没眉媒镁每美昧寐妹媚门闷们萌蒙檬盟锰猛梦孟眯醚靡糜迷谜弥米秘觅泌蜜密幂棉眠绵冕免勉娩缅面苗描瞄藐秒渺庙妙蔑灭民抿皿敏悯闽明螟鸣铭名命谬摸摹蘑模膜磨摩魔抹末莫墨默沫漠寞陌谋牟某拇牡亩姆母墓暮幕募慕木目睦牧穆拿哪呐钠那娜纳氖乃奶耐奈南男难囊挠脑恼闹淖呢馁内嫩能妮霓倪泥尼拟你匿腻逆溺蔫拈年碾撵捻念娘酿鸟尿捏聂孽啮镊镍涅您柠狞凝宁拧泞牛扭钮纽脓浓农弄奴努怒女暖虐疟挪懦糯诺哦欧鸥殴藕呕偶沤啪趴爬帕怕琶拍排牌徘湃派攀潘盘磐盼畔判叛乓庞旁耪胖抛咆刨炮袍跑泡呸胚培裴赔陪配佩沛喷盆砰抨烹澎彭蓬棚硼篷膨朋鹏捧碰坯砒霹批披劈琵毗啤脾疲皮匹痞僻屁譬篇偏片骗飘漂瓢票撇瞥拼频贫品聘乒坪苹萍平凭瓶评屏坡泼颇婆破魄迫粕剖扑铺仆莆葡菩蒲埔朴圃普浦谱曝瀑期欺栖戚妻七凄漆柒沏其棋奇歧畦崎脐齐旗祈祁骑起岂乞企启契砌器气迄弃汽泣讫掐恰洽牵扦钎铅千迁签仟谦乾黔钱钳前潜遣浅谴堑嵌欠歉枪呛腔羌墙蔷强抢橇锹敲悄桥瞧乔侨巧鞘撬翘峭俏窍切茄且怯窃钦侵亲秦琴勤芹擒禽寝沁青轻氢倾卿清擎晴氰情顷请庆琼穷秋丘邱球求囚酋泅趋区蛆曲躯屈驱渠取娶龋趣去圈颧权醛泉全痊拳犬券劝缺炔瘸却鹊榷确雀裙群然燃冉染瓤壤攘嚷让饶扰绕惹热壬仁人忍韧任认刃妊纫扔仍日戎茸蓉荣融熔溶容绒冗揉柔肉茹蠕儒孺如辱乳汝入褥软阮蕊瑞锐闰润若弱撒洒萨腮鳃塞赛三叁伞散桑嗓丧搔骚扫嫂瑟色涩森僧莎砂杀刹沙纱傻啥煞筛晒珊苫杉山删煽衫闪陕擅赡膳善汕扇缮墒伤商赏晌上尚裳梢捎稍烧芍勺韶少哨邵绍奢赊蛇舌舍赦摄射慑涉社设砷申呻伸身深娠绅神沈审婶甚肾慎渗声生甥牲升绳省盛剩胜圣师失狮施湿诗尸虱十石拾时什食蚀实识史矢使屎驶始式示士世柿事拭誓逝势是嗜噬适仕侍释饰氏市恃室视试收手首守寿授售受瘦兽蔬枢梳殊抒输叔舒淑疏书赎孰熟薯暑曙署蜀黍鼠属术述树束戍竖墅庶数漱恕刷耍摔衰甩帅栓拴霜双爽谁水睡税吮瞬顺舜说硕朔烁斯撕嘶思私司丝死肆寺嗣四伺似饲巳松耸怂颂送宋讼诵搜艘擞嗽苏酥俗素速粟僳塑溯宿诉肃酸蒜算虽隋随绥髓碎岁穗遂隧祟孙损笋蓑梭唆缩琐索锁所塌他它她塔獭挞蹋踏胎苔抬台泰酞太态汰坍摊贪瘫滩坛檀痰潭谭谈坦毯袒碳探叹炭汤塘搪堂棠膛唐糖倘躺淌趟烫掏涛滔绦萄桃逃淘陶讨套特藤腾疼誊梯剔踢锑提题蹄啼体替嚏惕涕剃屉天添填田甜恬舔腆挑条迢眺跳贴铁帖厅听烃汀廷停亭庭挺艇通桐酮瞳同铜彤童桶捅筒统痛偷投头透凸秃突图徒途涂屠土吐兔湍团推颓腿蜕褪退吞屯臀拖托脱鸵陀驮驼椭妥拓唾挖哇蛙洼娃瓦袜歪外豌弯湾玩顽丸烷完碗挽晚皖惋宛婉万腕汪王亡枉网往旺望忘妄威巍微危韦违桅围唯惟为潍维苇萎委伟伪尾纬未蔚味畏胃喂魏位渭谓尉慰卫瘟温蚊文闻纹吻稳紊问嗡翁瓮挝蜗涡窝我斡卧握沃巫呜钨乌污诬屋无芜梧吾吴毋武五捂午舞伍侮坞戊雾晤物勿务悟误昔熙析西硒矽晰嘻吸锡牺稀息希悉膝夕惜熄烯溪汐犀檄袭席习媳喜铣洗系隙戏细瞎虾匣霞辖暇峡侠狭下厦夏吓掀锨先仙鲜纤咸贤衔舷闲涎弦嫌显险现献县腺馅羡宪陷限线相厢镶香箱襄湘乡翔祥详想响享项巷橡像向象萧硝霄削哮嚣销消宵淆晓小孝校肖啸笑效楔些歇蝎鞋协挟携邪斜胁谐写械卸蟹懈泄泻谢屑薪芯锌欣辛新忻心信衅星腥猩惺兴刑型形邢行醒幸杏性姓兄凶胸匈汹雄熊休修羞朽嗅锈秀袖绣墟戌需虚嘘须徐许蓄酗叙旭序畜恤絮婿绪续轩喧宣悬旋玄选癣眩绚靴薛学穴雪血勋熏循旬询寻驯巡殉汛训讯逊迅压押鸦鸭呀丫芽牙蚜崖衙涯雅哑亚讶焉咽阉烟淹盐严研蜒岩延言颜阎炎沿奄掩眼衍演艳堰燕厌砚雁唁彦焰宴谚验殃央鸯秧杨扬佯疡羊洋阳氧仰痒养样漾邀腰妖瑶摇尧遥窑谣姚咬舀药要耀椰噎耶爷野冶也页掖业叶曳腋夜液一壹医揖铱依伊衣颐夷遗移仪胰疑沂宜姨彝椅蚁倚已乙矣以艺抑易邑屹亿役臆逸肄疫亦裔意毅忆义益溢诣议谊译异翼翌绎茵荫因殷音阴姻吟银淫寅饮尹引隐印英樱婴鹰应缨莹萤营荧蝇迎赢盈影颖硬映哟拥佣臃痈庸雍踊蛹咏泳涌永恿勇用幽优悠忧尤由邮铀犹油游酉有友右佑釉诱又幼迂淤于盂榆虞愚舆余俞逾鱼愉渝渔隅予娱雨与屿禹宇语羽玉域芋郁吁遇喻峪御愈欲狱育誉浴寓裕预豫驭鸳渊冤元垣袁原援辕园员圆猿源缘远苑愿怨院曰约越跃钥岳粤月悦阅耘云郧匀陨允运蕴酝晕韵孕匝砸杂栽哉灾宰载再在咱攒暂赞赃脏葬遭糟凿藻枣早澡蚤躁噪造皂灶燥责择则泽贼怎增憎曾赠扎喳渣札轧铡闸眨栅榨咋乍炸诈摘斋宅窄债寨瞻毡詹粘沾盏斩辗崭展蘸栈占战站湛绽樟章彰漳张掌涨杖丈帐账仗胀瘴障招昭找沼赵照罩兆肇召遮折哲蛰辙者锗蔗这浙珍斟真甄砧臻贞针侦枕疹诊震振镇阵蒸挣睁征狰争怔整拯正政帧症郑证芝枝支吱蜘知肢脂汁之织职直植殖执值侄址指止趾只旨纸志挚掷至致置帜峙制智秩稚质炙痔滞治窒中盅忠钟衷终种肿重仲众舟周州洲诌粥轴肘帚咒皱宙昼骤珠株蛛朱猪诸诛逐竹烛煮拄瞩嘱主著柱助蛀贮铸筑住注祝驻抓爪拽专砖转撰赚篆桩庄装妆撞壮状椎锥追赘坠缀谆准捉拙卓桌琢茁酌啄着灼浊兹咨资姿滋淄孜紫仔籽滓子自渍字鬃棕踪宗综总纵邹走奏揍租足卒族祖诅阻组钻纂嘴醉最罪尊遵昨左佐柞做作坐座$'
        # elif mode == 'strokelet':
        #     alphabet = '<12345$'
        # elif mode == 'character_ctw':
        #     alphabet = '<誓迭意祺厮锚红罹慌茹拷楼叉猎腺兽靠完发真树畸宿陽嫩适慈態初劳上珊寸骗威麵木纳鲜蜜泵掩隅挚粥孕谢晓宸垠凤焖娆撕梧冈代凳雏屿崋略幻认嗨籣滙益熔劵枼浆拿纸龙厉桩焚许众低營藻芸赤轰瓯農嘉洼贵叹刘腰治刑墩硕针狗反運讯陪榈犹卒穴鍋阀应围汇饱纫义開秉都謀豆何翳找栓岚埸逢處丈仃傻司闷空抚裂秒明粗繹显雨倍光戒田论篇撷靜羯矩裱示货淼灵圓钓址骂腔魂即門轻皮沛葆茜郊讲浸丘封夺簧珍战泌澋腚駿邑眯习望抱诗膠朴画逐琐回閣言荟豌碟腋佰媲廉尘去纵纺蚌歷凛海蓄除镗產緩须癜熏逅殷药株伪模契辖屯爱絨惜机炒蜗懋返雷礼兼狮珠伞邹贪鸟延绥栈拟鷗核假腕洪矗还脸缴京喀贴曾旋樣溝創轴腻杭晚锺做驿伶助檀厰训并杆灯铅茸舌谨俗願小陌税纯连择蚝碼豐沥邀将头肤铨阜分宏全坤劉崔荡米巅强盧茅龟否喽耕莅申陇礴充精暉公考酵避髓道鰲移活報犯黄失汾扇党計析順镶吹奈馒裡答收称坎愿燕叶葫舶俏癸虚祐淅貌仅購惟呈肖紡欢短钙病持划崂闵贡矢樾脖辟解朕赵控軒拱影剿项荐华濱帅八魔峭为靛车与硒虬绸孔蜘账约荤甸在柏昕频製野彌徐麒視熱唐占徳懒束跳拐害起玄侗忧渣切遨舱梭陆湃稼然阅缕薬烛隨黑泸随茵蚕菁簘尝脂配涂绝沁截吊瓷五欲颍征扎炖闽建鹰庙顶口科肽裝博穗味蘭沏您冻縁酷眀氧首姆澄开沙兢策样钨方尬棉禽億溪铠左借塱兴教莘宣铁呷锈鎮庸覺奖撈鸣到波菌原渡未帮缪检偉蚬僧偶险而耳氽锌艰父滞忙纬斋衬觅彰渔屎皇曲飮作慎飛浙廓覚華陕蘇骄豚盐臺割峰弯垒标徒下曦拌萄史西学雍泳妈頁桶养题锐怕惕销嫣璀碳擦产荊骊筑膨兜池油亩垂馈们癫臊必秦郎尿谛旁歇员冀近廠啸戴李绚掬藩总救汀蜒播片淇祖触暑衡珺次什侬清菱干食织玛佛纠匠尕閤涌箂鹿胖替蝶仟业畫爆疙鹅包受音邯撞阵輕云绕例索绪玮铺需媒按革告喬裘訪軸忠領泓詳觳逊费種壽氏妥拖璎昌胶臭嫁老犬暮哺澈诱础促扔汗诈卡豹晖葛绅皓峪态定改崛冉报仰慕绞廳栗湾页阔倾丝住丰耗哨浑睿碗审绿企遠刹凉肝超式昂洲液澳绮督罩梵铜厌學肆戏志富悔同茗累额休緹芝嗜也佳欧溧数迦艮枪杖型严昇婴赖确韵坦背昨搬取筷己莲葚紅具甲膳权孤桥泉古磐鞋圃鉑鹫手放康瀑螺嘟元譜网疲雁霜衙塑旯鴐帆币吞研灭从琦霄甩尧晌饮兵函姚冒唤敢赌较润底翅埔炸勁焊蓬勘芽册嬉動港桨罐耀庐蓟儲莼友荫瑰溶弘涮足芊盆我粉蕉璜净享輪榄玉軟復横斯稻练释诀寒朽羿级之車偭蕭钜棱蚊金粱聚群當副斤幺途钰兒茂嶺竿赐举系軍氡架婚饺寰终幸柄玻牵不现釉湯斗酝济傷跨妹诉橙可梅能钎禅折迠焦辱揭浣烯夹塔裁哆字军瞿啼鸥築傳舘瑞范沧声噪情秘漠青负浓蔽桑董俩聆焰疚邺菇各碑鳥枇吳维萌眸校栅牌继昔衞草罕聊闹氩環鎖入钻颐你麺肚募鴻匾力类梦驴白酸瘩喝蜻鸽觀夸葉阳射造陳性缸瀘炳玖姑罄引敦立攀皙赠碱据间秩萃多年盈芬橘韶驭歳芷筹裕乐斑圜托盾钧问魚涧圗鼓珲渎廖菜萦裹牦喫煌娲芦尖贯照赔站慶件亚汛吟酪荞憨卉肿顷會卓更零鄂琪倦萂彻靖跌潼阑扩径皆着患慰优雄映伦淋里罗雪台廈媛措厨括付誉亲羔砚崖黎眷瓢恰郸电艺没升月戈杨舍盎测场呱辈佣密腾擀魏松赂坛釜矫锁連蒙沟娃來夀驶棚師递聲嵩濠親轨扪鄭夕錢哗聨靶省眺芳盟淀泾吏右难廂文肢攻稽梯程塘典就奢九骨竞詩熟租哦效兮证是襲沐缤她巡羞霓帽樂团扣妍层澐稚晨喊附果合髪章晰棕牧妲独两过淑陀祭成崇细愧虎陡商匹枕郑任辐免壘锣渴驲污祸朔县聪哑绗歌议奉筆際委肩轿村鑫蛋掏似菧袖瑷诤腊矿戎肠品禹煕伽毯鞍钊脆号奎烘跶夌旨沂芋量垃讨禺壳林枢念残彩施二弟献怠前日屡醉恪纱痧萝城辅荆眠枯暖團请些保恤钛稳看泽梨胆经蒿飞筋笃咏亞務勋侵旧园粒織隍鼠壶茄吋编齐隔桂门硬冰德鹧器逃办恒室咸镜聫钵節煎宛支薪路仔懂财淄又慢扬喜刷題殊疾预廊柚傢专樓炉误胑笋很磁寝篮呼试谈現濛泪晏纶興素楽霞气莉扁窩滚允煮飲莺疝菊诊悦拒泗部私蛮腐突七如遇澤最毅琍歡葵偷衰让猫状椎誡王躺段捐倡幼忘賢胃漫霖宇番絲傑邮瓶毕舰扶瑚愁賣尚腸肥深羊亦姓啃乾拆逸锋叫晴宾鶏输腑染炅万拍澡汪蛇速挥母血绣葬钲哥孩鴿泛挖损烁茉储烫鄕甬帝破谁三子遍帐角悬吃它迁其闯串地咪渠鸭桦签语错奠諧廎协距锲浦川穩陋亏郁潮睡君熙鳌瘪衔僅扭通摸援乘心虫爪靡杰砺航南畵律尾暂布睫肴领循榻襄漾指辋阁祛弹芜蔓和旬飙另蹄扰漂贝迹东堰瑜瓦昊辆嶪藕銀咨象滇檬竖贰疯绘带邂妻煤船节歲牛禮莊衣臣赞直巢骆鷄溢急淮招档琢剧耻氯鬼癣楠桁愈体击貨鬥餐臨吗卧弥览倪自筝雹範妇砼浮椿習消详驱骐頭械虽仓劫苑谦蔡导良排杯圩厚椒壁息撼為魁单菩廛亢蜀槐躲兆萼接者柴土把杀歧壮停葡儿谋緣句仪十跃操临潞客早痘羙脱砾已缘凖役摄缎轮推娅濮究位混醇惬断蟹叁蜂汽浅守运洋绵烦伸酒吸决棵镀远樨卫蟠婆窖旦刺秀沫暨岸腌僖组變铸居给砖丫图鸪迩宪摘誠籬蕾丹荘琳伙寄追露掉疫理纲養飚积奋条孚驾忽娱锡逼疑要辛率練薄危旅岭钥浩新俬鲍給阖疼颢承茨且靚盘共峻凯兄踪朵刻满儒縣熬協藏默夢饿麥氙義悼玩棠訓饰呆恕盒炴障炬努视粘案泰褥鲤穿诏盔吨鞭脑楚登担庵贤軽昏質剂命滩故叽骇玺賓漁天幢痣暄炝抛长拓恋區购事緒搞吾贸钣衢磊龄克舒桃姜屏漕杠卢氛酗买翎恐機厢勿票栋框滾珞寇款炮兩再璟軰縂纪鐘姿坂肯爲链彧從毡貝備咽库英苯则坐蛟参拇鵝餃堵谱沱投鼻宴閏俞壬國蹈袁冬囱星遗灼农抢嫂幽樽贞孢勝动堅艾潇风瓜剃仑丧倒捕凌先脊算锂癌罪娘添来男予淘陈概個一診录汤筒袂盗氟韦薡灡补灾暴烂传與埠酿面潜齿馍謝聯籍序弱撮展廣簾胞鉴庭鳮銷肉春羚跑渗匮棘疏流邻翟像逝輝瘀贫囊琥燎悠街邦鍍汝侨弓佑桔嵗苟轧苦尊竟隐豫验匙窃弩调若晟太所店誼橄蚁蒲宜弈工藝潔汉洞被于硫振疗涵银记涛甫均履赏局醒問师撒护酱进见香俄佐民乎槽搜惩半椴彭靓冲辽选伐歹质阴股讼会好家钉磅敖戛价膜霍宅趣败高胎赁火偿黛甄软仗叔瀚装罚伏盛曹惰季榆畅丛知岛潘擅猴鐵闻邵掀扫麓访仲挂傣鹏娄韬雾殡耽陵吖崧尺調帛疮亿撤安闭彊榜拎垦驳颜妮贺浇午压雅几寨琼驰扒寿废善漆痛侯恢列巩蕴昼觉佩薯甘乔识螂普湘坞吐瑗疆璐花遊快剪嵊吵岳沉的歆悉將蟑迅衫箍芭翌饲仁醸历螃观对乃旮恩蹓置胗設闌宮胥灿痕枫等降沈丁蔚医炫糖骏哈启晋碎裤袭窝瓣闲仕阿雲烨烤眾说奥读名简滨巨沃喧眙炭還侈盖线槁焗禾崴采扦驻久墙孝翰粮點岁含馄嘻炊载冶属圈抄归憶滤啡薇套座只况圭向蒋缠芙贷負爸雞洗少毛鋼楞娠澎継违掌窄诺孟琅歴令及信搏咛拉燃闪狂倩楓惠钢绑警皂語坏散浜猛袋厂尔挽士夫饨杉辞甜缆堂炔艳极使窗千娟挑劲沌經妙邱妃烩止填抓表偢健生禦漪餠畏張祠莓菓雯余耘貢旭瞻圖当費磨塲稅乳奧留丑际书童銭咫培摩晶馨兰豪吴值颂利莱菠鲸瞩拔贿酥氣争屹狼顾窑境財基宫笛欣囧尤述闫晒易坪鄉褡隱铣此狠津碧北旺熄椰栖尹闸押侣中浒灸沪璨炼哇犇巫苔寻鱼内度霸奕哉芯脚离广長匡饶漢致馅憾傲厅达請目诸诚拨渚感岔席滁鉄待烟璧府喉牙由叙迪山釘蚂滴蝠扈亨葩骚缝袍敗材透规懈扑則术勇锯整鸡怡帕奶美倬床咯或裙般艇沾乡提勃婵脏磚笔转濟谓缺渍准宗译管格俶礫洽潤注霆赚巾黏凰挤內劝监授舟踩每權坝存鳯奏榭埋创粹牢鲁愛爬设氮化苗毁侥禧墨杷渤结死印荒夏嗑呵焕彤摇勞铭姨官雙證堡福脉迈鹤巧腩出歉搭哟塞肛盼禁挣勤麦缇伯犟灌坡俯晤菲曼央蜡拜森柜媚评育蜕嫌孙橋缔抹宠凇紧侧刊啦淡常颈臻龍漏擎橱惊亮伍稞唯奴覽龈狐铝希畔饼暗婷梓繁缅場糙哲翁柳综眼榨绽欺奇颗劈神黒爵块憩訂本莹坟肇茶粽辣抬版平陶飯板羅菀礻呐查咕尴帜么桌江總咱增囍他秋症恬马形差塌墅堤湛柱邨蕐硚稀桷珑后大紀浴卜姗便涨丽莎敏坷瞳削庞苏痤携俭悟汁伴六仆潭宋擒鋁饭橡靴昆因岩正往遵樞扛铂鋪曙功派浏铍屋瞬盅探估双融莭舜餅栏楹駕绎惯卖盏国份炎堆赣琊词枣鱿社舫羽逛匯萧皱庆凝洱仿璃穆技伊琨間咖虹莫市踏約刃拳候遮澜顿弄婕遣抗忆顺娜泡制涞兑想陷舞袜弧槟哪衷翼今诞吒煨乓倉隆谅售書四插求瘦召嘿峨垫琉冥缩荷竺嘘峡祈涉骅溜喱怀汕摊卤樟百棋屬资虾坚赈蛛笼铬吕邊御寶写身势雕巴尽沿泼寞様送绒埭覆域券墓责务耸井瑶防啄朗鮮蓝谷槓丸馋源髮比滑坊绩赶勒婧棰醋挞丶复迷鈴加貴颁赋暧洮换阻涿幕烈姐體聘詠晾貂畜殓笑錦強藤翡刀昱瑛计圳世刨廷啧女涡滥猪主胡熊痹俱询電队妩坑法尼描潢翻点邸柔喂得察帘朋瘤睛周杜第筱圆肾演仙捍钟贻祝灰發飓禄聖啊却醫话妊麟馬蝇唱粵雀兿服赫震涟实房续才翠苹葱棺鏡联試异但毫殿营塵有馆灶驼苼捷虞鸿励怎楊肌洄蒸揽彼惑風瘾濃毗辰飘阶耐珂时笕達雜步竹匝東钩術伟拾乱容鲢减瑙進驽祥兔皋汐薹樱打河乌乙够炙码环端岗瑟构蟲劍椅抵啥毂賀慧廚麯嚣寺幅渝水桐行末宝裢毒邪瑪筛永焙邓庄裳思呢跆糊柯硅泥纷亭镇旱奔备边握区搅館始笈涤贩褪蕓晦蔻执用厦诂恭挡舵柑婦凡鹭旗紫走石維猕寳璞乒修齊隧户壹圾醬旷揚頂丙玲窨蛙蝎集交吉纽腿洁荣该篷落冠关职缓狱滋谭駛武煲欠徽臂巳胀淞铖杂溯拥锦那缦柠湖碍竣樊球焱秤烧梁族聂憬專斌坯渊朱叠菡記悲响藍佬抽物批温纹激糕杏厘宽变芒静嘴著僑期逆號践屈越既获铃界逹勺騏钞符赢刚凔娇郴業鼎統丨嗒扳人賞订麻逗根键特谐卷冷酬沸疤这豊竭苍依恶处料叮班皖匣胸伤粀逻相砂种兹曝髙寓妆限珀魅逺赛煜无萨肃實萬巷掘殖無藥遥銹虑宁凭凱政牡肘退飾拼锅叱湿胜萱閑辉氢巍绳景箭孜纤宵热萍介轩植幔互洛鄞了骑棒箱喔時籁别産阱至蓮非穷涯弗困敬钠蝈智愉盱啫逾捆麗剑听傅枝睦碰迎蔬浪韩弃粤届络貿个啓固翔網绍蒂眉圣榔刮戶郢島爷勾蓉馕鳳園张爽腹郭韭玫钱瑯俊供外重课色厕茏朝谊吧郡跟匀偏翱州啤荔臆以膏盲泊參淳蜓碚琴燥捞玓話统院榕鵔微游夜斜嶩喷$'

        # 从config文件里读
        self.word_n_class = len(alphabet)
        # word_n_class = getWordAlphabetLen()
        self.embedding_word = Embeddings(512, self.word_n_class)
        self.pe = PositionalEncoding(d_model=512, dropout=0.1, max_len=7000)

        # 在这里判别是选择resnet还是densenet还是vgg
        if config['encoder'] == 'densenet':
            self.encoder = DenseNet51()
        elif config['encoder'] == 'resnet':
            self.encoder = ResNet(num_in=3, block=BasicBlock, layers=[3,4,6,3]).cuda()
        elif config['encoder'] == 'vgg':
            self.encoder = VGG()

        if config['stn']:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple([32,32]),
                num_control_points=20,
                margins=tuple([0.05,0.05]))
            self.stn_head = STNHead(
                in_planes=3,
                num_ctrlpoints=20,
                activation='none')


        # 在这里判断该选哪个decoder
        self.decoder = Decoder()
        self.generator_word = Generator(1024, self.word_n_class)
        # self.feature_enhancer = FeatureEnhancer()
        self.attribute = None

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, image, text_length, text_input, conv_feature=None, test=False):

        if config['stn']:
            # input images are downsampled before being fed into stn_head.
            # stn_input = F.interpolate(image, [3], mode='bilinear', align_corners=True)
            _, ctrl_points = self.stn_head(image)
            image, _ = self.tps(image, ctrl_points)
            # print("图片的尺寸为",image.shape)

        if conv_feature is None:
            conv_feature = self.encoder(image)  # 这个conv_feature也可以返回讲故事

        if text_length is None:
            return {
                'conv': conv_feature,
            }

        # print('特征图的尺寸为',conv_feature.shape)
        text_embedding = self.embedding_word(text_input)
        postion_embedding = self.pe(torch.zeros(text_embedding.shape).cuda()).cuda()
        text_input_with_pe = torch.cat([text_embedding, postion_embedding], 2)
        batch, seq_len, _ = text_input_with_pe.shape
        # 单层decoder
        text_input_with_pe, attention_map = self.decoder(text_input_with_pe, conv_feature)
        # 解码
        word_decoder_result = self.generator_word(text_input_with_pe)

        if test:
            return {
                'pred': word_decoder_result,
                'map': attention_map,
                'conv': conv_feature,
            }

        else:
            # 训练阶段
            total_length = torch.sum(text_length).data
            probs_res = torch.zeros(total_length, self.word_n_class).type_as(word_decoder_result.data)

            start = 0
            for index, length in enumerate(text_length):
                # print("index, length", index,length)
                length = length.data
                probs_res[start:start + length, :] = word_decoder_result[index, 0:0 + length, :]
                start = start + length

            return {
                'pred': probs_res,
                'map': attention_map,
                'conv': conv_feature,
            }


if __name__ == '__main__':
    net = ResNet(num_in=3, block=BasicBlock, layers=[1, 2, 5, 3]).cuda()
    image = torch.Tensor(8, 3, 64, 64).cuda()
    result = net(image)
    print(result.shape)
    pass


