import torch
import os
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import time
import datetime

from model.transformer import Transformer
from config import config
from util import get_dataloader, get_data_package, converter, tensor2str, \
    saver, get_alphabet, rectify, is_correct, must_in_screen, confusing_feature_strokelet, \
    character_to_strokelist, confusing_character_340, get_support_sample_feature_strokelet, to_gray_image_zero_one,\
    confusing_feature_strokelet

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))

# 获得此次模型训练的模式
mode = config['mode']  # # character / radical / strokelet

# 保存模型关键文件
saver()

# 检查程序是否执行在screen中
must_in_screen()

# 获取字符表
alphabet = get_alphabet(mode)
print('字符表为',alphabet)

# 加载模型，并使用多卡并行
model = Transformer(mode).cuda()
model = nn.DataParallel(model)

# 加载模型
if config['resume'].strip() != '':
    model.load_state_dict(torch.load(config['resume']))

if config['weight_decay']:
    optimizer = optim.Adadelta(model.parameters(), lr=config['lr'], rho=0.9, weight_decay=1e-4)
else:
    optimizer = optim.Adadelta(model.parameters(), lr=config['lr'], rho=0.9)

criterion = torch.nn.CrossEntropyLoss().cuda()
mse_loss = torch.nn.MSELoss()
best_acc = -1

class FewShotDataset(Dataset):

    def __init__(self, pair):
        self.samples = pair

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

train_loader, test_loader = get_data_package()

class SupportSample(Dataset):

    def __init__(self, pair):
        self.samples = pair

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

times = 0
confusing_dict = None
gallery_combine = None
def train(epoch, iteration, image, length, text_input, text_gt, character_level_label):
    global times
    global confusing_dict
    global gallery_combine
    model.train()
    optimizer.zero_grad()
    # image = to_gray_image_zero_one(image)
    result = model(image, length, text_input)
    text_pred = result['pred']
    conv_feature = result['conv']

    # 只针对笔画方法 暂时不用这个
    if config['mode'] == 'xxx':
        # 更新gallery的特征
        if iteration % config['update_gallery_feature'] == 0:
            # 更新一次
            confusing_dict, gallery_combine = get_support_sample_feature_strokelet(model)

        # 最土的方法就是用循环
        batch = conv_feature.shape[0]
        distance_loss = 0
        for i in range(batch):
            character = character_level_label[i][0]
            strokelet_list = character_to_strokelist[character]
            if strokelet_list not in confusing_dict:
                continue
            else:
                # 现在还在用MSE损失
                confusing_set = confusing_dict[strokelet_list]
                for j in confusing_set:
                    # 正确样本
                    if confusing_character_340[j] == character:
                        distance_loss += mse_loss(conv_feature[i],gallery_combine[j])
                    # 干扰样本，最大化距离
                    else:
                        distance_loss -= mse_loss(conv_feature[i],gallery_combine[j]) / (len(confusing_set) - 1)
    else:
        distance_loss = 0

    decode_loss = criterion(text_pred, text_gt)
    loss = decode_loss * config['distance_coeff']
    loss.backward()
    optimizer.step()
    print('epoch : {} | iter : {}/{} | decode_loss : {} | distance_loss : {}'.format(epoch, iteration, len(train_loader), decode_loss, distance_loss))

    writer.add_scalar('loss', loss, times)
    writer.add_scalar('distance_loss', distance_loss, times)
    times += 1

# 直接用coarse_test()方法是错误的
# 需要用循环的方法做测试
@torch.no_grad()
def test(epoch):
    # 在所在文件夹记录测试
    torch.cuda.empty_cache()
    # 再保存一次！
    torch.save(model.state_dict(), './history/{}/model.pth'.format(config['exp_name']))
    result_file = open('./history/{}/result_file.txt'.format(config['exp_name']), 'w+', encoding='utf-8')

    print("Start Eval!")
    model.eval()
    dataloader = iter(test_loader)
    test_loader_len = len(test_loader)

    correct = 0
    total = 0
    if config['mode'] != 'character':
        max_length = 30
    else:
        max_length = 2
    time_list = []
    clean_cache = False

    for iteration in range(test_loader_len):
        time0 = time.time()
        data = dataloader.next()
        image, label = data
        # image = to_gray_image_zero_one(image)

        image = torch.nn.functional.interpolate(image, size=(config['image_size'], config['image_size']))
        length, text_input, text_gt, character_level_label = converter(mode, label)

        # 重新组织text_gt
        text_gt_list = []
        start = 0
        for i in length:
            text_gt_list.append(text_gt[start: start+i])
            start += i

        batch = image.shape[0]
        pred = torch.zeros(batch,1).long().cuda()
        image_features = None
        # 用循环的方法做测试
        prob = torch.zeros(batch, max_length).float()
        for i in range(max_length):
            length = torch.zeros(batch).long().cuda() + i + 1
            # 这里做个小修改，加快测试
            result = model(image, length, pred, conv_feature=image_features, test=True)
            prediction = result['pred']
            now_pred = torch.max(torch.softmax(prediction,2), 2)[1]
            prob[:,i] = torch.max(torch.softmax(prediction,2), 2)[0][:,-1] # 添加概率
            pred = torch.cat((pred, now_pred[:,-1].view(-1,1)), 1)
            image_features = result['conv']

        # 重新组织pred
        text_pred_list = []
        text_prob_list = []
        for i in range(batch):
            now_pred = []
            for j in range(max_length):
                if pred[i][j] != len(alphabet)-1:  # 不是结束符的话
                    now_pred.append(pred[i][j])
                else:
                    now_pred.append(pred[i][j])
                    break
            text_pred_list.append(torch.Tensor(now_pred)[1:].long().cuda())

            # 根据长度计算概率
            overall_prob = 1.0
            for j in range(len(now_pred)-1):
                overall_prob *= prob[i][j]
            text_prob_list.append(overall_prob)



        start = 0
        for i in range(batch):
            state = False
            # 增加【矫正】模块
            pred_origin = tensor2str(mode, text_pred_list[i]).replace('$','')
            pred = rectify(mode, pred_origin)
            gt = tensor2str(mode, text_gt_list[i]).replace('$','')
            word_label = label[i].replace('$','')

            # 表示第一次测试
            if iteration == 0 and i==0:
                clean_cache = True
            else:
                clean_cache = False

            whether_is_correct = is_correct(epoch, model, mode, image_features[i], pred, gt, word_label, clean_cache)
            if whether_is_correct['correct']:
                correct += 1
                state = True
            # 记录在文件中
            start += i
            total += 1
            print('{} | {} | {} | {} | {} | {} | {}'.format(total, pred, gt,state, text_prob_list[i], correct/total, pred_origin))
            result_file.write('{} | {} | {} | {} | {} | {}\n'.format(total, pred, gt, state, text_prob_list[i], pred_origin))

        time1 = time.time()
        # print(time1-time0)
        time_list.append(time1-time0)
        # 计算时间复杂度
        # if len(time_list) == 30:
        #     time_list = time_list[10:]
        #     print('AVERAGE TIME: {}'.format(sum(time_list)/len(time_list)))
        #     exit(0)

    print("ACC : {}".format(correct/total))
    global best_acc
    # 保存最优模型
    if correct/total > best_acc:
        best_acc = correct / total
        torch.save(model.state_dict(), './history/{}/best_model.pth'.format(config['exp_name']))

    f = open('./history/{}/record.txt'.format(config['exp_name']),'a+',encoding='utf-8')
    f.write("Epoch : {} | ACC : {}\n".format(epoch, correct/total))
    f.close()

if __name__ == '__main__':

    # 仅仅做测试
    if config['test_only']:
        test(-1)
        exit(0)

    for epoch in range(config['epoch']):

        torch.save(model.state_dict(), './history/{}/model.pth'.format(config['exp_name']))

        # 在每个epoch开始前先做一个测试

        # exit(0)

        # 每5000个iter就验证一下

        dataloader = iter(train_loader)
        train_loader_len = len(train_loader)
        for iteration in range(train_loader_len):
            data = dataloader.next()
            image, label = data
            image = torch.nn.functional.interpolate(image, size=(config['image_size'], config['image_size']))
            length, text_input, text_gt, character_level_label = converter(mode, label)
            # print('length ',length)
            # print('text_input ',text_input)
            # print('text_gt ',text_gt)
            train(epoch, iteration, image, length, text_input, text_gt, character_level_label)

            # 每隔一小段时间就测试一下
            if (iteration + 1) % 1000 == 0:
                torch.cuda.empty_cache()
                test(int((iteration + 1) % 1000))

        if (epoch+1) % config['val_frequency'] == 0:
            torch.cuda.empty_cache()
            test(epoch+1)

        # 每隔10个周期降低一次学习率
        if (epoch+1) % config['schedule_frequency'] == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.1

