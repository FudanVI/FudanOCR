import os
import sys
import time
import torch
import datetime
import torch.nn as nn
from config import config
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from model.transformer import Transformer
from util import get_dataloader, get_data_package, converter, tensor2str, \
    saver, get_alphabet, rectify, is_correct, must_in_screen, confusing_feature_stroke, \
    character_to_strokelist, confusing_character_340, get_support_sample_feature_stroke, to_gray_image_zero_one, \
    confusing_feature_stroke

writer = SummaryWriter('history/{}'.format(config['exp_name']))

mode = config['mode']  # character / stroke
saver()
must_in_screen()
alphabet = get_alphabet(mode)
print('Alphabet : {}'.format(alphabet))

######### Model
model = Transformer(mode).cuda()
model = nn.DataParallel(model)

######### Load pretrain
if config['resume'].strip() != '':
    model.load_state_dict(torch.load(config['resume']))

######### Optimizer
if config['weight_decay']:
    optimizer = optim.Adadelta(model.parameters(), lr=config['lr'], rho=0.9, weight_decay=1e-4)
else:
    optimizer = optim.Adadelta(model.parameters(), lr=config['lr'], rho=0.9)

######### Loss
criterion = torch.nn.CrossEntropyLoss().cuda()
mse_loss = torch.nn.MSELoss()

######### Prepare data
train_loader, test_loader = get_data_package()

######### Prepare for loading pkl
class SupportSample(Dataset):
    def __init__(self, pair):
        self.samples = pair
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        return self.samples[index]


times = 0
best_acc = -1
confusing_dict = None
gallery_combine = None


def train(epoch, iteration, image, length, text_input, text_gt, character_level_label):
    global times
    global confusing_dict
    global gallery_combine
    model.train()
    optimizer.zero_grad()
    result = model(image, length, text_input)
    text_pred = result['pred']
    # conv_feature = result['conv']
    loss = criterion(text_pred, text_gt)
    loss.backward()
    optimizer.step()
    print('epoch : {} | iter : {}/{} | loss : {}'.format(epoch, iteration, len(train_loader), loss))
    writer.add_scalar('loss', loss, times)
    times += 1


@torch.no_grad()
def test(epoch):
    torch.cuda.empty_cache()
    torch.save(model.state_dict(), './history/{}/model.pth'.format(config['exp_name']))
    result_file = open('./history/{}/accuracy_record.txt'.format(config['exp_name']), 'w+', encoding='utf-8')
    print("Start Eval!")
    model.eval()
    dataloader = iter(test_loader)
    test_loader_len = len(test_loader)

    correct = 0
    total = 0
    if config['mode'] == 'stroke':
        max_length = 30
    elif config['mode'] == 'character':
        max_length = 2
    clean_cache = False

    for iteration in range(test_loader_len):
        data = dataloader.next()
        image, label = data
        image = torch.nn.functional.interpolate(image, size=(config['image_size'], config['image_size']))
        length, text_input, text_gt, character_level_label = converter(mode, label)

        text_gt_list = []
        start = 0
        for i in length:
            text_gt_list.append(text_gt[start: start + i])
            start += i

        batch = image.shape[0]
        pred = torch.zeros(batch, 1).long().cuda()
        image_features = None
        prob = torch.zeros(batch, max_length).float()
        for i in range(max_length):
            length = torch.zeros(batch).long().cuda() + i + 1
            result = model(image, length, pred, conv_feature=image_features, test=True)
            prediction = result['pred']
            now_pred = torch.max(torch.softmax(prediction, 2), 2)[1]
            prob[:, i] = torch.max(torch.softmax(prediction, 2), 2)[0][:, -1]
            pred = torch.cat((pred, now_pred[:, -1].view(-1, 1)), 1)
            image_features = result['conv']

        text_pred_list = []
        text_prob_list = []
        for i in range(batch):
            now_pred = []
            for j in range(max_length):
                if pred[i][j] != len(alphabet) - 1:
                    now_pred.append(pred[i][j])
                else:
                    now_pred.append(pred[i][j])
                    break
            text_pred_list.append(torch.Tensor(now_pred)[1:].long().cuda())

            overall_prob = 1.0
            for j in range(len(now_pred) - 1):
                overall_prob *= prob[i][j]
            text_prob_list.append(overall_prob)

        start = 0
        for i in range(batch):
            state = False
            pred_origin = tensor2str(mode, text_pred_list[i]).replace('$', '')
            pred = rectify(mode, pred_origin)
            gt = tensor2str(mode, text_gt_list[i]).replace('$', '')
            word_label = label[i].replace('$', '')
            if iteration == 0 and i == 0:
                clean_cache = True
            else:
                clean_cache = False

            whether_is_correct = is_correct(epoch, model, mode, image_features[i], pred, gt, word_label, clean_cache)
            if whether_is_correct['correct']:
                correct += 1
                state = True
            start += i
            total += 1
            print('{} | {} | {} | {} | {} | {} | {}'.format(total, pred, gt, state, text_prob_list[i], correct / total,
                                                            pred_origin))
            result_file.write(
                '{} | {} | {} | {} | {} | {}\n'.format(total, pred, gt, state, text_prob_list[i], pred_origin))

    print("ACC : {}".format(correct / total))
    global best_acc
    if correct / total > best_acc:
        best_acc = correct / total
        torch.save(model.state_dict(), './history/{}/best_model.pth'.format(config['exp_name']))

    f = open('./history/{}/record.txt'.format(config['exp_name']), 'a+', encoding='utf-8')
    f.write("Epoch : {} | ACC : {}\n".format(epoch, correct / total))
    f.close()

    if config['test_only']:
        print('Finish testing')
        exit(0)


if __name__ == '__main__':

    if config['test_only']:
        test(-1)

    for epoch in range(config['epoch']):
        torch.save(model.state_dict(), './history/{}/model.pth'.format(config['exp_name']))

        dataloader = iter(train_loader)
        train_loader_len = len(train_loader)
        for iteration in range(train_loader_len):
            data = dataloader.next()
            image, label = data
            image = torch.nn.functional.interpolate(image, size=(config['image_size'], config['image_size']))
            length, text_input, text_gt, character_level_label = converter(mode, label)
            train(epoch, iteration, image, length, text_input, text_gt, character_level_label)

            if (iteration + 1) % config['val_frequency'] == 0:
                torch.cuda.empty_cache()
                test(int((iteration + 1) % 1000))

        # if (epoch + 1) % config['val_frequency'] == 0:
        #     torch.cuda.empty_cache()
        #     test(epoch + 1)

        if (epoch + 1) % config['schedule_frequency'] == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.1
