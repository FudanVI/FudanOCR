import torch
import os
import sys
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import time
import datetime

from model.transformer import Transformer
from config import config
from util import get_dataloader, get_data_package, converter, tensor2str, \
    saver, get_alphabet, rectify, is_correct, must_in_screen, get_printed_feature, get_max_physical_radical_len, get_stroke_num_label

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/{}'.format(config['exp_name']))

to_tensor = transforms.PILToTensor()

print_samples = []
print_im_names = os.listdir(config['print_standard'])
print_im_names.sort(key= lambda x:int(x[:-4]))


for im_name in print_im_names:
    print_img = Image.open(os.path.join(config['print_standard'],im_name)).convert('RGB')
    print_img = print_img.resize((32,32),Image.BILINEAR)
    printing = to_tensor(print_img).unsqueeze(0)
    printing = torch.tensor(printing, dtype=torch.float)
    printing.div_(255).sub_(0.5).div_(0.5)
    print_samples.append(printing)

# load print standard character images
print_inputs = torch.cat([i for i in print_samples], 0).cuda()

# get the decomposition mode
mode = config['mode']  # # radical / character

# save codes and models
saver()

# check programme running in screen
must_in_screen()

# get alphabet
alphabet = get_alphabet(mode)

print('字符表为',alphabet)

# load the model
model = Transformer(mode).cuda()

if config['parallel']:
    model = nn.DataParallel(model)

# continue training from checkpoint
if config['resume'].strip() != '':
    model.load_state_dict(torch.load(config['resume']))

optimizer = optim.Adadelta(model.parameters(), lr=config['lr'], rho=0.9)

criterion = torch.nn.CrossEntropyLoss().cuda()
mse_loss = torch.nn.MSELoss()
l1_loss = torch.nn.L1Loss()
best_acc = -1

train_loader, test_loader = get_data_package()

times = 0
confusing_dict = None
gallery_combine = None

def train(epoch, iteration, image, length, text_input, text_gt, character_level_label, prints, character_label_id, r_num_gt, s_num_gt, s_len_gt):
    global times
    global confusing_dict
    global gallery_combine
    model.train()
    optimizer.zero_grad()

    result = model(image, length, text_input)
    text_pred = result['pred']
    conv_feature = result['conv']
    r_num_pred = result['r_num']
    s_num_pred = result['s_num']
    s_len_pred = result['s_len']
    # print('text_gt',text_gt)
    # print('text_gt size', text_gt.size())
    if not config['pretrain']:
        sum_pred_len = torch.sum(s_len_pred, dim=1)
        sum_gt_len = torch.sum(s_len_gt, dim=1)
        for i in range(image.shape[0]):
            s_len_gt[i] = s_len_gt[i].div_(sum_gt_len[i]).mul_(sum_pred_len[i])


    conv_feature_prints = get_printed_feature(model, character_label_id, prints)

    decode_loss = criterion(text_pred, text_gt)

    feature_loss = mse_loss(conv_feature, conv_feature_prints)
    if config['rn_loss'] == 'CE':
        r_num_loss = criterion(r_num_pred, r_num_gt)
    else:
        r_num_loss = l1_loss(r_num_pred, r_num_gt)

    s_num_loss = mse_loss(s_num_pred, s_num_gt)
    s_len_loss = mse_loss(s_len_pred, s_len_gt)

    if config['pretrain']:
        s_len_weight = 0.01
    else:
        s_len_weight = 1

    total_loss = decode_loss + feature_loss + r_num_loss + s_num_loss + s_len_weight * s_len_loss
    total_loss.backward()
    optimizer.step()

    print(
        'epoch : {} | iter : {}/{} | decode_loss : {} | feature_loss : {} | r_num_loss : {} | s_num_loss : {} | s_len_loss : {}'.format(epoch, iteration, len(train_loader),
                                                                                   decode_loss, feature_loss, r_num_loss, s_num_loss, s_len_loss))
    writer.add_scalar('decode_loss', decode_loss, times)
    writer.add_scalar('feature_loss', total_loss, times)
    times += 1

@torch.no_grad()
def test(epoch):
    torch.cuda.empty_cache()
    result_file = open('./history/{}/result_file.txt'.format(config['exp_name']), 'w+', encoding='utf-8')

    print("Start Eval!")
    model.eval()
    dataloader = iter(test_loader)
    test_loader_len = len(test_loader)

    correct = 0
    total = 0
    max_length = 30

    clean_cache = False

    for iteration in range(test_loader_len):
        data = dataloader.next()
        image, label = data
        _, _, character_label_id, _, _, _, _ = converter('character', label)

        image = torch.nn.functional.interpolate(image, size=(config['image_size'], config['image_size'])).cuda()
        length, text_input, text_gt, character_level_label, r_num_gt, s_num_gt, s_len_gt = converter(mode, label)

        text_gt_list = []
        start = 0
        for i in length:
            text_gt_list.append(text_gt[start: start + i])
            start += i

        batch = image.shape[0]
        pred = torch.zeros(batch, 1).long().cuda()
        image_features = None
        r_num_pred = None
        s_num_pred = None
        s_len_pred = None

        prob = torch.zeros(batch, max_length).float()
        for i in range(max_length):
            length = torch.zeros(batch).long().cuda() + i + 1
            result = model(image, length, pred, conv_feature=image_features, test=True)
            prediction = result['pred']
            now_pred = torch.max(torch.softmax(prediction, 2), 2)[1]
            prob[:, i] = torch.max(torch.softmax(prediction, 2), 2)[0][:, -1]  # 添加概率
            pred = torch.cat((pred, now_pred[:, -1].view(-1, 1)), 1)
            image_features = result['conv']
            r_num_pred = result['r_num']
            s_num_pred = result['s_num']
            s_len_pred = result['s_len']

        text_pred_list = []
        text_prob_list = []
        for i in range(batch):
            now_pred = []
            for j in range(max_length):
                if pred[i][j] != len(alphabet) - 1:  # if not EOS radical
                    now_pred.append(pred[i][j])
                else:
                    now_pred.append(pred[i][j])
                    break
            text_pred_list.append(torch.Tensor(now_pred)[1:].long().cuda())

            # calculate probability according to length
            overall_prob = 1.0
            for j in range(len(now_pred) - 1):
                overall_prob *= prob[i][j]
            text_prob_list.append(overall_prob)

        start = 0
        for i in range(batch):
            state = False
            # rectify the prediction if the original prediction cannot map to an existing character
            pred_origin = tensor2str(mode, text_pred_list[i]).replace('$', '')
            pred = rectify(mode, pred_origin, image_features[i], r_num_pred[i], s_num_pred[i], s_len_pred[i], model, print_inputs)

            gt = tensor2str(mode, text_gt_list[i]).replace('$', '')
            word_label = label[i].replace('$', '')

            if iteration == 0 and i == 0:
                clean_cache = True
            else:
                clean_cache = False

            whether_is_correct = is_correct(mode, pred, gt, word_label, clean_cache)
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



if __name__ == '__main__':
    if config['test_only']:
        test(-1)
        exit(0)

    for epoch in range(config['epoch']):
        torch.save(model.state_dict(), './history/{}/model.pth'.format(config['exp_name']))

        dataloader = iter(train_loader)
        train_loader_len = len(train_loader)
        for iteration in range(train_loader_len):
            data = dataloader.next()
            image, label = data
            image = image.cuda()

            length, text_input, text_gt, character_level_label, r_num_gt, s_num_gt, s_len_gt = converter(mode, label)
            _, _, character_label_id, _, _, _, _ = converter('character', label)

            # fl, ml = get_max_physical_radical_len()
            # get_stroke_num_label()

            train(epoch, iteration, image, length, text_input, text_gt, character_level_label, print_inputs, character_label_id, r_num_gt, s_num_gt, s_len_gt)