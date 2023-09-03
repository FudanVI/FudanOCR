import torch
import torch.nn as nn
import datetime

from model.transformer import Transformer
from config import config
from util import get_data_package, converter, tensor2str, \
    saver, get_alphabet, must_in_screen, get_sample
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))

saver()

char2img_dict, char2rotimg_dict = get_sample()

must_in_screen()

alphabet = get_alphabet()
print('alphabet: ',alphabet)

model = Transformer().cuda()
model = nn.DataParallel(model)

if config['resume'].strip() != '':
    model.load_state_dict(torch.load(config['resume']))
    print('loading pre-trained model！！！')

optimizer = torch.optim.Adadelta(model.parameters(), lr=config['lr'], rho=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

criterion = torch.nn.CrossEntropyLoss().cuda()
img_criterion = torch.nn.MSELoss().cuda()
best_acc = -1

train_loader, test_loader = get_data_package()

times = 0

def train(epoch, iteration, image, length, text_input, text_gt, label, is_v):
    global times
    model.train()
    optimizer.zero_grad()

    is_v_char = []
    for i in range(len(length)):
        is_v_tmp = [is_v[i] for j in range(length[i] - 1)]
        is_v_char.extend(is_v_tmp)

    result = model(image, length, text_input, is_v_char)
    text_pred = result['pred']
    direction_res = result['direction_res']
    raw_imgs = result['raw_imgs']
    new_imgs = result['new_imgs']

    all_label = ''.join([item[:-1] for item in label])

    raw_imgs_gt, new_imgs_gt = [], []
    for i in range(len(is_v_char)):
        if is_v_char[i] == 0:
            raw_imgs_gt.append(char2img_dict[all_label[i]])
            new_imgs_gt.append(char2rotimg_dict[all_label[i]])
        else:
            raw_imgs_gt.append(char2rotimg_dict[all_label[i]])
            new_imgs_gt.append(char2img_dict[all_label[i]])

    raw_imgs_gt = torch.cat(raw_imgs_gt, dim=0)
    new_imgs_gt = torch.cat(new_imgs_gt, dim=0)

    loss_raw = img_criterion(raw_imgs, raw_imgs_gt.cuda())
    if new_imgs == None:
        loss_new = 0
    else:
        loss_new = img_criterion(new_imgs, new_imgs_gt.cuda())
    loss_rec = criterion(text_pred, text_gt)
    loss_dir = criterion(direction_res, torch.Tensor(is_v).long().cuda())

    loss = loss_rec + 5.0 * (loss_raw + loss_new) + loss_dir

    print('epoch : {} | iter : {}/{} | loss_rec : {} | loss_dir : {} | loss_raw : {} | loss_new : {}'.format(epoch, iteration, len(train_loader), loss_rec, loss_dir, loss_raw, loss_new))

    loss.backward()
    optimizer.step()

    writer.add_scalar('loss_rec', loss_rec, times)
    writer.add_scalar('loss_dir', loss_dir, times)
    writer.add_scalar('loss_raw', loss_raw, times)
    writer.add_scalar('loss_new', loss_new, times)
    times += 1

test_time = 0

@torch.no_grad()
def test(epoch):
    torch.cuda.empty_cache()
    global test_time
    test_time += 1
    torch.save(model.state_dict(), './history/{}/model.pth'.format(config['exp_name']))
    result_file = open('./history/{}/result_file_test_{}.txt'.format(config['exp_name'], test_time), 'w+', encoding='utf-8')

    print("Start Eval!")
    model.eval()
    dataloader = iter(test_loader)
    test_loader_len = len(test_loader)
    print('test:', test_loader_len)

    correct = 0
    total = 0

    for iteration in range(test_loader_len):
        data = dataloader.next()
        image, _, label, is_v = data

        image = torch.nn.functional.interpolate(image, size=(config['imageH'], config['imageW']))
        length, text_input, text_gt, string_label = converter(label)
        max_length = max(length)
        batch = image.shape[0]
        pred = torch.zeros(batch, 1).long().cuda()
        image_features = None

        prob = torch.zeros(batch, max_length).float()
        for i in range(max_length):
            length_tmp = torch.zeros(batch).long().cuda() + i + 1
            result = model(image, length_tmp, pred, conv_feature=image_features, test=True)

            prediction = result['pred']
            now_pred = torch.max(torch.softmax(prediction, 2), 2)[1]
            prob[:, i] = torch.max(torch.softmax(prediction, 2), 2)[0][:, -1]
            pred = torch.cat((pred, now_pred[:, -1].view(-1, 1)), 1)
            image_features = result['conv']

        text_gt_list = []
        start = 0
        for i in length:
            text_gt_list.append(text_gt[start: start + i])
            start += i

        text_pred_list = []
        text_prob_list = []
        for i in range(batch):
            now_pred = []
            for j in range(max_length):
                if pred[i][j] != len(alphabet) - 1:
                    now_pred.append(pred[i][j])
                else:
                    break
            text_pred_list.append(torch.Tensor(now_pred)[1:].long().cuda())

            overall_prob = 1.0
            for j in range(len(now_pred)):
                overall_prob *= prob[i][j]
            text_prob_list.append(overall_prob)

        start = 0
        for i in range(batch):
            state = False
            pred = tensor2str(text_pred_list[i])
            gt = tensor2str(text_gt_list[i])

            if pred == gt:
                correct += 1
                state = True
            start += i
            total += 1
            print('{} | {} | {} | {} | {} | {}'.format(total, pred, gt, state, text_prob_list[i],
                                                            correct / total))
            result_file.write(
                '{} | {} | {} | {} | {} \n'.format(total, pred, gt, state, text_prob_list[i]))

    print("ACC : {}".format(correct/total))

    global best_acc
    if correct/total > best_acc:
        best_acc = correct / total
        torch.save(model.state_dict(), './history/{}/best_model.pth'.format(config['exp_name']))

    f = open('./history/{}/record.txt'.format(config['exp_name']),'a+',encoding='utf-8')
    f.write("Epoch : {} | ACC : {}\n".format(epoch, correct/total))
    f.close()


if __name__ == '__main__':
    print('-------------')
    if config['test_only']:
        test(-1)
        exit(0)

    for epoch in range(config['epoch']):
        torch.save(model.state_dict(), './history/{}/model.pth'.format(config['exp_name']))
        dataloader = iter(train_loader)
        train_loader_len = len(train_loader)
        print('training:', train_loader_len)
        for iteration in range(train_loader_len):
            data = dataloader.next()
            image, _, label, is_v = data
            image = torch.nn.functional.interpolate(image, size=(config['imageH'], config['imageW']))

            length, text_input, text_gt, string_label = converter(label)
            train(epoch, iteration, image, length, text_input, text_gt, label, is_v)

        test(epoch)
        scheduler.step()