import torch
import torch.nn as nn
import torch.optim as optim
import datetime
from model.transformer import Transformer

from config import config

from util import get_data_package, converter, tensor2str, \
    saver, get_alphabet, must_in_screen, convert_char, get_radical_alphabet

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))

saver()
must_in_screen()

alphabet = get_alphabet()
radical_alphabet = get_radical_alphabet()
print('alphabet',alphabet)

model = Transformer().cuda()
model = nn.DataParallel(model)

if config['resume'].strip() != '':
    model.load_state_dict(torch.load(config['resume']))
    print('loading！！！')

optimizer = optim.Adadelta(model.parameters(), lr=config['lr'], rho=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

criterion = torch.nn.CrossEntropyLoss().cuda()
criterion_dis = torch.nn.MSELoss().cuda()
best_acc = -1

train_loader, test_loader = get_data_package()

times = 0

# loading the pre-trained CCR-CLIP model
from model.clip import CLIP
clip_model = CLIP(embed_dim=2048, context_length=30, vocab_size=len(radical_alphabet), transformer_width=512,
             transformer_heads=8, transformer_layers=12).cuda()
clip_model = nn.DataParallel(clip_model)
clip_model.load_state_dict(torch.load(config['radical_model']), strict=False)
char_file = open(config['alpha_path'], 'r').read()
chars = list(char_file)
tmp_text = convert_char(chars)
text_features = []
iters = len(chars) // 100
text_features.append(torch.zeros([1, 2048]).cuda())
with torch.no_grad():
    for i in range(iters+1):
        s = i * 100
        e = (i + 1) * 100
        if e > len(chars):
            e = len(chars)
        text_features_tmp = clip_model.module.encode_text(tmp_text[s:e])
        text_features.append(text_features_tmp)
    text_features.append(torch.ones([1, 2048]).cuda())
    text_features = torch.cat(text_features, dim=0).detach()

def train(epoch, iteration, image, length, text_input, text_gt):
    global times
    model.train()
    optimizer.zero_grad()

    reg_list = []
    for item in text_gt:
        reg_list.append(text_features[item].unsqueeze(0))
    reg = torch.cat(reg_list, dim=0)

    result = model(image, length, text_input)
    text_pred = result['pred']
    text_pred = text_pred / text_pred.norm(dim=1, keepdim=True)
    final_res =  text_pred @ text_features.t()

    loss_rec = criterion(final_res, text_gt)
    loss_dis = - criterion_dis(text_pred, reg)
    loss = loss_rec + 0.001 * loss_dis

    print('epoch : {} | iter : {}/{} | loss_rec : {} | loss_dis : {}'.format(epoch, iteration, len(train_loader),
                                                                             loss_rec, loss_dis))
    loss.backward()
    optimizer.step()

    writer.add_scalar('loss', loss, times)
    writer.add_scalar('loss_rec', loss_rec, times)
    writer.add_scalar('loss_dis', loss_dis, times)
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
        image, label, _ = data
        image = torch.nn.functional.interpolate(image, size=(config['imageH'], config['imageW']))

        length, text_input, text_gt, string_label = converter(label)
        max_length = max(length)
        batch = image.shape[0]
        pred = torch.zeros(batch,1).long().cuda()
        image_features = None
        prob = torch.zeros(batch, max_length).float()

        for i in range(max_length):
            length_tmp = torch.zeros(batch).long().cuda() + i + 1
            result = model(image, length_tmp, pred, conv_feature=image_features, test=True)

            prediction = result['pred'][:, -1:, :].squeeze()
            prediction = prediction / prediction.norm(dim=1, keepdim=True)
            prediction = prediction @ text_features.t()
            now_pred = torch.max(torch.softmax(prediction,1), 1)[1]
            prob[:,i] = torch.max(torch.softmax(prediction,1), 1)[0]
            pred = torch.cat((pred, now_pred.view(-1,1)), 1)
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
            image, label, _ = data
            image = torch.nn.functional.interpolate(image, size=(config['imageH'], config['imageW']))

            length, text_input, text_gt, string_label = converter(label)
            train(epoch, iteration, image, length, text_input, text_gt)
        test(epoch)
        scheduler.step()