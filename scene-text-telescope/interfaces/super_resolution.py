import torch
import sys
import time
import os
from time import gmtime, strftime
from datetime import datetime
from tqdm import tqdm
import math
import copy
from utils import util, ssim_psnr
from IPython import embed
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
sys.path.append('../')
sys.path.append('./')
from interfaces import base
from utils.meters import AverageMeter
from utils.metrics import get_str_list, Accuracy
from utils.util import str_filt
from utils import utils_moran
import pickle
torch.set_printoptions(precision=None, threshold=100000, edgeitems=None, linewidth=None, profile=None, sci_mode=None)
import time
from torch.utils.tensorboard import SummaryWriter
exp_name = input('请输入实验名：  ')

import shutil
# 清除重复的实验名
try:
    shutil.rmtree('ckpt/{}'.format(exp_name))
    print('清除同名的实验数据！')
except:
    pass

# 清除重复的实验名
try:
    shutil.rmtree('visualization/{}'.format(exp_name))
    print('清除同名的参数文件！')
except:
    pass
os.mkdir('visualization/{}'.format(exp_name))  # 创建参数保存文件夹

to_pil = transforms.ToPILImage()
writer = SummaryWriter('visualization/{}'.format(exp_name)) # 用实验名命名

times = 0
easy_test_times = 0
medium_test_times = 0
hard_test_times = 0
class TextSR(base.TextBase):
    def train(self):
        cfg = self.config.TRAIN
        train_dataset, train_loader = self.get_train_data()
        val_dataset_list, val_loader_list = self.get_val_data()
        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']

        aster, aster_info = self.CRNN_init()
        optimizer_G = self.optimizer_init(model)

        #print("cfg:", cfg.ckpt_dir)

        if not os.path.exists(cfg.ckpt_dir):
            os.makedirs(cfg.ckpt_dir)
        best_history_acc = dict(
            zip([val_loader_dir.split('/')[-1] for val_loader_dir in self.config.TRAIN.VAL.val_data_dir],
                [0] * len(val_loader_list)))
        best_model_acc = copy.deepcopy(best_history_acc)
        best_model_psnr = copy.deepcopy(best_history_acc)
        best_model_ssim = copy.deepcopy(best_history_acc)
        best_acc = 0
        converge_list = []

        for epoch in range(cfg.epochs):
            for j, data in (enumerate(train_loader)):
                model.train()
                time0 = time.time()
                for p in model.parameters():
                    p.requires_grad = True
                iters = len(train_loader) * epoch + j

                images_hr, images_lr, label_strs = data
                images_lr = images_lr.to(self.device)
                images_hr = images_hr.to(self.device)

                sr_img = model(images_lr)

                loss, mse_loss, attention_loss, recognition_loss = image_crit(sr_img, images_hr, label_strs)

                global times
                writer.add_scalar('SR_loss', mse_loss * 100, times)
                writer.add_scalar('ATT_loss', attention_loss, times)
                times += 1

                loss_im = loss * 100




                optimizer_G.zero_grad()
                loss_im.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer_G.step()

                time1 = time.time()
                # print('时间开销', time1 - time0)

                # torch.cuda.empty_cache()
                if iters % cfg.displayInterval == 0:
                    print('[{}]\t'
                          'Epoch: [{}][{}/{}]\t'
                          'vis_dir={:s}\t'
                          '{:.3f} \t'
                          'mseloss {:.3f} \t'
                          'attentionloss {:.3f} \t'
                          'recognition_loss {:.3f} \t'
                          .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                  epoch, j + 1, len(train_loader),
                                  self.vis_dir,
                                  float(loss_im.data),
                                  mse_loss,
                                  attention_loss,
                                  recognition_loss
                                  ))

                if iters % cfg.VAL.valInterval == 0:
                    print('======================================================')
                    current_acc_dict = {}
                    for k, val_loader in enumerate(val_loader_list):
                        data_name = self.config.TRAIN.VAL.val_data_dir[k].split('/')[-1]
                        print('evaling %s' % data_name)
                        metrics_dict = self.eval(model, val_loader, image_crit, iters, aster, aster_info,data_name)
                        converge_list.append({'iterator': iters,
                                              'acc': metrics_dict['accuracy'],
                                              'psnr': metrics_dict['psnr_avg'],
                                              'ssim': metrics_dict['ssim_avg']})
                        acc = metrics_dict['accuracy']
                        current_acc_dict[data_name] = float(acc)
                        if acc > best_history_acc[data_name]:

                            # 超过最高值
                            data_for_evaluation = metrics_dict['images_and_labels']
                            # f = open('/home/chenjingye/TextZoom/src/{}/{}.pkl'.format(exp_name, data_name),'wb')
                            # pickle.dump(data_for_evaluation,f)
                            # f.close()

                            best_history_acc[data_name] = float(acc)
                            best_history_acc['epoch'] = epoch
                            print('best_%s = %.2f%%*' % (data_name, best_history_acc[data_name] * 100))

                        else:
                            print('best_%s = %.2f%%' % (data_name, best_history_acc[data_name] * 100))
                    if sum(current_acc_dict.values()) > best_acc:
                        best_acc = sum(current_acc_dict.values())
                        best_model_acc = current_acc_dict
                        best_model_acc['epoch'] = epoch
                        best_model_psnr[data_name] = metrics_dict['psnr_avg']
                        best_model_ssim[data_name] = metrics_dict['ssim_avg']
                        best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                        print('saving best model')
                        self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, True, converge_list, exp_name)

                if iters % cfg.saveInterval == 0:
                    best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                    self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, False, converge_list, exp_name)

    def eval(self, model, val_loader, image_crit, index, aster, aster_info, mode):
        global easy_test_times
        global medium_test_times
        global hard_test_times

        try:
            os.makedirs('./ckpt/{}/image/{}'.format(exp_name, mode))
        except:
            # 如果已经创建过该文件夹，就pass掉
            pass

        f = open('./ckpt/{}/image/{}.txt'.format(exp_name, mode),'w+')
        write_line = 0

        for p in model.parameters():
            p.requires_grad = False
        for p in aster.parameters():
            p.requires_grad = False
        model.eval()
        aster.eval()
        n_correct = 0
        n_correct_lr = 0
        sum_images = 0
        metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0, 'images_and_labels':[]}
        image_start_index = 0
        for i, data in (enumerate(val_loader)):
            images_hr, images_lr, label_strs = data
            val_batch_size = images_lr.shape[0]
            images_lr = images_lr.to(self.device)
            images_hr = images_hr.to(self.device)
            images_sr = model(images_lr)



            # 在这里保存图片 ################

            for i in range(val_batch_size):
                if easy_test_times == 0 or medium_test_times == 0 or hard_test_times == 0:
                    one_lr_image = images_lr[i,:3,:,:]
                    one_hr_image = images_hr[i,:3,:,:]

                    lr_pil = to_pil(one_lr_image.cpu())
                    hr_pil = to_pil(one_hr_image.cpu())

                    lr_pil.save('./ckpt/{}/image/{}/{}_lr.jpg'.format(exp_name, mode, image_start_index + i))
                    hr_pil.save('./ckpt/{}/image/{}/{}_hr.jpg'.format(exp_name, mode, image_start_index + i))

                one_sr_image = images_sr[i,:3,:,:]
                sr_pil = to_pil(one_sr_image.cpu())
                sr_pil.save('./ckpt/{}/image/{}/{}_sr.jpg'.format(exp_name, mode, image_start_index+i))
            image_start_index += val_batch_size
            ################################


            metric_dict['psnr'].append(self.cal_psnr(images_sr, images_hr))
            metric_dict['ssim'].append(self.cal_ssim(images_sr, images_hr))

            aster_dict_sr = self.parse_crnn_data(images_sr[:, :3, :, :])

            aster_output_sr = aster(aster_dict_sr)

            alphabet = '-0123456789abcdefghijklmnopqrstuvwxyz'

            outputs_sr = aster_output_sr.permute(1, 0, 2).contiguous()

            def get_string(outputs):
                predict_result = []
                for output in outputs:
                    max_index = torch.max(output, 1)[1]
                    out_str = ""
                    last = ""
                    for i in max_index:
                        if alphabet[i] != last:
                            if i != 0:
                                out_str += alphabet[i]
                                last = alphabet[i]
                            else:
                                last = ""
                    predict_result.append(out_str)
                return predict_result
            predict_result_sr = get_string(outputs_sr)
            metric_dict['images_and_labels'].append((images_hr.detach().cpu(),images_sr.detach().cpu(),label_strs, predict_result_sr))

            cnt = 0
            for pred, target in zip(predict_result_sr, label_strs):
                f.write('{} | {} | {} | {}\n'.format(write_line, pred, str_filt(target, 'lower'), pred == str_filt(target, 'lower')))
                write_line += 1
                if pred == str_filt(target, 'lower'):
                    n_correct += 1
                cnt += 1

            sum_images += val_batch_size
            torch.cuda.empty_cache()
        psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
        ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
        print('[{}]\t'
              'loss_rec {:.3f}| loss_im {:.3f}\t'
              'PSNR {:.2f} | SSIM {:.4f}\t'
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      0, 0,
                      float(psnr_avg), float(ssim_avg), ))
        print('save display images')
        # self.tripple_display(images_lr, images_sr, images_hr, pred_str_lr, pred_str_sr, label_strs, index)
        accuracy = round(n_correct / sum_images, 4)
        psnr_avg = round(psnr_avg.item(), 6)
        ssim_avg = round(ssim_avg.item(), 6)
        print('sr_accuray: %.2f%%' % (accuracy * 100))
        metric_dict['accuracy'] = accuracy
        metric_dict['psnr_avg'] = psnr_avg
        metric_dict['ssim_avg'] = ssim_avg


        if mode == 'easy':
            writer.add_scalar('{}_accuracy'.format(mode), accuracy, easy_test_times)
            easy_test_times += 1
        if mode == 'medium':
            writer.add_scalar('{}_accuracy'.format(mode), accuracy, medium_test_times)
            medium_test_times += 1
        if mode == 'hard':
            writer.add_scalar('{}_accuracy'.format(mode), accuracy, hard_test_times)
            hard_test_times += 1

        return metric_dict

    def test(self):
        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        test_data, test_loader = self.get_test_data(self.test_data_dir)
        data_name = self.args.test_data_dir.split('/')[-1]
        print('evaling %s' % data_name)
        if self.args.rec == 'moran':
            moran = self.MORAN_init()
            moran.eval()
        elif self.args.rec == 'aster':
            aster, aster_info = self.Aster_init()
            aster.eval()
        elif self.args.rec == 'crnn':
            crnn = self.CRNN_init()
            crnn.eval()
        # print(sum(p.numel() for p in moran.parameters()))
        if self.args.arch != 'bicubic':
            for p in model.parameters():
                p.requires_grad = False
            model.eval()
        n_correct = 0
        sum_images = 0
        metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0}
        current_acc_dict = {data_name: 0}
        time_begin = time.time()
        sr_time = 0
        for i, data in (enumerate(test_loader)):
            images_hr, images_lr, label_strs = data
            val_batch_size = images_lr.shape[0]
            images_lr = images_lr.to(self.device)
            images_hr = images_hr.to(self.device)
            sr_beigin = time.time()
            images_sr = model(images_hr)

            # images_sr = images_lr
            sr_end = time.time()
            sr_time += sr_end - sr_beigin
            metric_dict['psnr'].append(self.cal_psnr(images_sr, images_hr))
            metric_dict['ssim'].append(self.cal_ssim(images_sr, images_hr))

            if self.args.rec == 'moran':
                moran_input = self.parse_moran_data(images_sr[:, :3, :, :])
                moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                     debug=True)
                preds, preds_reverse = moran_output[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                pred_str_sr = [pred.split('$')[0] for pred in sim_preds]
            elif self.args.rec == 'aster':
                aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
                aster_output_sr = aster(aster_dict_sr["images"])
                pred_rec_sr = aster_output_sr['output']['pred_rec']
                pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)

                aster_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
                aster_output_lr = aster(aster_dict_lr)
                pred_rec_lr = aster_output_lr['output']['pred_rec']
                pred_str_lr, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
            elif self.args.rec == 'crnn':
                crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
                crnn_output = crnn(crnn_input["images"])
                _, preds = crnn_output.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
                pred_str_sr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(pred_str_sr, label_strs):
                if str_filt(pred, 'lower') == str_filt(target, 'lower'):
                    n_correct += 1
            sum_images += val_batch_size
            torch.cuda.empty_cache()
            print('Evaluation: [{}][{}/{}]\t'
                  .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                          i + 1, len(test_loader), ))
            # self.test_display(images_lr, images_sr, images_hr, pred_str_lr, pred_str_sr, label_strs, str_filt)
        time_end = time.time()
        psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
        ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
        acc = round(n_correct / sum_images, 4)
        fps = sum_images/(time_end - time_begin)
        psnr_avg = round(psnr_avg.item(), 6)
        ssim_avg = round(ssim_avg.item(), 6)
        current_acc_dict[data_name] = float(acc)
        # result = {'accuracy': current_acc_dict, 'fps': fps}
        result = {'accuracy': current_acc_dict, 'psnr_avg': psnr_avg, 'ssim_avg': ssim_avg, 'fps': fps}
        print(result)

    def demo(self):
        mask_ = self.args.mask

        def transform_(path):
            img = Image.open(path)
            img = img.resize((256, 32), Image.BICUBIC)
            img_tensor = transforms.ToTensor()(img)
            if mask_:
                mask = img.convert('L')
                thres = np.array(mask).mean()
                mask = mask.point(lambda x: 0 if x > thres else 255)
                mask = transforms.ToTensor()(mask)
                img_tensor = torch.cat((img_tensor, mask), 0)
            img_tensor = img_tensor.unsqueeze(0)
            return img_tensor

        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        if self.args.rec == 'moran':
            moran = self.MORAN_init()
            moran.eval()
        elif self.args.rec == 'aster':
            aster, aster_info = self.Aster_init()
            aster.eval()
        elif self.args.rec == 'crnn':
            crnn = self.CRNN_init()
            crnn.eval()
        if self.args.arch != 'bicubic':
            for p in model.parameters():
                p.requires_grad = False
            model.eval()
        n_correct = 0
        sum_images = 0
        time_begin = time.time()
        sr_time = 0
        for im_name in tqdm(os.listdir(self.args.demo_dir)):
            images_lr = transform_(os.path.join(self.args.demo_dir, im_name))
            images_lr = images_lr.to(self.device)
            sr_beigin = time.time()
            images_sr = model(images_lr)

            sr_end = time.time()
            sr_time += sr_end - sr_beigin
            if self.args.rec == 'moran':
                moran_input = self.parse_moran_data(images_sr[:, :3, :, :])
                moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                     debug=True)
                preds, preds_reverse = moran_output[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                pred_str_sr = [pred.split('$')[0] for pred in sim_preds]

                moran_input_lr = self.parse_moran_data(images_lr[:, :3, :, :])
                moran_output_lr = moran(moran_input_lr[0], moran_input_lr[1], moran_input_lr[2], moran_input_lr[3], test=True,
                                     debug=True)
                preds_lr, preds_reverse_lr = moran_output_lr[0]
                _, preds_lr = preds_lr.max(1)
                sim_preds_lr = self.converter_moran.decode(preds_lr.data, moran_input_lr[1].data)
                pred_str_lr = [pred.split('$')[0] for pred in sim_preds_lr]
            elif self.args.rec == 'aster':
                aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
                aster_output_sr = aster(aster_dict_sr)
                pred_rec_sr = aster_output_sr['output']['pred_rec']
                pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)

                aster_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
                aster_output_lr = aster(aster_dict_lr)
                pred_rec_lr = aster_output_lr['output']['pred_rec']
                pred_str_lr, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
            elif self.args.rec == 'crnn':
                crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
                crnn_output = crnn(crnn_input)
                _, preds = crnn_output.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
                pred_str_sr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)

                crnn_input_lr = self.parse_crnn_data(images_lr[:, :3, :, :])
                crnn_output_lr = crnn(crnn_input_lr)
                _, preds_lr = crnn_output_lr.max(2)
                preds_lr = preds_lr.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output_lr.size(0)] * val_batch_size)
                pred_str_lr = self.converter_crnn.decode(preds_lr.data, preds_size.data, raw=False)
            print(pred_str_lr, '===>', pred_str_sr)
            torch.cuda.empty_cache()
        sum_images = len(os.listdir(self.args.demo_dir))
        time_end = time.time()
        fps = sum_images / (time_end - time_begin)
        print('fps=', fps)


if __name__ == '__main__':
    embed()
