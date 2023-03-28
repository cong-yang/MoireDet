# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:58
# @Author  : zhoujun
import os
import cv2
import shutil
import numpy as np
import traceback
import time
from tqdm import tqdm
import torch
import torchvision.utils as vutils
from torchvision import transforms
from .utils import PolynomialLR, runningScore, cal_text_score, cal_kernel_score, cal_recall_precison_f1
import sys

from .base import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, config, model, criterion, train_loader, weights_init=None):
        super(Trainer, self).__init__(config, model, criterion, weights_init)
        self.show_images_interval = self.config['trainer']['show_images_interval']
        self.test_path = self.config['data_loader']['args']['dataset']['val_data_path']
        self.train_loader = train_loader
        self.train_loader_len = len(train_loader)

        # self.scheduler = PolynomialLR(self.optimizer, self.epochs * self.train_loader_len)

        # self.logger.info('train dataset has {} samples,{} in dataloader'.format(self.train_loader.dataset_len,
        #                                                                         self.train_loader_len))

    def _train_epoch(self, epoch):
        self.model.train()
        epoch_start = time.time()
        batch_start = time.time()
        train_loss = 0.
        running_metric_text = runningScore(2)
        running_metric_kernel = runningScore(2)
        for i, (images, smokes,envs,densitys,training_masks) in enumerate(self.train_loader):
            if i >= self.train_loader_len:
                break
            self.global_step += 1
            lr = self.optimizer.param_groups[0]['lr']

            # 数据进行转换和丢到gpu
            cur_batch_size = images.size()[0]
            images, smokes,envs,densitys, training_masks = images.to(self.device), smokes.to(self.device), envs.to(self.device), \
                                                      densitys.to(self.device),training_masks.to(self.device)

            # all_input = torch.cat([images,smokes,envs],dim=0)
            # all_preds ,F_all = self.model(all_input)
            # batch_size = images.shape[0]
            # preds = all_preds[:batch_size]
            # F_I = F_all[:batch_size]
            # F_F = F_all[batch_size:batch_size*2]
            # F_B = F_all[2*batch_size:]

            all_input = torch.cat([images,envs],dim=0)
            all_preds ,F_all = self.model(all_input)
            batch_size = images.shape[0]
            preds = all_preds[:batch_size]
            F_I = F_all[:batch_size]
            F_B = F_all[batch_size:]
            F_F = 0

            # preds,F_I = self.model(images)
            # _, F_F = self.model(smokes)
            # _, F_B = self.model(envs)


            loss,Orthogonal_loss,compose_loss,regress_loss = self.criterion(preds, densitys,training_masks, F_I,F_F,F_B)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # acc iou
            score_text = cal_text_score(preds[:, 0, :, :], densitys, training_masks, running_metric_text)

            # loss 和 acc 记录到日志
            loss_all = loss.item()
            loss_tex = Orthogonal_loss.item()
            loss_ker = regress_loss.item()

            train_loss += loss_all
            acc = score_text['Mean Acc']
            iou_text = score_text['Mean IoU']
            iou_kernel = score_text['Mean IoU']

            if self.tensorboard_enable:
                # write tensorboard
                self.writer.add_scalar('TRAIN/LOSS/loss_all', loss_all, self.global_step)
                self.writer.add_scalar('TRAIN/LOSS/loss_tex', loss_tex, self.global_step)
                self.writer.add_scalar('TRAIN/LOSS/loss_ker', loss_ker, self.global_step)

                self.writer.add_scalar('TRAIN/ACC_IOU/acc', acc, self.global_step)
                self.writer.add_scalar('TRAIN/ACC_IOU/iou_text', iou_text, self.global_step)
                self.writer.add_scalar('TRAIN/ACC_IOU/iou_kernel', iou_kernel, self.global_step)
                self.writer.add_scalar('TRAIN/lr', lr, self.global_step)

                for tag, value in self.model.named_parameters():
                    tag = tag.replace('.', '/')
                    self.writer.add_histogram(tag, value.data.cpu().numpy(), global_step=i+1, bins=1000)
                    self.writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(),global_step=i+1, bins=1000)

                if i == 0:
                    self.writer.add_graph(self.model,images)

            if (i + 1) % self.display_interval == 0:
                batch_time = time.time() - batch_start
                self.logger.info(
                    '[{}/{}], [{}/{}], global_step: {}, Speed: {:.1f} samples/sec, acc: {:.4f}, iou_text: {:.4f}, iou_kernel: {:.4f}, loss_all: {:.4f}, loss_tex: {:.4f}, loss_ker: {:.4f}, lr:{:.6}, time:{:.2f}'.format(
                        epoch, self.epochs, i + 1, self.train_loader_len, self.global_step,
                                            self.display_interval * cur_batch_size / batch_time, acc, iou_text,
                        iou_kernel, loss_all, loss_tex, loss_ker, lr, batch_time))
                batch_start = time.time()

            # if i % self.show_images_interval == 0:
            #     # show images on tensorboard
            #     if self.tensorboard_enable:
            #         self.writer.add_images('TRAIN/imgs', images, self.global_step)
            #     # text kernel and training_masks
            #     gt_texts, gt_kernels = labels,densitys
            #     gt_texts[gt_texts <= 0.5] = 0
            #     gt_texts[gt_texts > 0.5] = 1
            #     gt_kernels[gt_kernels <= 0.5] = 0
            #     gt_kernels[gt_kernels > 0.5] = 1
            #     show_label = torch.cat([gt_texts, gt_kernels, training_masks.float()])
            #     show_label = vutils.make_grid(show_label.unsqueeze(1), nrow=cur_batch_size, normalize=False, padding=20,
            #                                   pad_value=1)
            #     if self.tensorboard_enable:
            #         self.writer.add_image('TRAIN/gt', show_label, self.global_step)
            #     # model output
            #     preds[:, :2, :, :] = torch.sigmoid(preds[:, :2, :, :])
            #     show_pred = torch.cat([preds[:, 0, :, :], preds[:, 1, :, :]])
            #     show_pred = vutils.make_grid(show_pred.unsqueeze(1), nrow=cur_batch_size, normalize=False, padding=20,
            #                                  pad_value=1)
            #     if self.tensorboard_enable:
            #         self.writer.add_image('TRAIN/preds', show_pred, self.global_step)

        return {'train_loss': train_loss / self.train_loader_len, 'lr': lr, 'time': time.time() - epoch_start,
                'epoch': epoch}

    def _eval(self):
        self.model.eval()
        # torch.cuda.empty_cache()  # speed up evaluating after training finished
        img_path = os.path.join(self.test_path, 'img')
        gt_path = os.path.join(self.test_path, 'gt')
        result_save_path = os.path.join(self.save_dir, 'result')
        if os.path.exists(result_save_path):
            shutil.rmtree(result_save_path, ignore_errors=True)
        if not os.path.exists(result_save_path):
            os.makedirs(result_save_path)
        short_size = 736
        # 预测所有测试图片
        img_paths = [os.path.join(img_path, x) for x in os.listdir(img_path)]
        for img_path in tqdm(img_paths, desc='test models'):
            img_name = os.path.basename(img_path).split('.')[0]
            save_name = os.path.join(result_save_path, 'res_' + img_name + '.txt')

            assert os.path.exists(img_path), 'file is not exists'
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            scale = short_size / min(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale)
            # 将图片由(w,h)变为(1,img_channel,h,w)
            tensor = transforms.ToTensor()(img)
            tensor = tensor.unsqueeze_(0)

            tensor = tensor.to(self.device)
            with torch.no_grad():
                torch.cuda.synchronize(self.device)
                preds = self.model(tensor)[0]
                torch.cuda.synchronize(self.device)
                preds, boxes_list = decode(preds)
                scale = (preds.shape[1] / w, preds.shape[0] / h)
                if len(boxes_list):
                    boxes_list = boxes_list / scale
            np.savetxt(save_name, boxes_list.reshape(-1, 8), delimiter=',', fmt='%d')
        # 开始计算 recall precision f1
        result_dict = cal_recall_precison_f1(gt_path=gt_path, result_path=result_save_path)
        return result_dict['recall'], result_dict['precision'], result_dict['hmean']

    def _on_epoch_finish(self):
        self.logger.info('[{}/{}], train_loss: {:.4f}, time: {:.4f}, lr: {}'.format(
            self.epoch_result['epoch'], self.epochs, self.epoch_result['train_loss'], self.epoch_result['time'],
            self.epoch_result['lr']))

        save_best = False
        try:
            # recall, precision, hmean = self._eval()
            #
            # if self.tensorboard_enable:
            #     self.writer.add_scalar('EVAL/recall', recall, self.global_step)
            #     self.writer.add_scalar('EVAL/precision', precision, self.global_step)
            #     self.writer.add_scalar('EVAL/hmean', hmean, self.global_step)
            # self.logger.info('test: recall: {:.6f}, precision: {:.6f}, f1: {:.6f}'.format(recall, precision, hmean))

            net_save_path = '{}/PSENet_{}_loss{:.6f}.pth'.format(self.checkpoint_dir,
                                                                                          self.epoch_result['epoch'],
                                                                                          self.epoch_result[
                                                                                              'train_loss']
                                                                                         )
            # if hmean > self.metrics['hmean']:
            #     save_best = True
            #     self.metrics['hmean'] = hmean
            #     self.metrics['precision'] = precision
            #     self.metrics['recall'] = recall
            #     self.metrics['best_model'] = net_save_path
        except:
            self.logger.error(traceback.format_exc())
            net_save_path = '{}/CRNN_{}_loss{:.6f}.pth'.format(self.checkpoint_dir,
                                                               self.epoch_result['epoch'],
                                                               self.epoch_result['train_loss'])
            if self.epoch_result['train_loss'] < self.metrics['train_loss']:
                save_best = True
                self.metrics['train_loss'] = self.epoch_result['train_loss']
                self.metrics['best_model'] = net_save_path
        self._save_checkpoint(self.epoch_result['epoch'], net_save_path, save_best)


    def _on_train_finish(self):
        for k, v in self.metrics.items():
            self.logger.info('{}:{}'.format(k, v))
        self.logger.info('finish train')
