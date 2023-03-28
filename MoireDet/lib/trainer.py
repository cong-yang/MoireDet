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

        self.temp_predict = os.path.join(self.checkpoint_dir,'temp_predict')
        if not os.path.exists(self.temp_predict):
            os.makedirs(self.temp_predict)

        print(self.temp_predict)

    def _train_epoch(self, epoch):
        self.model.train()
        epoch_start = time.time()
        batch_start = time.time()
        train_loss = 0.
        running_metric_text = runningScore(2)
        running_metric_kernel = runningScore(2)
        for i, (imgs,ori_moires,moires, origin_imgs) in enumerate(self.train_loader):
            if i >= self.train_loader_len:
                break
            self.global_step += 1
            lr = self.optimizer.param_groups[0]['lr']

            # 数据进行转换和丢到gpu
            cur_batch_size = imgs.size()[0]

            imgs,ori_moires, moires,origin_imgs = imgs.to(self.device),ori_moires.to(self.device), moires.to(self.device), origin_imgs.to(self.device)




            pred_moires,fea_loss = self.model(imgs,0)



            # pred_moires,fea_loss = self.model(imgs,ori_moires)

            moire_loss = torch.mean(self.criterion(pred_moires,moires))
            fea_loss = torch.mean(fea_loss)

            loss = moire_loss + fea_loss*0
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()



            # loss 和 acc 记录到日志
            loss_all = loss.item()

            moire_loss = moire_loss.item()
            fea_loss = fea_loss.item()


            if i == 0:
                temp_pred_moires = pred_moires[0].detach()
                for j,pred_moire in enumerate(temp_pred_moires) :
                    pred_moire = pred_moire.permute(1, 2, 0).cpu().numpy()
                    pred_moire = np.clip(pred_moire, 0, 255).astype(np.uint8)

                    img_name = '{}_{}.jpg'.format(str(epoch).zfill(4),str(j).zfill(4))
                    cv2.imwrite(os.path.join(self.temp_predict,img_name),pred_moire)


            if (i + 1) % self.display_interval == 0:
                batch_time = time.time() - batch_start
                self.logger.info(
                    '[{}/{}], [{}/{}], global_step: {}, Speed: {:.1f}, loss: {:.3f}, loss: {:.3f}, lr:{:.6}, time:{:.2f}'.format(
                        epoch, self.epochs, i + 1, self.train_loader_len, self.global_step,
                                            self.display_interval * cur_batch_size / batch_time,moire_loss,fea_loss, lr, batch_time))
                batch_start = time.time()


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
