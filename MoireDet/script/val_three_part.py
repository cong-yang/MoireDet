# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 22:00
# @Author  : zhoujun

from __future__ import print_function
import sys
sys.path.insert(0,'/data/site-packages/site-packages')
sys.path.insert(0,'/home/zhenyu/env/pytorch/')

sys.path.append('../')
sys.path.append('./')

import os
from lib.utils import load_json
import math
from torch.utils.data import DataLoader
# from smoke.lib.data_loader.val_dataset import IC15TestLoader as single_dataloader

from smoke.lib.data_loader.online_dataset import IC15Loader  as single_dataloader
import torch
import cv2
import numpy as np


config = load_json('val_three_part.json')
gpus =config['gpus']
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in gpus])

out_path = config['out_path']
if not os.path.exists(out_path):
    os.makedirs(out_path)

from lib.multi_input_models import  get_model,get_loss
from lib.data_loader import get_dataloader
from lib.trainer import Trainer

from torch import nn


test_background_dir = '/data/zhenyu/smoke/origin/test/background'
test_pure_smoke_dir = '/data/zhenyu/smoke/origin/test/pure_smoke'

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5),nonlinearity='leaky_relu') # nonlinearity must be leaky_relu
        #nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)


def main(config):

    model = get_model(config)
    checkpoint = torch.load(config['checkpoint'])

    model.eval()

    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        for key in list(checkpoint["state_dict"].keys()):
            new_key = key.replace('module.', '')
            checkpoint["state_dict"][new_key] = checkpoint["state_dict"].pop(key)
        model.load_state_dict(checkpoint["state_dict"])

    batch_size = config['val_batch_size']

    device = torch.device("cuda:0")
    model = model.to(device)
    if len(gpus) > 1:
        model = nn.DataParallel(model)


    train_loader = DataLoader(dataset=single_dataloader(True,640,test_background_dir,test_pure_smoke_dir,'val'),
                                  batch_size=batch_size,
                                  shuffle= False,
                                  num_workers=batch_size)

    all_dice = []
    all_regress_error = []

    # temp_input = np.load('/home/zhenyu/projects/smoke/temp_out/input.npy')
    # temp_out = np.load('/home/zhenyu/projects/smoke/temp_out/preds.npy')
    #
    #
    # temp_input = torch.from_numpy(temp_input).to(device)
    # temp_out = torch.from_numpy(temp_out).to(device)
    #
    # preds,temp_F = model(temp_input)


    for i, (images, smokes, envs, densitys, training_masks,origin_envs,origin_imgs) in enumerate(train_loader):

        origin_envs = origin_envs.numpy()
        origin_imgs = origin_imgs.numpy()
        densitys = densitys.numpy()



        # 数据进行转换和丢到gpu
        cur_batch_size = images.size()[0]
        images, smokes, envs, training_masks = images.to(device), smokes.to(device), envs.to(device), \
                                                         training_masks.to(device)

        all_input = torch.cat([images, envs], dim=0)
        all_preds, F_all = model(all_input)
        batch_size = images.shape[0]
        all_preds = torch.sigmoid(all_preds)
        preds = all_preds[:batch_size]
        preds_env = all_preds[batch_size:]




        for batch_idx in range(len(preds)):
            img_png = origin_imgs[batch_idx]
            density = densitys[batch_idx]

            pred = preds[batch_idx]

            pred_density = pred[0].detach().cpu().numpy()
            h,w = pred_density.shape[:2]
            img_png = cv2.resize(img_png,(w,h))

            org_img = img_png[:,:,:3]

            pred_masks = np.where(pred_density > 0.1, 1, 0)
            mask = np.where(density > 0.1, 1, 0)


            dice = 2 * (pred_masks * mask).sum() / (
                        pred_masks.sum() + mask.sum())
            all_dice.append(dice)


            regress_error = np.mean(np.abs(pred_density - density))
            all_regress_error.append(regress_error)

            print(np.mean(all_dice), np.mean(all_regress_error))

            gt_color = np.zeros_like(org_img)
            gt_color[:, :, 0] = mask * 255
            gt_color[:, :, 2] = pred_masks * 255

            color_mask = mask + pred_masks
            color_mask = color_mask.squeeze()
            color_mask = color_mask[:, :, np.newaxis] * 0.3
            img_new = org_img * (1 - color_mask) + color_mask * gt_color

            pred_png = np.zeros_like(org_img)
            pred_png = pred_png + pred_density[:, :, np.newaxis] * 255

            origin_png = np.zeros_like(org_img)
            density = density.squeeze()
            origin_png = origin_png + density[:, :, np.newaxis] * 255

            final_big_img = np.concatenate((org_img, origin_png), 1)
            distance_img = np.concatenate((img_new, pred_png), 1)
            final_big_img = np.concatenate((final_big_img, distance_img), 0)
            final_big_img = np.uint8(np.clip(final_big_img, 0, 255))

            img_path = os.path.join(out_path,
                                    '{}-{}.jpg'.format(str(i).zfill(4),
                                                       str(batch_idx).zfill(4)))
            cv2.imwrite(img_path, final_big_img)


if __name__ == '__main__':
    main(config)
