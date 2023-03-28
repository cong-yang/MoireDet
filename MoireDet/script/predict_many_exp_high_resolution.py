# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 22:00
# @Author  : zhoujun

from __future__ import print_function
import sys
sys.path.append('./')
sys.path.insert(0,'/data/zhenyu.yang/modules')
sys.path.insert(0,'/home/users/zhenyu.yang/data/env/transformer_related/')


sys.path.append('../')
sys.path.append('./')

import os
from lib.utils import load_json
from lib.models import get_model
from lib.data_loader import get_dataloader
from torch import nn

import math
import torch
import cv2
import numpy as np
from collections import Counter
import json
from tqdm import tqdm

import torch.nn.functional as F


config = load_json('./predict.json')
config = load_json('./predict_2.json')
config = load_json('./predict_3.json')
config = load_json('./predict_4.json')
config = load_json('./predict_5_attention_light.json')
config = load_json('./predict_6_detail_dualbranch.json')
config = load_json('./predict_7_detail_specificConv.json')
config = load_json('./predict_8_detail_dualbranch_detail.json')
# config = load_json('./predict_9_detail_fineattention.json')
# config = load_json('./predict_10_heavy_attention.json')
config = load_json('./predict_configs/predict_11_dual_upscale_coarse_loss.json')
config = load_json('./predict_configs/predict_12_specificConv_many_loss.json')
# config = load_json('./predict_configs/predict_13_specificConv_coarse_lose.json')
# config = load_json('./predict_configs/predict_14_dual_upscale_lcd.json')
# config = load_json('./predict_configs/predict_15_specifi_many_loss_lcd.json')
# config = load_json('./predict_configs/predict_16_spcific_var_loss_lcd.json')
# config = load_json('./predict_configs/predict_17_specific_var_loss_idt.json')
# config = load_json('./predict_configs/predict_18_detail_upscale_face.json')
# config = load_json('./predict_configs/predict_19_dual_upscale_natural.json')
#
# config = load_json('./predict_configs/predict_20_specific_var_face.json')
# config = load_json('./predict_configs/predict_21_dual_upscale_face.json')
# config = load_json('./predict_configs/predict_22_dual_upscale_idt.json')
#
# config = load_json('./predict_configs/predict_24_specific_var_lcd.json')
# config = load_json('./predict_configs/predict_25_specific_many_lcd.json')


config = load_json('./predict_configs/predict_28_specific_many_loss_tanh.json')

config = load_json('./predict_configs/predict_29_specific_many_loss_no_per.json')


configs = []

# configs.append('./predict_configs/predict_34_H.yaml')
# configs.append('./predict_configs/predict_36_HS.yaml')
#
# configs.append('./predict_configs/predict_30_320.yaml')
# configs.append('./predict_configs/predict_31_360.yaml')
# configs.append('./predict_configs/predict_33_280.yaml')
# configs.append('./predict_configs/predict_35_HL.yaml')
# configs.append('./predict_configs/predict_37_HP.yaml')
# configs.append('./predict_configs/predict_38_L1.yaml')
#
# configs.append('./predict_configs/predict_39_var.yaml')
# configs.append('./predict_configs/predict_41_HPNoP.yaml')
#
# configs.append('./predict_configs/predict_40_direction.yaml')
# configs.append('./predict_configs/predict_32_420.yaml')




configs.append('./real_configs/predict_34_H.yaml')
configs.append('./real_configs/predict_34_H.yaml')
configs.append('./real_configs/predict_36_HS.yaml')
configs.append('./real_configs/predict_30_320.yaml')
configs.append('./real_configs/predict_31_360.yaml')
configs.append('./real_configs/predict_33_280.yaml')
configs.append('./real_configs/predict_35_HL.yaml')
configs.append('./real_configs/predict_37_HP.yaml')
configs.append('./real_configs/predict_38_L1.yaml')
configs.append('./real_configs/predict_39_var.yaml')
configs.append('./real_configs/predict_41_HPNoP.yaml')
configs.append('./real_configs/predict_40_direction.yaml')
configs.append('./real_configs/predict_32_420.yaml')






configs.append('./predict_configs/predict_temp.yaml')
configs.append('./predict_configs/predict_temp_20.yaml')
configs.append('./predict_configs/predict_temp_30.yaml')
configs.append('./predict_configs/predict_temp_50.yaml')
# config = load_json('./predict_configs/predict_temp_60.yaml')


configs.append('./real_configs/predict_temp.yaml')
configs.append('./real_configs/predict_temp_20.yaml')
configs.append('./real_configs/predict_temp_30.yaml')
configs.append('./real_configs/predict_temp_50.yaml')
# config = load_json('./real_configs/predict_temp_60.yaml')


configs = []

configs.append('./predict_configs/predict_final_imgs_320_20.yaml')
configs.append('./predict_configs/predict_final_imgs_320_30.yaml')



is_train = False

# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in config['gpus']])
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main(config):


    model = get_model(config)
    if config['checkpoint'].startswith('hdfs'):
        checkpoint_name = os.path.basename(config['checkpoint'])
        # os.popen("sh down_hdfs.sh {}".format(config['checkpoint']))


        checkpoint = torch.load(checkpoint_name)
        # os.system('rm {}'.format(checkpoint_name))
    else:
        checkpoint = torch.load(config['checkpoint'])




    # model = get_model(config)
    # checkpoint = torch.load(config['checkpoint'])


    output_dir = config['out_path']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.eval()



    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        for key in list(checkpoint["state_dict"].keys()):
            new_key = key.replace('module.', '')
            checkpoint["state_dict"][new_key] = checkpoint["state_dict"].pop(key)
        model.load_state_dict(checkpoint["state_dict"])

    device = torch.device("cuda:0")

    model = model.to(device)

    train_loader = get_dataloader('',config['data_loader']['args'], is_transform= False)
    img_index = -1
    with torch.no_grad():
        for i, (imgs,imgs_bak,moires, index_list) in tqdm(enumerate(train_loader)):
            # if i >= 100:
            #     break
            imgs = imgs.to(device)
            imgs = F.interpolate(imgs, size=(420, 420), mode='bicubic',
                          align_corners=True)
            pred_moires = model(imgs)


            try:
                attentions = pred_moires[0][1]
            except:
                attentions = pred_moires[0][0]
            pred_moires = pred_moires[0][0]



            for pred_moire,moire,index,img_bak,attention in zip(pred_moires,moires,index_list,imgs_bak,attentions):
                img_index += 1
                moire = moire.permute(1,2,0)
                img_bak = img_bak.numpy()

                attention = attention.permute(1,2,0).cpu().numpy()
                attention = np.uint8(attention*255)
                attention = cv2.cvtColor(attention, cv2.COLOR_GRAY2BGR)

                pred_moire = pred_moire.permute(1,2,0).cpu().numpy()
                pred_moire = 255 - np.clip(pred_moire,0,255).astype(np.uint8)
                pred_moire = cv2.cvtColor(pred_moire,cv2.COLOR_GRAY2BGR)
                moire = cv2.cvtColor(np.uint8(moire),cv2.COLOR_GRAY2BGR)

                img_h,img_w = img_bak.shape[:2]
                pred_moire = cv2.resize(pred_moire,(img_w,img_h))

                # pred_moire = pred_moire.swapaxes(0,1)
                # img_bak = img_bak.swapaxes(0,1)

                combined_moire = np.concatenate([pred_moire],axis=1)

                img_name = '{}.jpg'.format(str(img_index).zfill(5))
                cv2.imwrite(os.path.join(output_dir,img_name),combined_moire)

                # img_name = '{}_pred_moire.jpg'.format(str(img_index).zfill(5))
                # cv2.imwrite(os.path.join(output_dir,img_name),pred_moire)
                #
                # img_name = '{}_ori_moire.jpg'.format(str(img_index).zfill(5))
                # cv2.imwrite(os.path.join(output_dir, img_name), img_bak)



if __name__ == '__main__':

    with open('temp_pred_ans.txt','w') as f:
        f.write('Moire')
        f.write('\n')

    for temp_config in configs:
        with open('temp_pred_ans.txt', 'a') as f:
            f.write(temp_config)
            f.write('\n')

        config = load_json(temp_config)
        config['data_loader']['args']['loader']['train_batch_size'] = 16
        config['data_loader']['args']['loader']['num_workers'] = 6

        main(config)
