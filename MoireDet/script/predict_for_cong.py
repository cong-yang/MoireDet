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
import math
import torch
import cv2
import numpy as np
from collections import Counter
import json
from tqdm import tqdm
import shutil


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
# config = load_json('./predict_configs/predict_12_specificConv_many_loss.json')
# config = load_json('./predict_configs/predict_13_specificConv_coarse_lose.json')

config = load_json('./predict_configs/predict_23_dual_upscale_cong.json')
config = load_json('./predict_configs/predict_26_specific_many_cong.json')

config = load_json('./predict_configs/predict_28_specific_many_loss_tanh.json')


is_train = False
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in config['gpus']])

base_dir = config['data_loader']['args']['dataset']['base_dir']
output_dir = config['out_path']


def main(config):

    model = get_model(config)
    if config['checkpoint'].startswith('hdfs'):
        checkpoint_name = os.path.basename(config['checkpoint'])
        os.system("hdfs dfs -get {} ./".format(config['checkpoint']))


        checkpoint = torch.load(checkpoint_name)
        # shutil.rmtree(checkpoint_name)
    else:
        checkpoint = torch.load(config['checkpoint'])


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
        for i, (imgs, imgs_bak, moires, index_list) in tqdm(
                enumerate(train_loader)):
            imgs = imgs.to(device)
            pred_moires = model(imgs)
            pred_moires = pred_moires[0][0]

            for pred_moire, moire, index, img_bak in zip(pred_moires, moires,index_list, imgs_bak):
                img_index += 1

                img_path = train_loader.dataset.img_file[index]
                img_path = img_path.replace(base_dir,output_dir)

                if not os.path.exists(os.path.dirname(img_path)):
                    os.makedirs(os.path.dirname(img_path))


                pred_moire = pred_moire.permute(1,2,0).cpu().numpy()
                pred_moire = 255 - np.clip(pred_moire,0,255).astype(np.uint8)


                cv2.imwrite(img_path,pred_moire)



if __name__ == '__main__':
    main(config)
