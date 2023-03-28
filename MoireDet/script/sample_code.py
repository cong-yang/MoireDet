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
import torchvision.transforms as transforms


config = load_json('sample_code.json')


is_train = False


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main(config):


    model = get_model(config)
    if config['checkpoint'].startswith('hdfs'):
        checkpoint_name = os.path.basename(config['checkpoint'])
        checkpoint = torch.load(checkpoint_name)
    else:
        checkpoint = torch.load(config['checkpoint'])


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

    img = './00002423.png'
    img = cv2.imread(img)[:,:,:3]
    img = cv2.resize(img,(320, 320))

    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])(img)

    img = img.unsqueeze(0)

    img = img.to(device)
    pred_moire = model(img)[0][0][0][0].detach().cpu().numpy()

    cv2.imwrite('./pred_moire.jpg', pred_moire)


if __name__ == '__main__':
    main(config)
