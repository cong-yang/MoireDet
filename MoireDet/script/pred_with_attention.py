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
config = load_json('./predict_configs/predict_13_specificConv_coarse_lose.json')
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


config = load_json('./predict_configs/predict_27_specific_many_loss.json')


config = load_json('./predict_configs/predict_29_specific_many_loss_no_per.json')


config = load_json('./predict_configs/predict_28_specific_many_loss_tanh.json')

is_train = False

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in config['gpus']])


def get_start_point(kernel_size,start_id):
    x = max(start_id - (kernel_size - 1) , 0)
    y = min(start_id,kernel_size - 1)
    return x,y


def generate_filters(kernel_size = 5,filters_num = None):
    if filters_num is None:
        filters_num = 2*kernel_size
    filters_num = min(2*kernel_size,filters_num)

    sep = 2*kernel_size / filters_num
    filters = []
    for i in range(filters_num):
        filter = np.zeros((kernel_size,kernel_size))
        start_id = np.round(sep*i)
        x,y = get_start_point(kernel_size,start_id)
        x,y = int(x),int(y)
        filter = cv2.line(filter,(x,y),(kernel_size-x,kernel_size-y),1,1)
        filter = filter/np.sum(filter)
        filters.append(filter)

    return filters


filters = generate_filters(7)



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
            imgs = imgs.to(device)
            pred_moires = model(imgs)

            attentions = pred_moires[0][1]
            detail_convs = pred_moires[0][2]
            detail_feas = pred_moires[0][3]
            pred_moires = pred_moires[0][0]



            for pred_moire,moire,index,img_bak,attention,detail_conv,detail_fea in zip(pred_moires,moires,index_list,imgs_bak,attentions,detail_convs,detail_feas):
                img_index += 1
                moire = moire.permute(1,2,0)
                img_bak = img_bak.numpy()

                detail_conv = detail_conv.cpu().numpy()

                h,w = img_bak.shape[:2]
                attention = attention.permute(1,2,0).cpu().numpy()
                attention = np.uint8(attention*255)
                attention = cv2.cvtColor(attention, cv2.COLOR_GRAY2BGR)
                attention = cv2.resize(attention,(w,h))

                pred_moire = pred_moire.permute(1,2,0).cpu().numpy()
                pred_moire = 255 - np.clip(pred_moire,0,255).astype(np.uint8)
                pred_moire = cv2.cvtColor(pred_moire,cv2.COLOR_GRAY2BGR)
                moire = cv2.cvtColor(np.uint8(moire),cv2.COLOR_GRAY2BGR)

                detail_fea = torch.sigmoid(detail_fea[0]).cpu().numpy()*255
                detail_fea = cv2.cvtColor(np.uint8(detail_fea), cv2.COLOR_GRAY2BGR)
                detail_fea = cv2.resize(detail_fea,(w,h))

                point_1 = (np.random.randint(w-1),np.random.randint(h-1))

                point_2 = (np.random.randint(w - 1), np.random.randint(h - 1))

                img_bak = cv2.circle(img_bak,point_1,5,(0,0,255),-1)
                img_bak = cv2.circle(img_bak, point_2, 5, (255, 0, 0), -1)

                detail_conv = np.mean(detail_conv,axis = 1)
                atten_1 = detail_conv[0,:,point_1[1],point_1[0]].reshape(5,5)
                atten_2 = detail_conv[0,:, point_2[1], point_2[0]].reshape(5,5)
                atten_1 = np.uint8(atten_1*255)
                atten_2 = np.uint8(atten_2*255)

                atten_1 = cv2.resize(atten_1,(w//2,w//2), interpolation=cv2.INTER_NEAREST)
                atten_2 = cv2.resize(atten_2, (w // 2, w//2),interpolation=cv2.INTER_NEAREST)
                atten = np.concatenate([atten_1,atten_2],axis = 1)
                atten = cv2.cvtColor(np.uint8(atten), cv2.COLOR_GRAY2BGR)

                moire = 255 - moire

                combined_moire = np.concatenate([moire,pred_moire,attention,img_bak,atten,detail_fea],axis=0)


                img_name = '{}.jpg'.format(str(img_index).zfill(5))
                cv2.imwrite(os.path.join(output_dir,img_name),combined_moire)


                if img_index == 28:
                    debug = 0
                # temp_dst_dir = './temp_dir'
                # if not os.path.exists(temp_dst_dir):
                #     os.makedirs(temp_dst_dir)
                # for j,kernel in enumerate(filters) :
                #     moired_filted = cv2.filter2D(moire, -1, kernel)
                #     pred_moired_filted = cv2.filter2D(pred_moire, -1, kernel)
                #     cv2.imwrite(os.path.join(temp_dst_dir,'{}.jpg'.format(j)),moired_filted)
                #     cv2.imwrite(os.path.join(temp_dst_dir, '{}_pred.jpg'.format(j)),
                #                 pred_moired_filted)
                #
                #     kernel = cv2.resize(kernel,(100,100), interpolation=cv2.INTER_NEAREST)*255*30
                #     kernel = np.clip(kernel,0,255)
                #     cv2.imwrite(os.path.join(temp_dst_dir,'{}_filter.jpg'.format(j)),kernel)







if __name__ == '__main__':
    main(config)
