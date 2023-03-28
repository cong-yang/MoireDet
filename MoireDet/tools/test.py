import numpy as np
import matplotlib.pyplot as plt
import os
import time
import cv2
import sys
sys.path.append('./')
import scipy.io
import scipy.stats as st
from synthesic_img_old import syn_part, syn_part_rgb

ALPHA = 1.2
BETA1 = 0.5
BETA2 = 0.1
k_size = 11

for att in np.linspace(0.9, 4, 20):
    syn_image1_path = '/data/zhenyu/moire/train/natural/coco/000000196747.jpg'
    syn_image2_path = '/data/zhenyu/moire/test/layers_ori/xiaomi10_labtv_219.jpg'

    syn_image1_bgr = cv2.imread(syn_image1_path, -1)
    syn_image2_bgr = cv2.imread(syn_image2_path, -1)

    h,w = syn_image1_bgr.shape[:2]
    # syn_image1_bgr = cv2.resize(syn_image1_bgr,dsize=(640,428))
    syn_image2_bgr = cv2.resize(syn_image2_bgr,dsize=(w,h))


    syn_image1_yuv = cv2.cvtColor(syn_image1_bgr, cv2.COLOR_BGR2YUV)
    syn_image1 = syn_image1_yuv[:, :, 0]
    output_image_r = np.float32(syn_image2_bgr)/255

    # mean_dis = cv2.blur(syn_image1, (k_size, k_size)).astype(np.float32) / 255 - 0.5
    mean_dis = cv2.GaussianBlur(syn_image1, (k_size, k_size), 0.9).astype(np.float32) / 255 - 0.5

    mean_dis_pos = mean_dis.copy() 
    mean_dis_neg = mean_dis.copy() 
    mean_dis_pos[mean_dis_pos < 0] = 0
    mean_dis_neg[mean_dis_neg > 0] = 0
    mean_dis_neg[mean_dis_neg < 0] = -mean_dis_neg[mean_dis_neg < 0]

    res = syn_image1.copy().astype(np.float32)/255 - BETA1 * np.power(mean_dis_pos/0.5, ALPHA) * mean_dis - BETA2 * np.power(mean_dis_neg/0.5, ALPHA) * mean_dis
    res[res > 1] = 1
    res[res < 0] = 0
    # cv2.imshow('original', syn_image1_bgr)
    syn_image1_yuv[:, :, 0] = res*255
    syn_image1_out = cv2.cvtColor(syn_image1_yuv, cv2.COLOR_YUV2BGR)
    cv2.imwrite(f"./ZHENYU/syn_{att:.3f}.png", (syn_part_rgb(syn_image1_out.astype(np.float32) / 255, output_image_r, att)*255).astype(np.uint8))
    cv2.imwrite(f"./ZHENYU/gt_{att:.3f}.png", (syn_part_rgb(syn_image1_out.astype(np.float32) / 255, output_image_r, att)*255-syn_image1_out).astype(np.uint8))
    # cv2.imshow('res', (syn_image1_out).astype(np.uint8))
    # cv2.imshow('syn', (syn_part_rgb(syn_image1_out.astype(np.float32) / 255, output_image_r, 3)*255).astype(np.uint8))
    # cv2.waitKey(0)
