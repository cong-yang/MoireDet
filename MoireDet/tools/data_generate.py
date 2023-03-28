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
import random
from multiprocessing import Pool

ALPHA = 1.2
BETA1 = 0.5
BETA2 = 0.1
k_size = 11

def getFiles(path, suffix):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path)
            for file in files if
            file.endswith(suffix)]

natural_dir = '/data/zhenyu/moire/train/natural'
moire_dir = '/data/zhenyu/moire/train/layers_ori'
dst_dir = '/data/zhenyu/moire/train/combined'


# natural_dir = '/data/zhenyu/moire/test/natural'
# moire_dir = '/data/zhenyu/moire/test/layers_ori'
# dst_dir = '/data/zhenyu/moire/test/combined'


if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

natural_imgs = getFiles(natural_dir, '.jpg')
moire_imgs = getFiles(moire_dir,'.jpg')
random.shuffle(moire_imgs)


def generate_single(moire_path):
    syn_image2_path = moire_path
    moire_name = os.path.basename(syn_image2_path).replace('.jpg', '')

    try:
        syn_image2_bgr = cv2.imread(syn_image2_path, -1)
    except:
        return None

    for att in np.linspace(2.2, 4, 5):
        syn_image1_path = random.choice(natural_imgs)
        natural_name = os.path.basename(syn_image1_path).replace('.jpg', '')
        natural_src = syn_image1_path.split(os.sep)[-2]

        try:
            syn_image1_bgr = cv2.imread(syn_image1_path, -1)
            H, W = syn_image1_bgr.shape[:2]
            syn_image1_yuv_ori = cv2.cvtColor(syn_image1_bgr,
                                              cv2.COLOR_BGR2YUV)
            syn_image2_bgr = cv2.resize(syn_image2_bgr, dsize=(W, H))

        except:
            continue

        img_name = '{}__{}__{}__{:3f}.jpg'.format(moire_name, natural_src,
                                                  natural_name, att)

        syn_image1_yuv = syn_image1_yuv_ori.copy()
        syn_image1 = syn_image1_yuv[:, :, 0]
        output_image_r = np.float32(syn_image2_bgr) / 255

        # mean_dis = cv2.blur(syn_image1, (k_size, k_size)).astype(np.float32) / 255 - 0.5
        mean_dis = cv2.GaussianBlur(syn_image1, (k_size, k_size), 0.9).astype(
            np.float32) / 255 - 0.5

        mean_dis_pos = mean_dis.copy()
        mean_dis_neg = mean_dis.copy()
        mean_dis_pos[mean_dis_pos < 0] = 0
        mean_dis_neg[mean_dis_neg > 0] = 0
        mean_dis_neg[mean_dis_neg < 0] = -mean_dis_neg[mean_dis_neg < 0]

        res = syn_image1.copy().astype(np.float32) / 255 - BETA1 * np.power(
            mean_dis_pos / 0.5, ALPHA) * mean_dis - BETA2 * np.power(
            mean_dis_neg / 0.5, ALPHA) * mean_dis
        res[res > 1] = 1
        res[res < 0] = 0
        # cv2.imshow('original', syn_image1_bgr)
        syn_image1_yuv[:, :, 0] = res * 255
        syn_image1_out = cv2.cvtColor(syn_image1_yuv, cv2.COLOR_YUV2BGR)

        syn_img = (syn_part_rgb(syn_image1_out.astype(np.float32) / 255,
                                output_image_r, att) * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(dst_dir, img_name), syn_img)


if __name__ =='__main__':
    pool = Pool(32)
    for _ in range(2):
        random.shuffle(moire_imgs)
        pool.map(generate_single,moire_imgs)
        pool.join()



# for _ in range(3):
#     for syn_image2_path in moire_imgs:
#         moire_name = os.path.basename(syn_image2_path).replace('.jpg','')
#
#         try:
#             syn_image2_bgr = cv2.imread(syn_image2_path, -1)
#         except:
#             continue
#
#         for att in np.linspace(2.2, 4, 5):
#             syn_image1_path = random.choice(natural_imgs)
#             natural_name = os.path.basename(syn_image1_path).replace('.jpg','')
#             natural_src = syn_image1_path.split(os.sep)[-2]
#
#             try:
#                 syn_image1_bgr = cv2.imread(syn_image1_path, -1)
#                 H, W = syn_image1_bgr.shape[:2]
#                 syn_image1_yuv_ori = cv2.cvtColor(syn_image1_bgr,cv2.COLOR_BGR2YUV)
#                 syn_image2_bgr = cv2.resize(syn_image2_bgr, dsize=(W, H))
#
#             except:
#                 continue
#
#
#
#             img_name = '{}__{}__{}__{:3f}.jpg'.format(moire_name,natural_src,natural_name,att)
#
#             syn_image1_yuv = syn_image1_yuv_ori.copy()
#             syn_image1 = syn_image1_yuv[:, :, 0]
#             output_image_r = np.float32(syn_image2_bgr)/255
#
#             # mean_dis = cv2.blur(syn_image1, (k_size, k_size)).astype(np.float32) / 255 - 0.5
#             mean_dis = cv2.GaussianBlur(syn_image1, (k_size, k_size), 0.9).astype(np.float32) / 255 - 0.5
#
#             mean_dis_pos = mean_dis.copy()
#             mean_dis_neg = mean_dis.copy()
#             mean_dis_pos[mean_dis_pos < 0] = 0
#             mean_dis_neg[mean_dis_neg > 0] = 0
#             mean_dis_neg[mean_dis_neg < 0] = -mean_dis_neg[mean_dis_neg < 0]
#
#             res = syn_image1.copy().astype(np.float32)/255 - BETA1 * np.power(mean_dis_pos/0.5, ALPHA) * mean_dis - BETA2 * np.power(mean_dis_neg/0.5, ALPHA) * mean_dis
#             res[res > 1] = 1
#             res[res < 0] = 0
#             # cv2.imshow('original', syn_image1_bgr)
#             syn_image1_yuv[:, :, 0] = res*255
#             syn_image1_out = cv2.cvtColor(syn_image1_yuv, cv2.COLOR_YUV2BGR)
#
#             syn_img = (syn_part_rgb(syn_image1_out.astype(np.float32) / 255, output_image_r, att)*255).astype(np.uint8)
#
#             cv2.imwrite(os.path.join(dst_dir,img_name),syn_img)

