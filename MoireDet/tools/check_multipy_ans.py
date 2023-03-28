# -*- coding: UTF-8 -*-
'''=================================================
@Author ：zhenyu.yang
@Date   ：2021/2/4 11:13 AM
=================================================='''
import cv2
import os
import random
import numpy as np

src_dir = '/home/users/zhenyu.yang/data/research/moire_new/train/combine_natural_new/combined'
img_list = [os.path.join(src_dir,v) for v in os.listdir(src_dir)]


dst_dir = '/home/users/zhenyu.yang/data/research/moire_new/train/combine_natural_new_check'
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)


random.shuffle(img_list)

for img in img_list[:400]:
    img_name = os.path.basename(img)
    img = cv2.imread(img,cv2.IMREAD_UNCHANGED)
    moire = img[:, :, 3:]
    img = img[:, :, :3]

    moire = cv2.cvtColor(moire,cv2.COLOR_GRAY2BGR)

    img = np.concatenate([img,moire],axis = 0)

    cv2.imwrite(os.path.join(dst_dir,img_name),img)

