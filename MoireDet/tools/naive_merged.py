# -*- coding: UTF-8 -*-
'''=================================================
@Author ：zhenyu.yang
@Date   ：2021/1/8 10:25 AM
=================================================='''
import cv2
import numpy as np
import os

natural_img = '/data/zhenyu/moire/train/natural/coco/000000196747.jpg'
moire_img = '/data/zhenyu/moire/test/layers_ori/xiaomi10_labtv_219.jpg'
moire_conv_img = './ZHENYU/xiaomi10_labtv_208.png'

dst_dir = './ZHENYU'
natural_img = cv2.imread(natural_img)
h,w = natural_img.shape[:2]


moire_img = cv2.imread(moire_img)
moire_img = cv2.resize(moire_img,dsize=(w,h))

natural_img,moire_img = natural_img.astype(np.float),moire_img.astype(np.float)
w = 0.3
for w in np.linspace(0.1, 1, 10):
    merged_img = np.uint8(np.clip(w*natural_img + (1-w)*moire_img,0,255))
    img_name = 'naive_{:.1f}.jpg'.format(w)
    cv2.imwrite(os.path.join(dst_dir,img_name),merged_img)



