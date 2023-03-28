# -*- coding: UTF-8 -*-
'''=================================================
@Author ：zhenyu.yang
@Date   ：2021/1/7 3:04 PM
=================================================='''

import cv2
import numpy as np
import os
import  shutil



img_path = '/data/zhenyu/moire/test/layers_ori/xiaomi10_labtv_219.jpg'
# img_path = '/data/zhenyu/moire/train/layers_ori/note3_labtv_544.jpg'
# img_path = '/data/zhenyu/moire/train/layers_ori/jianguo_aoc_544.jpg'
shutil.copy(img_path,'./')

img_name = os.path.basename(img_path).replace('.jpg','')
img = cv2.imread(img_path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
sobel = (sobelx**2+sobely**2)**0.5
Max = sobel.max()
sobelx = sobelx/Max*255
sobely = sobely/Max*255
sobel = sobel/Max*255

rgb_soble = np.stack([sobelx,sobely,sobel],axis=-1)
cv2.imwrite('{}_gradient_x.jpg'.format(img_name),np.uint8(sobelx))
cv2.imwrite('{}_gradient_y.jpg'.format(img_name),np.uint8(sobely))
cv2.imwrite('{}_gradient_amp.jpg'.format(img_name),np.uint8(sobel))
cv2.imwrite('{}_gradient_x_y_amp.jpg'.format(img_name),np.uint8(rgb_soble))

