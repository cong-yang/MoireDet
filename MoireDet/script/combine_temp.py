# -*- coding: UTF-8 -*-
'''=================================================
@Author ：zhenyu.yang
@Date   ：2021/1/4 10:47 AM
=================================================='''
import cv2
import os
import numpy as np


src = '/home/users/zhenyu.yang/data/output/moire_single_exp/'
src = '/home/users/zhenyu.yang/data/output/moire_real_exp/'


dirs_list = os.listdir(src)
dirs_list.sort()
dirs_list = [os.path.join(src,v) for v in dirs_list]
# dirs_list = [v for v in dirs_list if len(os.listdir(v)) >= 50]
print([os.path.basename(v) for v in dirs_list])

dirs_list = [dirs_list[-4],dirs_list[-5],dirs_list[0]]
dirs_list.insert(1,os.path.join(src,'Blur_New'))
dirs_list.insert(0,os.path.join(src,'Matting_New'))
print([os.path.basename(v) for v in dirs_list])

# dirs_list = dirs_list[:4]
# print([os.path.basename(v) for v in dirs_list])

dst_dir = '/home/users/zhenyu.yang/data/output/moire/real_temp_pk_new'

# origin = dirs_list[0]
# dirs_list = dirs_list[4:8]
# dirs_list.insert(0,origin)
# print([os.path.basename(v) for v in dirs_list])
# dst_dir = '/home/users/zhenyu.yang/data/output/moire/combined_moire_real_exp_net'


# origin = dirs_list[0]
# dirs_list = dirs_list[8:]
# dirs_list.insert(0,origin)
# print([os.path.basename(v) for v in dirs_list])
# dst_dir = '/home/users/zhenyu.yang/data/output/moire/combined_moire_real_exp_loss'



if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

imgs_list = []
for dir in dirs_list:
    imgs_list.append(os.listdir(dir))
img_list = set(imgs_list[0])
for temp_img_list in imgs_list[1:]:
    if len(temp_img_list) >= 50:
        img_list = set(temp_img_list) & img_list

img_list = list(img_list)

for img_name in img_list:
    imgs = []
    for dir in dirs_list:
        img = os.path.join(dir,img_name)
        if not os.path.exists(img):
            continue
        img = cv2.imread(img)
        img = cv2.resize(img,(320,960))
        imgs.append(img)

    imgs = np.concatenate(imgs,axis=1)

    cv2.imwrite(os.path.join(dst_dir ,img_name),imgs)

