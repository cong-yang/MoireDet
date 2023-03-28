# -*- coding: UTF-8 -*-
'''=================================================
@Author ：zhenyu.yang
@Date   ：2021/1/4 10:47 AM
=================================================='''
import cv2
import os
import numpy as np
from multiprocessing import  Pool

src = '/home/users/zhenyu.yang/data/output/moire_single_exp/'
src = '/home/users/zhenyu.yang/data/output/moire_single_exp/final_mp4_all'


dirs_list = os.listdir(src)
dirs_list.sort()
dirs_list = [os.path.join(src,v) for v in dirs_list if 'VID_20210318_230058' in v and '.avi' not in v and '.zip' not in v  and '.tar' not in v]
dirs_list.sort()
dirs_list = [v for v in dirs_list if len(os.listdir(v)) >= 50]
print([os.path.basename(v) for v in dirs_list])
del dirs_list[0]
del dirs_list[1]
del dirs_list[1]
del dirs_list[-1]
del dirs_list[-1]

base_list = ['VID_20210318_230058_320', 'VID_20210318_230058_HFNet', 'VID_20210318_230058_DSC', 'VID_20210318_230058_DCNN', 'VID_20210318_230058_DED']


index_list = [0,-1,-2,1,2]
dirs_list = [dirs_list[v] for v in index_list]
print([os.path.basename(v) for v in dirs_list])

# dirs_list = dirs_list[:4]
# print([os.path.basename(v) for v in dirs_list])

dst_dir = '/home/users/zhenyu.yang/data/output/moire_single_exp/final_mp4_all/VID_20210318_230058_all_320'

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
img_list.sort()
img_list = img_list[:1000]
w = 1920
h = 1080


# for img_name in img_list:
#     imgs = []
#     for index,dir in enumerate(dirs_list) :
#         img = os.path.join(dir,img_name)
#         if not os.path.exists(img):
#             continue
#         img = cv2.imread(img)
#         if index == 0:
#             h,w = img.shape[:2]
#             w = w//2
#         if index > 0:
#             img = cv2.resize(img,(w,h))
#         imgs.append(img)
#
#     imgs = np.concatenate(imgs,axis=1)
#
#     cv2.imwrite(os.path.join(dst_dir ,img_name),imgs)

def combine_img(img_name):
    imgs = []
    for index,dir in enumerate(dirs_list) :
        img = os.path.join(dir,img_name)
        if not os.path.exists(img):
            continue
        img = cv2.imread(img)
        if index == 0:
            h,w = img.shape[:2]
            w = w//2
        if index > 0:
            img = cv2.resize(img,(w,h))
        imgs.append(img)

    imgs_1 = np.concatenate(imgs[:2],axis=1)
    imgs_2 = np.concatenate(imgs[2:],axis=1)
    imgs = np.concatenate([imgs_1,imgs_2], axis=0)


    cv2.imwrite(os.path.join(dst_dir ,img_name),imgs)




pool = Pool(32)
pool.map(combine_img,img_list)
pool.close()
pool.join()