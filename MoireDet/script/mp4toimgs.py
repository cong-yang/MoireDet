# -*- coding: UTF-8 -*-
'''=================================================
@Author ：zhenyu.yang
@Date   ：2021/3/22 11:05 PM
=================================================='''
import cv2
import os
src_dir = '/home/users/zhenyu.yang/data/research/moire_new/test/final_mp4'
dst_dir = '/home/users/zhenyu.yang/data/research/moire_new/test/final_mp4_imgs'

video_list = os.listdir(src_dir)
for video in video_list:
    cap = cv2.VideoCapture(os.path.join(src_dir,video))
    index = -1
    while True:
        index += 1
        ret,frame = cap.read()
        if not ret:
            break

        temp_dir = os.path.join(dst_dir,video.replace('.mp4','').replace('.avi',''))
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        cv2.imwrite(os.path.join(temp_dir,'{}.jpg'.format(str(index).zfill(5))),frame)

