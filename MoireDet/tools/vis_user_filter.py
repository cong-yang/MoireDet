# -*- coding: UTF-8 -*-
'''=================================================
@Author ：zhenyu.yang
@Date   ：2020/12/26 6:14 PM
=================================================='''
import cv2
import numpy as np
import shutil


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
        filter = cv2.line(filter,(x,y),(kernel_size-x-1,kernel_size-y-1),1,1)
        filter = filter/np.sum(filter)
        filter = filter/np.max(filter) * 255
        filters.append(filter)

    return filters


filters = generate_filters(7)

for i,filter in enumerate(filters):
    filter = cv2.resize(filter,(120,120), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('{}.jpg'.format(i),filter)



