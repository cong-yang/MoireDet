import numpy as np
import matplotlib.pyplot as plt
import os
import time
import cv2
import sys
sys.path.append('./')
import scipy.io
import scipy.stats as st
import random
from multiprocessing import Pool

moire_dir = '/data/zhenyu/moire/train/layers_conv'
dst_dir = '/data/zhenyu/moire/train/layers_blur'

moire_dir = '/data/zhenyu/moire/test/layers_conv'
dst_dir = '/data/zhenyu/moire/test/layers_blur'


if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)



def getFiles(path, suffix):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path)
            for file in files if file.endswith(suffix)]

def blur_moire_conv(moire_path,dst_size = (600,600)):
    dst_w,dst_h = dst_size
    moire_name = os.path.basename(moire_path)
    moire_conv = cv2.imread(moire_path)
    # moire_conv = cv2.resize(moire_conv,(720,480))

    moire_conv = 255 - moire_conv[:, :, 0]
    moire_conv = cv2.blur(moire_conv, (21, 21))
    moire_conv = cv2.GaussianBlur(moire_conv, (13, 13), 0).astype(np.float)
    moire_conv = np.uint8(np.clip(moire_conv * 3, 0, 255))

    cv2.imwrite(os.path.join(dst_dir,moire_name),moire_conv)


def blur_moire_conv(moire_path,dst_size = (600,600)):
    dst_len = min(dst_size)
    moire_name = os.path.basename(moire_path)
    moire_conv = cv2.imread(moire_path)
    moire_conv = cv2.resize(moire_conv,dst_size)

    moire_conv = 255 - moire_conv[:, :, 0]
    kernel_size = int(21/600*dst_len)
    moire_conv = cv2.blur(moire_conv, (kernel_size, kernel_size))

    kernel_size = int(13/600*dst_len)
    moire_conv = cv2.GaussianBlur(moire_conv, (kernel_size, kernel_size), 0).astype(np.float)
    moire_conv = np.uint8(np.clip(moire_conv * 3, 0, 255))
    return moire_conv

    cv2.imwrite(os.path.join(dst_dir,moire_name),moire_conv)


if __name__ =='__main__':
    pool = Pool(32)
    moire_list = getFiles(moire_dir,'.png')
    pool.map(blur_moire_conv,moire_list)
    pool.join()
