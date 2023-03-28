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

from albumentations import (IAAPiecewiseAffine,OpticalDistortion,IAAPerspective,RandomRotate90,HorizontalFlip,VerticalFlip)



def getFiles(path, suffix):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path)
            for file in files if
            file.endswith(suffix)]

natural_dir = '/data/zhenyu/moire/train/natural'
moire_dir = '/data/zhenyu/moire/train/layers_ori'
dst_dir = '/data/zhenyu/moire/train/combined_more_trans'


# natural_dir = '/data/zhenyu/moire/test/natural'
# moire_dir = '/data/zhenyu/moire/test/layers_ori'
# dst_dir = '/data/zhenyu/moire/test/combined_new'


if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

natural_imgs = getFiles(natural_dir, '.jpg')
moire_imgs = getFiles(moire_dir,'.jpg')
random.shuffle(moire_imgs)


def scale(img,ratio,size):
    w = int(size[0]*ratio)
    h = int(size[1] * ratio)
    img = cv2.resize(img,dst_dir = (w,h))
    return img


def perspective(img):
    trans = IAAPerspective((0.1,0.18),p=0.8)
    return trans(image=img)['image']


def crop(img,max_size = (640,380),min_ratio = 0.5):
    h,w = img.shape[:2]

    max_h ,max_w = max_size
    max_h = min(max_h,h)
    max_w = min(max_w, w)

    min_h , min_w = int(max_h*min_ratio),int(max_w*min_ratio)

    new_h = random.randrange(min_h,max_h-1)
    new_w = random.randrange(min_w, max_h - 1)

    y = random.randrange(0,h-new_h)
    x = random.randrange(0, w - new_w)

    return img[y:y+new_h,x:x+new_w]


def rotate(img):
    h,w = img.shape[:2]
    angle = random.randrange(0,180)
    rotate = cv2.getRotationMatrix2D((h * 0.5, w * 0.5), angle, 1)
    img = cv2.warpAffine(img, rotate, (h, w))

    if random.random() < 0.2:
        img = img[::-1,:,:]
    if random.random() < 0.2:
        img = img[:,::-1,:]

    return img


def split_moire_mask_edge():
    pass


def transform(moire,img_size=(640,480)):
    scale_ratio = random.uniform(0.5,3)
    w = int(img_size[0]*scale_ratio)
    h = int(img_size[1] * scale_ratio)
    moire = cv2.resize(moire,dst_dir = (w,h))




def generate_single(moire_path):
    natural_img = random.choice(natural_imgs)

    moire_name = os.path.basename(moire_path).replace('.jpg', '')
    natural_name = os.path.basename(natural_img).replace('.jpg', '')
    natural_src = natural_img.split(os.sep)[-2]
    img_name = '{}__{}__{}.jpg'.format(moire_name, natural_src,natural_name)
    try:
        natural_img = cv2.imread(natural_img)
        moire_img = cv2.imread(moire_path)
    except:
        return None

    h, w = natural_img.shape[:2]
    h = 480
    w = 640
    natural_img = cv2.resize(natural_img, dsize=(w, h))

    moire_img = cv2.resize(moire_img, dsize=(w, h))

    mask = 255 * np.ones(moire_img.shape, moire_img.dtype)
    center = (int(w / 2), int(h / 2))

    mixed_clone = cv2.seamlessClone(moire_img, natural_img, mask, center,cv2.MIXED_CLONE)



    cv2.imwrite(os.path.join(dst_dir, img_name), mixed_clone)



if __name__ =='__main__':
    for _ in range(1):
        pool = Pool(12)
        random.shuffle(moire_imgs)
        pool.map(generate_single,moire_imgs)
        pool.join()
