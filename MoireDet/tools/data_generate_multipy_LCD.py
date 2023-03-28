import sys
sys.path.append('./')
sys.path.insert(0,'/data/zhenyu.yang/modules')
sys.path.insert(0,'/home/users/zhenyu.yang/data/env/transformer_related/')



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

import albumentations as A


def getFiles(path, suffix):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path)
            for file in files if
            file.endswith(suffix)]


natural_dir = '/home/users/zhenyu.yang/data/research/moire_new/test/natural'
moire_dir = '/home/users/zhenyu.yang/data/research/moire_new/test/moire_LCD'
moire_conv_dir = '/home/users/zhenyu.yang/data/research/moire_new/test/moire_LCD_pattern'

dst_dir = '/home/users/zhenyu.yang/data/research/moire_new/test/combine_LCD'


combined_dir = os.path.join(dst_dir,'combined')
img_dir = os.path.join(dst_dir,'img')
transformed_moire_dir = os.path.join(dst_dir,'moire')

for path in [combined_dir,img_dir,transformed_moire_dir]:
    if not os.path.exists(path):
        os.makedirs(path)

# natural_dir = '/data/zhenyu/moire/test/natural'
# moire_dir = '/data/zhenyu/moire/test/layers_ori'
# dst_dir = '/data/zhenyu/moire/test/combined_new'


if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

natural_imgs = getFiles(natural_dir, '.jpg')
moire_imgs = getFiles(moire_dir,'.jpg')
random.shuffle(moire_imgs)



def Multiply(moire,origin,merged=None):
    moire = moire.astype(np.float)/255
    origin = origin.astype(np.float)/255

    merged = (moire*origin)*255
    merged = np.uint8(np.clip(merged,0,255))

    return merged



def scale(img,ratio,size):
    w = int(size[0]*ratio)
    h = int(size[1] * ratio)
    img = cv2.resize(img,dst_dir = (w,h))
    return img

def perspective(img):
    trans = IAAPerspective((0.1,0.18),p=0.8)
    return trans(image=img)['image']


def crop_fix_ratio(img,ratio):
    h,w = img.shape[:2]



    new_h = int(h*ratio)
    new_w = int(w*ratio)

    y = random.randrange(0,h-new_h)
    x = random.randrange(0, w - new_w)

    return img[y:y+new_h,x:x+new_w]



def crop(img,max_size = (640,380),min_ratio = 0.6,full_prob = 0.1):
    h,w = img.shape[:2]

    max_w,max_h = max_size
    max_h = min(max_h,h)
    max_w = min(max_w, w)

    min_h , min_w = int(max_h*min_ratio),int(max_w*min_ratio)

    if random.random() < full_prob:
        min_h, min_w = max_h,max_w

    new_h = random.randrange(min_h-1,max_h)
    new_w = random.randrange(min_w-1, max_w)

    y = random.randrange(0,h-new_h)
    x = random.randrange(0, w - new_w)

    return img[y:y+new_h,x:x+new_w]


def rotate(img,borderValue=(255,255,255)):
    h,w = img.shape[:2]
    angle = random.randrange(0,180)
    rotate = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, 1)
    img = cv2.warpAffine(img, rotate, (w, h),borderValue=borderValue)

    if random.random() < 0.2:
        img = img[::-1,:,:]
    if random.random() < 0.2:
        img = img[:,::-1,:]

    return img



def transform(moire,img_size=(640,480),min_ratio = 0.6,full_prob=0.1):
    scale_ratio = random.uniform(0.8,2)
    w = int(img_size[0]*scale_ratio)
    h = int(img_size[1] * scale_ratio)
    moire = cv2.resize(moire,dsize = (w,h))
    moire = perspective(moire)
    moire = crop(moire,max_size=img_size,min_ratio = min_ratio,full_prob=full_prob)

    b,g,r = np.mean(moire[:,:,0]),np.mean(moire[:,:,1]),np.mean(moire[:,:,2])

    bgr = np.mean([b,g,r])

    moire = rotate(moire,(bgr,bgr,bgr,255))

    h,w = moire.shape[:2]

    blank = np.zeros((img_size[1],img_size[0],4))
    blank[:,:,0] += b
    blank[:, :, 1] += g
    blank[:, :, 2] += r
    blank[:, :, 3] += 255


    x = random.randint(0,img_size[0] - w )
    y = random.randint(0, img_size[1] - h)

    blank[y:y+h,x:x+w,:] = moire

    return blank





def generate_single(index):
    moire_path = moire_imgs[index%len(moire_imgs)]

    natural_img = random.choice(natural_imgs)

    moire_conv_path = os.path.join(moire_conv_dir,os.path.basename(moire_path))


    img_name = '{}.png'.format(str(index).zfill(8))

    try:
        natural_img = cv2.imread(natural_img)
        moire_img = cv2.imread(moire_path)
        moire_conv_img = cv2.imread(moire_conv_path)
        moire_conv_img = moire_conv_img[len(moire_conv_img)//2:]

        moire_img = np.concatenate([moire_img,moire_conv_img[:,:,:1]],axis=-1)

    except:
        return None

    temp_trans = A.ShiftScaleRotate(shift_limit=0.2,scale_limit=0.4)
    natural_img = temp_trans(image=natural_img)['image']

    h, w = natural_img.shape[:2]

    h = 480
    w = 640
    min_ratio = 0.6
    full_prob = 0.1

    natural_img = cv2.resize(natural_img,(w,h),min_ratio,full_prob)


    moire_img = crop_fix_ratio(moire_img,0.4)

    transformed_moire = transform(moire_img,(w,h))

    moire_conv = transformed_moire[:,:,3:]
    moire_img = transformed_moire[:,:,:3]

    combined = Multiply(moire_img,natural_img)

    combined = np.concatenate([combined,moire_conv],axis=-1)


    cv2.imwrite(os.path.join(combined_dir,img_name),combined)
    cv2.imwrite(os.path.join(img_dir, img_name), natural_img)
    cv2.imwrite(os.path.join(transformed_moire_dir, img_name), transformed_moire)



if __name__ =='__main__':
    _ = generate_single(1)
    pool = Pool(32)
    pool.map(generate_single,range(2*len(moire_imgs)))
    pool.close()
    pool.join()
