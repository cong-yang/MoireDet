# -*- coding: UTF-8 -*-
'''=================================================
@Author ：zhenyu.yang
@Date   ：2020/12/22 9:24 PM
=================================================='''

import os
import shutil
import random

def split_datasets(src_dir,train_dir,test_dir,train_ratio = 0.8):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    imgs =[os.path.join(src_dir,v) for v in os.listdir(src_dir)]

    random.shuffle(imgs)

    train_imgs = imgs[:int(len(imgs)*train_ratio)]
    eval_imgs = imgs[int(len(imgs)*train_ratio):]

    for img in train_imgs:
        shutil.copy(img,train_dir)

    for img in eval_imgs:
        shutil.copy(img,test_dir)



def split_moire_datasets(src_dir,train_dir,test_dir,train_ratio = 0.8):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(train_dir.replace('layers_ori','layers_conv')):
        os.makedirs(train_dir.replace('layers_ori','layers_conv'))


    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    if not os.path.exists(test_dir.replace('layers_ori','layers_conv')):
        os.makedirs(test_dir.replace('layers_ori','layers_conv'))



    imgs =[os.path.join(src_dir,v) for v in os.listdir(src_dir)]

    random.shuffle(imgs)

    train_imgs = imgs[:int(len(imgs)*train_ratio)]
    eval_imgs = imgs[int(len(imgs)*train_ratio):]

    for img in train_imgs:

        shutil.copy(img.replace('layers_ori','layers_conv').replace('.jpg','.png'), train_dir.replace('layers_ori','layers_conv'))
        shutil.copy(img,train_dir)


    for img in eval_imgs:
        shutil.copy(img.replace('layers_ori','layers_conv').replace('.jpg','.png'), test_dir.replace('layers_ori','layers_conv'))
        shutil.copy(img,test_dir)



if __name__ == '__main__':
    train_ratio = 0.8
    src_dir = '/data/cong/moire/final/layers_ori'
    train_dir = '/data/zhenyu/moire/train/layers_ori'
    test_dir = '/data/zhenyu/moire/test/layers_ori'
    split_moire_datasets(src_dir, train_dir, test_dir, train_ratio)


    src_dirs = ['/data/cong/moire/natural/coco','/data/cong/moire/natural/imagenetsmall',
                '/data/cong/moire/natural/retail','/data/cong/moire/natural/voc']

    train_dirs = ['/data/zhenyu/moire/train/natural/coco','/data/zhenyu/moire/train/natural/imagenetsmall',
                  '/data/zhenyu/moire/train/natural/retail','/data/zhenyu/moire/train/natural/voc']

    test_dirs = ['/data/zhenyu/moire/test/natural/coco','/data/zhenyu/moire/test/natural/imagenetsmall',
                '/data/zhenyu/moire/test/natural/retail','/data/zhenyu/moire/test/natural/voc']

    for src_dir,train_dir,test_dir in zip(src_dirs,train_dirs,test_dirs):
        split_datasets(src_dir, train_dir, test_dir, train_ratio)
