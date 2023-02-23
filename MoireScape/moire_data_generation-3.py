# -*- coding: UTF-8 -*-
'''=================================================
@Author : zhenyu.yang
@Date   : 2020/12/22 9:24 PM
=================================================='''

import shutil
import numpy as np
import os
import cv2
import random
from multiprocessing import Pool
import albumentations as A
from functools import partial


def split_moire_datasets(src_dir, train_dir, test_dir, train_ratio=0.8):
    os.makedirs(train_dir, exist_ok=True)

    os.makedirs(train_dir.replace('layers_ori', 'layers_ori_pattern'), exist_ok=True)

    os.makedirs(test_dir, exist_ok=True)

    os.makedirs(test_dir.replace('layers_ori', 'layers_ori_pattern'), exist_ok=True)

    imgs = [os.path.join(src_dir, v) for v in os.listdir(src_dir)]

    random.shuffle(imgs)

    train_imgs = imgs[:int(len(imgs) * train_ratio)]
    eval_imgs = imgs[int(len(imgs) * train_ratio):]

    for img in train_imgs:
        shutil.copy(
            img.replace('layers_ori', 'layers_ori_pattern'),
            train_dir.replace('layers_ori', 'layers_ori_pattern'))
        shutil.copy(img, train_dir)

    for img in eval_imgs:
        shutil.copy(
            img.replace('layers_ori', 'layers_ori_pattern'),
            test_dir.replace('layers_ori', 'layers_ori_pattern'))
        shutil.copy(img, test_dir)


def split_datasets(src_dir, train_dir, test_dir, train_ratio=0.8):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    imgs = [os.path.join(src_dir, v) for v in os.listdir(src_dir)]

    random.shuffle(imgs)

    train_imgs = imgs[:int(len(imgs) * train_ratio)]
    eval_imgs = imgs[int(len(imgs) * train_ratio):]

    for img in train_imgs:
        shutil.copy(img, train_dir)

    for img in eval_imgs:
        shutil.copy(img, test_dir)


def getFiles(path, suffix):
    return [
        os.path.join(root, file)
        for root, dirs, files in os.walk(path)
        for file in files
        if file.endswith(suffix)
    ]


def Multiply(moire, origin, merged=None):
    # Mix moire and natural images
    moire = moire.astype(np.float) / 255
    origin = origin.astype(np.float) / 255

    merged = (moire * origin) * 255
    merged = np.uint8(np.clip(merged, 0, 255))

    return merged


def scale(img, ratio, size):
    w = int(size[0] * ratio)
    h = int(size[1] * ratio)
    img = cv2.resize(img, dst_dir=(w, h))
    return img


def crop(img, max_size=(640, 380), min_ratio=0.6, full_prob=0.1):
    h, w = img.shape[:2]

    max_w, max_h = max_size
    max_h = min(max_h, h)
    max_w = min(max_w, w)

    min_h, min_w = int(max_h * min_ratio), int(max_w * min_ratio)

    if random.random() < full_prob:
        min_h, min_w = max_h, max_w

    new_h = random.randrange(min_h - 1, max_h)
    new_w = random.randrange(min_w - 1, max_w)

    y = random.randrange(0, h - new_h)
    x = random.randrange(0, w - new_w)

    return img[y:y + new_h, x:x + new_w]


def rotate(img, borderValue=(255, 255, 255)):
    h, w = img.shape[:2]
    angle = random.randrange(-20, 20)
    rotate = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, 1)
    img = cv2.warpAffine(img, rotate, (w, h), borderValue=borderValue)
    # Flip
    if random.random() < 0.2:
        img = img[::-1, :, :]
    if random.random() < 0.2:
        img = img[:, ::-1, :]

    return img


def transform(moire, img_size=(640, 480), min_ratio=0.8, full_prob=0.3):
    scale_ratio = random.uniform(0.8, 1.25)
    w = int(img_size[0] * scale_ratio)
    h = int(img_size[1] * scale_ratio)
    moire = cv2.resize(moire, dsize=(w, h))
    moire = crop(moire, max_size=img_size, min_ratio=min_ratio, full_prob=full_prob)

    b, g, r = np.mean(moire[:, :, 0]), np.mean(moire[:, :, 1]), np.mean(moire[:, :, 2])

    bgr = np.mean([b, g, r])

    moire = rotate(moire, (bgr, bgr, bgr, 255))
    # moire 旋转之后会有一些黑边，如果直接和原图融合，会导致失真，此处用均值来填充黑边。

    h, w = moire.shape[:2]

    blank = np.zeros((img_size[1], img_size[0], 4))
    blank[:, :, 0] += b
    blank[:, :, 1] += g
    blank[:, :, 2] += r
    blank[:, :, 3] += 255

    x = random.randint(0, img_size[0] - w)
    y = random.randint(0, img_size[1] - h)

    blank[y:y + h, x:x + w, :] = moire

    return blank


def generate_single(
        index,
        moire_imgs,
        natural_imgs,
        moire_pattern_dir,
        dst_img_dir,
        dst_combined_dir,
        transformed_moire_dir,
):
    moire_path = moire_imgs[index % len(moire_imgs)]

    natural_img = random.choice(natural_imgs)

    moire_pattern_path = os.path.join(moire_pattern_dir, os.path.basename(moire_path))

    img_name = '{}.png'.format(str(index).zfill(8))

    try:
        natural_img = cv2.imread(natural_img)
        moire_img = cv2.imread(moire_path)
        moire_pattern_img = cv2.imread(moire_pattern_path)
        moire_img = np.concatenate([moire_img, moire_pattern_img[:, :, :1]], axis=-1)

    except Exception as e:
        print(e)
        return None

    temp_trans = A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.4)
    natural_img = temp_trans(image=natural_img)['image']

    h, w = natural_img.shape[:2]

    h = 480
    w = 640
    min_ratio = 0.6
    full_prob = 0.1

    natural_img = cv2.resize(natural_img, (w, h), min_ratio, full_prob)

    transformed_moire = transform(moire_img, (w, h))

    moire_conv = transformed_moire[:, :, 3:]
    moire_img = transformed_moire[:, :, :3]

    combined = Multiply(moire_img, natural_img)

    combined = np.concatenate([combined, moire_conv], axis=-1)

    cv2.imwrite(os.path.join(dst_combined_dir, img_name), combined)
    cv2.imwrite(os.path.join(dst_img_dir, img_name), natural_img)
    cv2.imwrite(os.path.join(transformed_moire_dir, img_name), transformed_moire)


def combine_moire_nature_imgs(moire_dir, natural_dir, moire_pattern_dir, dst_dir):
    dst_combined_dir = os.path.join(dst_dir, 'combined')
    dst_img_dir = os.path.join(dst_dir, 'img')
    transformed_moire_dir = os.path.join(dst_dir, 'moire')

    for path in [dst_dir, dst_combined_dir, dst_img_dir, transformed_moire_dir]:
        os.makedirs(path, exist_ok=True)

    natural_imgs = getFiles(natural_dir, '.jpg')
    moire_imgs = getFiles(moire_dir, '.jpg')
    random.shuffle(moire_imgs)
    _generate_single_img = partial(
        generate_single,
        moire_imgs=moire_imgs,
        natural_imgs=natural_imgs,
        moire_pattern_dir=moire_pattern_dir,
        dst_img_dir=dst_img_dir,
        dst_combined_dir=dst_combined_dir,
        transformed_moire_dir=transformed_moire_dir)

    pool = Pool(16)
    print(f"We will generate {2 * len(moire_imgs)} images")
    pool.map(_generate_single_img, range(2 * len(moire_imgs)))

    pool.close()
    pool.join()


if __name__ == '__main__':
    # STEP - 1: Define the original dataset

    #Moire imgs means the original moire camera images
    moire_imgs_dir = '/data/cong/moire/final/layers_ori'

    #Moire pattern means the extracted moire's edge By Cong Yang
    # 特别提醒，此处生成的文件和原始Moire图片名字是完全对齐的
    moire_patterns_dir = '/data/cong/moire/final/layers_ori_pattern'

    # Nature images means pure photo without any moire.
    natures_imgs_dirs = [
        '/data/cong/moire/natural/coco', '/data/cong/moire/natural/imagenetsmall',
        '/data/cong/moire/natural/retail', '/data/cong/moire/natural/voc'
    ]
    # 最终生成的数据地址
    dst_dir = "/data/cong/moire/"
  
    """
        Final dir tree under this dst_dir like:
        train
        --layers_ori
        --layers_ori_pattern
        --natural
        ----coco
        ----imagenetsmall
        ----retail
        ----voc
        test
        --layers_ori
        --layers_ori_pattern
        --natural
        ----coco
        ----imagenetsmall
        ----retail
        ----voc
    """

    # Setp - 2: Split datasets
    train_ratio = 0.8
    moire_train_dir = os.path.join(dst_dir, "train", "layers_ori")
    moire_test_dir = os.path.join(dst_dir, "test", "layers_ori")
    split_moire_datasets(moire_imgs_dir, moire_train_dir, moire_test_dir, train_ratio)

    for natures_imgs_dir in natures_imgs_dirs:
        nature_type = os.path.basename(natures_imgs_dir)
        train_dir = os.path.join(dst_dir, "train", "natural", nature_type)
        test_dir = os.path.join(dst_dir, "test", "natural", nature_type)
        split_datasets(natures_imgs_dir, train_dir, test_dir, train_ratio)

    for data_type in ["train", "test"]:
        moire_dir = os.path.join(dst_dir, data_type, "layers_ori")
        natural_dir = os.path.join(dst_dir, data_type, "natural")
        moire_pattern_dir = os.path.join(dst_dir, data_type, "layers_ori_pattern")
        final_dst_dir = os.path.join(dst_dir, data_type, "generated_data")
        combine_moire_nature_imgs(moire_dir, natural_dir, moire_pattern_dir, final_dst_dir)
