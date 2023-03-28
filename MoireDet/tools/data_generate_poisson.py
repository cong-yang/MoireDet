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



def getFiles(path, suffix):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path)
            for file in files if
            file.endswith(suffix)]

natural_dir = '/data/zhenyu/moire/train/natural'
moire_dir = '/data/zhenyu/moire/train/layers_ori'
dst_dir = '/data/zhenyu/moire/train/combined_new'


# natural_dir = '/data/zhenyu/moire/test/natural'
# moire_dir = '/data/zhenyu/moire/test/layers_ori'
# dst_dir = '/data/zhenyu/moire/test/combined_new'


if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

natural_imgs = getFiles(natural_dir, '.jpg')
moire_imgs = getFiles(moire_dir,'.jpg')
random.shuffle(moire_imgs)


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
