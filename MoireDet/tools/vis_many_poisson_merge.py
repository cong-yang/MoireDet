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
dst_dir = '/data/zhenyu/moire/train/combined'


natural_dir = '/data/zhenyu/moire/test/natural'
moire_dir = '/data/zhenyu/moire/test/layers_ori'
moire_conv_dir = '/data/zhenyu/moire/test/layers_conv'

dst_dir = '/data/zhenyu/moire/test/temp_combined_new_poisson_3'


if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

natural_imgs = getFiles(natural_dir, '.jpg')
moire_imgs = getFiles(moire_dir,'.jpg')
random.shuffle(moire_imgs)


def generate_single(moire_path):
    moire_conv_path = moire_path.replace('layers_ori','layers_conv').replace('.jpg', '.png')
    moire_conv = cv2.imread(moire_conv_path)

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
    h, w = 480, 720

    natural_img = cv2.resize(natural_img, dsize=(w, h))


    moire_img = cv2.resize(moire_img, dsize=(w, h))

    mask = 255 * np.ones(moire_img.shape, moire_img.dtype)
    center = (int(w / 2), int(h / 2))

    mixed_clone = cv2.seamlessClone(moire_img, natural_img, mask, center,cv2.MIXED_CLONE)


    get_moire = cv2.seamlessClone(moire_img, natural_img*0+128, mask, center,cv2.MIXED_CLONE)
    get_moire = cv2.cvtColor(get_moire,cv2.COLOR_BGR2GRAY)

    # mixed_clone = np.clip(cv2.medianBlur(mixed_clone, 7).astype(np.float) * 7,0, 255)

    # get_moire = np.clip(
    #     cv2.GaussianBlur(get_moire, (7, 7), 0).astype(np.float) * 7, 0, 255)
    # mixed_clone = (mixed_clone > 50)*255
    get_moire = np.uint8(get_moire)




    moire_conv = cv2.resize(moire_conv, dsize=(w, h))
    moire_conv_ori = moire_conv.copy()

    moire_conv = 255 - moire_conv[:, :, 0]
    # mixed_clone = cv2.GaussianBlur(mixed_clone,(13,13),0)
    moire_conv = cv2.blur(moire_conv, (21, 21))
    moire_conv = cv2.GaussianBlur(moire_conv, (13, 13), 0).astype(np.float)
    moire_conv = np.uint8(np.clip(moire_conv * 3, 0, 255))
    # moire_conv = cv2.resize(moire_conv, dsize=(w, h))



    moire_conv = 255 - cv2.cvtColor(moire_conv, cv2.COLOR_GRAY2BGR)
    get_moire = cv2.cvtColor(get_moire, cv2.COLOR_GRAY2BGR)




    imgs = np.concatenate([natural_img, moire_img,moire_conv_ori,moire_conv,get_moire,mixed_clone],axis=0)

    cv2.imwrite(os.path.join(dst_dir, img_name), imgs)



if __name__ =='__main__':
    for _ in range(1):
        pool = Pool(32)
        random.shuffle(moire_imgs)
        pool.map(generate_single,moire_imgs)
        pool.join()
