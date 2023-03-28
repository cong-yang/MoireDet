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
        filter = cv2.line(filter,(x,y),(kernel_size-x,kernel_size-y),1,1)
        filter = filter/np.sum(filter)
        filters.append(filter)

    return filters


filters = generate_filters(7)


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

dst_dir = '/data/zhenyu/moire/test/temp_combined_new_poisson_user_filters_4'



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

    moire_conv = cv2.resize(moire_conv, dsize=(w, h))

    moire_img = cv2.resize(moire_img, dsize=(w, h))

    mask = 255 * np.ones(moire_img.shape, moire_img.dtype)
    center = (int(w / 2), int(h / 2))

    mixed_clone_1 = 255 - cv2.seamlessClone(255 - moire_img, 255 - natural_img, mask, center,cv2.MIXED_CLONE)
    mixed_clone = cv2.seamlessClone(moire_img, natural_img, mask, center,cv2.MIXED_CLONE)


    get_moire = cv2.seamlessClone(moire_img, natural_img*0, mask, center,cv2.MIXED_CLONE)
    get_moire = cv2.cvtColor(get_moire,cv2.COLOR_BGR2GRAY)
    filted_imgs = []
    for filter in filters:
        temp_mixed_clone = 255 - moire_conv[:,:,0].copy()

        # temp_mixed_clone = moire_conv.copy()
        filted_imgs.append(
            cv2.filter2D(temp_mixed_clone, -1, filter).astype(np.float))
    filted_imgs = np.stack(filted_imgs, axis=0)
    weight = 0.3
    # mixed_clone = np.max(filted_imgs,axis=0)*weight + np.min(filted_imgs,axis=0) *(1-weight)
    get_moire = (np.max(filted_imgs, axis=0) * np.min(filted_imgs,
                                                        axis=0)) ** 0.5

    get_moire = np.clip(get_moire.astype(np.float) * 4, 0, 255)

    # get_moire = np.clip(
    #     cv2.GaussianBlur(get_moire, (7, 7), 0).astype(np.float) * 7, 0, 255)
    # mixed_clone = (mixed_clone > 50)*255
    get_moire = np.uint8(get_moire)
    get_moire = cv2.medianBlur(get_moire,7)





    moire_conv_ori = moire_conv.copy()

    moire_conv = 255 - moire_conv[:, :, 0]
    # mixed_clone = cv2.GaussianBlur(mixed_clone,(13,13),0)
    moire_conv = cv2.blur(moire_conv, (21, 21))
    moire_conv = cv2.GaussianBlur(moire_conv, (13, 13), 0).astype(np.float)
    moire_conv = np.uint8(np.clip(moire_conv * 3, 0, 255))
    # moire_conv = cv2.resize(moire_conv, dsize=(w, h))



    moire_conv = 255 - cv2.cvtColor(moire_conv, cv2.COLOR_GRAY2BGR)
    get_moire = 255 - cv2.cvtColor(get_moire, cv2.COLOR_GRAY2BGR)




    imgs = np.concatenate([natural_img, moire_img,moire_conv_ori,moire_conv,get_moire,mixed_clone],axis=0)

    cv2.imwrite(os.path.join(dst_dir, img_name), imgs)



if __name__ =='__main__':
    for _ in range(1):
        pool = Pool(32)
        random.shuffle(moire_imgs)
        pool.map(generate_single,moire_imgs)
        pool.join()
