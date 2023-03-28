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
        filter = cv2.line(filter,(x,y),(kernel_size-x,kernel_size-y),1,1)
        filter = filter/np.sum(filter)
        filters.append(filter)

    return filters


filters = generate_filters(7)

natural_img = '/data/zhenyu/moire/train/natural/coco/000000196747.jpg'

# moire_img = '/data/zhenyu/moire/train/layers_ori/note3_labtv_544.jpg'
# moire_conv_img = '/data/zhenyu/moire/train/layers_conv/note3_labtv_544.png'

# moire_img = '/data/zhenyu/moire/test/layers_ori/xiaomi10_labtv_219.jpg'
# moire_conv_img = '/data/zhenyu/moire/test/layers_conv/xiaomi10_labtv_219.png'

natural_img = './ZHENYU/IMG_20171013_115530R.jpg'
moire_img = './ZHENYU/xiaomi10_labtv_208.jpg'
moire_conv_img = './ZHENYU/xiaomi10_labtv_208.png'

moire_conv = cv2.imread(moire_conv_img)
shutil.copy(moire_conv_img,'./')
shutil.copy(natural_img,'./')
shutil.copy(moire_img,'./')


natural_img = cv2.imread(natural_img)
natural_img = natural_img
h,w = natural_img.shape[:2]
h,w = 480,720
# h,w = h*2,w*2
natural_img = cv2.resize(natural_img,dsize=(w,h))

moire_img = cv2.imread(moire_img)
moire_img = cv2.resize(moire_img,dsize=(w,h))


mask = 255 * np.ones(moire_img.shape, moire_img.dtype)
center = (int(w/2),int(h/2))


mixed_clone = cv2.seamlessClone(moire_img, natural_img*0, mask, center, cv2.MIXED_CLONE)
# mixed_clone = np.clip(cv2.blur(mixed_clone, (3, 3)).astype(np.float)*4,0,255)
mixed_clone = cv2.cvtColor(mixed_clone,cv2.COLOR_BGR2GRAY)
filted_imgs = []
for filter in filters:
    temp_mixed_clone = mixed_clone.copy()
    filted_imgs.append(cv2.filter2D(temp_mixed_clone,-1,filter).astype(np.float))
filted_imgs = np.stack(filted_imgs,axis=0)
weight = 0.3
# mixed_clone = np.max(filted_imgs,axis=0)*weight + np.min(filted_imgs,axis=0) *(1-weight)

# mixed_clone = (np.max(filted_imgs,axis=0)*np.min(filted_imgs,axis=0))**0.5
# mixed_clone = np.clip(mixed_clone.astype(np.float)*4,0,255)



filted_imgs = [255 - np.min(filted_imgs,axis=0),np.max(filted_imgs,axis=0)*30]
filted_imgs = np.argmax(np.stack(filted_imgs,axis=0),axis=0)*255
mixed_clone = np.clip(filted_imgs.astype(np.float),0,255)
# mixed_clone = np.clip(cv2.medianBlur(mixed_clone, 7).astype(np.float)*7,0,255)
# mixed_clone = np.clip(cv2.bilateralFilter(mixed_clone,0,30,30) .astype(np.float)*7,0,255)


# mixed_clone = np.clip(cv2.GaussianBlur(mixed_clone, (7, 7),0).astype(np.float)*7,0,255)
# mixed_clone = (mixed_clone > 50)*255
mixed_clone = np.uint8(mixed_clone)

moire_conv = cv2.resize(moire_conv, (720, 480))

moire_conv = 255 - moire_conv[:,:,0]

# moire_conv = cv2.resize(moire_conv,dsize=(w//2,h//2))
# mixed_clone = cv2.GaussianBlur(mixed_clone,(13,13),0)
moire_conv = cv2.blur(moire_conv, (21, 21))
moire_conv = cv2.GaussianBlur(moire_conv, (13, 13), 0).astype(np.float)
moire_conv = np.uint8(np.clip(moire_conv * 3, 0, 255))
moire_conv = cv2.resize(moire_conv,dsize=(w,h))
moire_conv = 255 - cv2.cvtColor(moire_conv,cv2.COLOR_GRAY2BGR)
cv2.imwrite('moire_conv.jpg',moire_conv)


mixed_clone = 255-mixed_clone
mixed_clone = cv2.cvtColor(mixed_clone,cv2.COLOR_GRAY2BGR)

imgs = np.concatenate([natural_img,moire_img,moire_conv,mixed_clone],axis=0)

# mixed_clone = 255 - mixed_clone
cv2.imwrite('mixed_clone.jpg',mixed_clone)
cv2.imwrite('mixed_clone_imgs_big.jpg',imgs)