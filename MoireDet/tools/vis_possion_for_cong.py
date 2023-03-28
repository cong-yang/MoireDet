# -*- coding: UTF-8 -*-
'''=================================================
@Author ：zhenyu.yang
@Date   ：2020/12/26 6:14 PM
=================================================='''
import cv2
import numpy as np
import shutil

natural_img = '/data/zhenyu/moire/test/natural/voc/2008_006470.jpg'

moire_img = '/data/zhenyu/moire/test/layers_ori/note3_labtv_202.jpg'
moire_conv_img = '/data/zhenyu/moire/test/layers_conv/note3_labtv_202.png'
img_name = '00000350648_note3_labtv_202.jpg'


moire_img = '/data/zhenyu/moire/test/layers_ori/xiaomi10_aoc_4656.jpg'
moire_conv_img = '/data/zhenyu/moire/test/layers_conv/xiaomi10_aoc_4656.png'
img_name = '00000350648_xiaomi10_aoc_4656.jpg'
#
#
#
# moire_img = '/data/zhenyu/moire/test/layers_ori/xiaomi10_aoc_4187.jpg'
# moire_conv_img = '/data/zhenyu/moire/test/layers_conv/xiaomi10_aoc_4187.png'
# img_name = '00000350648_xiaomi10_aoc_4187.jpg'


# moire_img = '/data/zhenyu/moire/train/layers_ori/note3_labtv_544.jpg'
# moire_conv_img = '/data/zhenyu/moire/train/layers_conv/note3_labtv_544.png'





moire_conv = cv2.imread(moire_conv_img)
shutil.copy(moire_conv_img,'./')
shutil.copy(natural_img,'./')
shutil.copy(moire_img,'./')


natural_img = cv2.imread(natural_img)
natural_img = natural_img
h,w = natural_img.shape[:2]
w,h = (720, 480)
# h,w = h*2,w*2
natural_img = cv2.resize(natural_img,dsize=(w,h))

moire_img = cv2.imread(moire_img)
moire_img = cv2.resize(moire_img,dsize=(w,h))


mask = 255 * np.ones(moire_img.shape, moire_img.dtype)
center = (int(w/2),int(h/2))


mixed_clone = cv2.seamlessClone(moire_img, natural_img, mask, center, cv2.MIXED_CLONE)


moire_conv = cv2.resize(moire_conv, (720, 480))
moire_conv_ori = moire_conv.copy()

moire_conv = 255 - moire_conv[:,:,0]

# moire_conv = cv2.resize(moire_conv,dsize=(w//2,h//2))
# mixed_clone = cv2.GaussianBlur(mixed_clone,(13,13),0)
moire_conv = cv2.blur(moire_conv, (23, 23))
moire_conv = cv2.GaussianBlur(moire_conv, (13, 13), 0).astype(np.float)
moire_conv = np.uint8(np.clip(moire_conv * 3, 0, 255))
moire_conv = cv2.resize(moire_conv,dsize=(w,h))
moire_conv = 255 - cv2.cvtColor(moire_conv,cv2.COLOR_GRAY2BGR)
# cv2.imwrite('moire_conv.jpg',moire_conv)




imgs = np.concatenate([mixed_clone,moire_conv],axis=0)


cv2.imwrite('./ZHENYU/{}'.format(img_name),imgs)
# cv2.imwrite('mixed_clone_imgs_big.jpg',imgs)