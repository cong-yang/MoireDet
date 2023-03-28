import sys
sys.path.insert(0,'/data/site-packages/site-packages')
sys.path.insert(0,'/home/zhenyu/env/pytorch/')
sys.path.insert(0,'./')
sys.path.insert(0,'/home/zhenyu/env/transformer_related')


import torch
import cv2
import numpy as np

img = cv2.imread('/data/zhenyu/moire/test/layers_ori/xiaomi10_labtv_219.jpg',0)
img = cv2.imread('/data/zhenyu/moire/train/natural/coco/000000196747.jpg',0)
img = cv2.imread('../tools/mixed_clone.jpg',0)
h,w = img.shape[:2]
data = np.stack([img,img*0],axis=-1)
x = torch.from_numpy(data).float().unsqueeze(0).repeat(5,1,1,1)
x = torch.fft(x, 2,normalized=False)


img = cv2.imread('../tools/ZHENYU/IMG_20171013_115530R.jpg',0)
img = cv2.imread('../tools/ZHENYU/xiaomi10_labtv_208.jpg',0)
img = cv2.resize(img,dsize=(w,h))



data = np.stack([img,img*0],axis=-1)
x2 = torch.from_numpy(data).float().unsqueeze(0).repeat(5,1,1,1)
x2 = torch.fft(x2, 2,normalized=False)
x = x - x2


amp = torch.log((x[0,:,:,0]**2+x[0,:,:,1]**2)**0.5)*20
# amp = x[:,:,0].abs()
amp = amp/2
amp = np.uint8(amp.numpy())
cv2.imwrite('amp_sub.jpg',amp)

y = torch.ifft(x,2)


y = y[0][:,:,0].numpy()+255
y = np.clip(y,0,255)
img_new = np.uint8(y)
cv2.imwrite('test_sub.jpg',img_new)