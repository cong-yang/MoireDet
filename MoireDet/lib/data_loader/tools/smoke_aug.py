import cv2
import numpy as np
import random

def  png_slim(png):
    density = png[:,:,-1]
    pos = np.where(density > 0)
    y_min,y_max = np.min(pos[0]),np.max(pos[0])+1
    x_min, x_max = np.min(pos[1]), np.max(pos[1]) + 1
    return png[y_min:y_max,x_min:x_max,:]


def smoke_aug(smoke,env_size = (720,720)):
    # The transform including rotate,resize,color transform

    img_h,img_w = smoke.shape[:2]

    angle = 90
    area_scales = [0.3,0.8]
    h_w_ratio_scale = 0.6

    color_min_ratio = 0.3
    color_max_ratio = 1.2

    horizon_flip_ratio = 0.3
    vertical_flip_ratio = 0.3

    color_change_ratio = 0.8


    if random.random() < horizon_flip_ratio:
        smoke = smoke[::-1,:,:]
    if random.random() < vertical_flip_ratio:
        smoke = smoke[:,::-1,:]

    rand_color_ratio = np.array([random.uniform(color_min_ratio, color_max_ratio),
                                 random.uniform(color_min_ratio, color_max_ratio),
                                 random.uniform(color_min_ratio, color_max_ratio),
                                 1])

    if random.random() < color_change_ratio:
        rand_color_ratio = rand_color_ratio[np.newaxis,np.newaxis,:]
    else:
        rand_color_ratio = rand_color_ratio*0.0 + random.uniform(0.5,1)


    rand_thelta = random.uniform(-angle,angle)
    rand_scale = 1

    M = cv2.getRotationMatrix2D((img_w * 0.5, img_h * 0.5), rand_thelta,
                                rand_scale)

    img = cv2.warpAffine(smoke, M, (img_w, img_h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0,0))
    img = np.uint(img)
    img = np.uint8(np.clip(img*rand_color_ratio,0,255))

    img = png_slim(img)

    img_h, img_w = img.shape[:2]
    area_scale = random.uniform(*area_scales)

    side_scale = (env_size[0]*env_size[1]*area_scale/(img_h*img_w))**0.5

    h_w_ratio = random.uniform(h_w_ratio_scale,1/h_w_ratio_scale)

    img_h = img_h*side_scale*h_w_ratio
    img_w = img_w*side_scale/h_w_ratio

    img = cv2.resize(img,(int(img_w),int(img_h)))

    return img


def get_big_smoke(smoke,pos,env_size = (720,720)):
    x_ratio,y_ratio= pos
    env_h,env_w = env_size[0],env_size[1]
    smoke_h,smoke_w = smoke.shape[:2]

    x,y = int(x_ratio*env_w+smoke_w / 2),int(y_ratio*env_h+smoke_h/2)
    big_smoke = np.zeros((env_h + smoke_h, env_w + smoke_w, 4))

    big_smoke[y-int(smoke_h/2):y+int((smoke_h+1)/2),
    x - int(smoke_w / 2):x + int((smoke_w + 1) / 2),
    :] = smoke[:,:,:]

    big_smoke = big_smoke[int(smoke_h / 2):int(smoke_h / 2) + env_h, \
          int(smoke_w / 2):int(smoke_w / 2) + env_w, \
          :]

    return big_smoke


def get_big_aug_smoke(smoke,env_size= (720,720)):
    smoke = smoke_aug(smoke,env_size)
    pos = [random.uniform(0,1),random.uniform(0,1)]
    return get_big_smoke(smoke,pos,env_size)

