import json
import os
import random

import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils import data
from PIL import Image
import torch
from .imgpath2dct import path2dct,img2dct
# from mxnet import gluon, nd

# ic15_root_dir = './data/'

ic15_train_data_dir = '/data/zhenyu/smoke/Min_smoke'

random.seed(123456)

"""
Author:Zhenyu Yang
Time:2019-11-25
Version:1.0

The program will normalize the data while loading the data here
"""


def get_img(img_path):
    try:
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
    except Exception as e:
        print(img_path)
        raise
    return img


def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_vertical_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=0).copy()
    return imgs


def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
        imgs[i] = img_rotation
    return imgs


def scale(img, long_size=2240):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img


def random_scale(img, min_size, scale=None):
    if scale is None:
        h, w = img.shape[0:2]
        scale = 1
        if min(h, w) > 800:
            scale = 800 / min(h, w)
        scale *= random.uniform(0.8, 1.5)

        if min(h, w) * scale <= min_size:
            scale = (min_size + 10) * 1.0 / min(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img


def imgs_random_scale(imgs, min_size):
    img = imgs[0]
    h, w = img.shape[0:2]
    scale = 1
    if min(h, w) > 800:
        scale = 800 / min(h, w)
    scale *= random.uniform(0.8, 1.5)

    if min(h, w) * scale <= min_size:
        scale = (min_size + 10) * 1.0 / min(h, w)
    ans = []
    for img in imgs:
        ans.append(random_scale(img, min_size, scale))
    return ans


def random_crop(imgs, img_size):
    h, w = imgs[0].shape[0:2]
    th, tw = img_size
    if w == tw and h == th:
        return imgs

    if random.random() > 1.0 / 10.0 and np.max(imgs[1]) > 0:
        tl = np.min(np.where(imgs[1] > 0), axis=1)
        tl[tl < 0] = 0
        tl[0] = min(tl[0], h - th)
        tl[1] = min(tl[1], w - tw)
        br = np.max(np.where(imgs[1] > 0), axis=1) - img_size
        br[br < 0] = 0
        br[0] = min(br[0], h - th)
        br[1] = min(br[1], w - tw)
        if (br < tl).all():
            i = random.randint(br[0], tl[0])
            j = random.randint(br[1], tl[1])
        else:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)

    else:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

    # return i, j, th, tw
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw, :]
        else:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw]
    return imgs


def judge_point(bbox):
    bbox_x = bbox[..., 0]
    bbox_y = bbox[..., 1]
    x_min = np.min(bbox_x)
    y_min = np.min(bbox_y)
    x_max = np.max(bbox_x)
    y_max = np.max(bbox_y)
    return x_min, y_min, x_max, y_max


def change_color(img, boxes, factor=0.7, thresh=160):
    if random.random() > factor and boxes.shape[0] > 0:
        box = judge_point(boxes)
        crop = img[box[1]:box[3], box[0]:box[2], :]
        a = crop[..., 0] > thresh
        b = crop[..., 1] > thresh
        c = crop[..., 2] > thresh
        d = a & b & c
        crop[d] = 255 - crop[d]
    return img


def gasuss_noise(image, mean=0.01, var=0.03, factor=0.7):
    if random.random() > factor:
        image = np.array(image / 255, dtype=float)
        noise = np.random.normal(mean, var, image.shape)
        out = image + noise
        low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        out = np.uint8(out * 255)
    else:
        out = image
    return out


def add_noise(img):
    img = gasuss_noise(img)
    return img


def blur(img, factor=0.7):
    if random.random() > factor:
        img = cv2.GaussianBlur(img, (3, 3), 2)
    return img


def img_pad(img):
    h, w = img.shape[0:2]
    l = max(h, w)
    img = np.pad(img, (((l - h) // 2 + (l - h) % 2, (l - h) // 2),
                       ((l - w) // 2 + (l - w) % 2, (l - w) // 2), (0, 0)),
                 'constant')
    return img


class IC15Loader(data.Dataset):
    def __init__(self, is_transform=False, img_size=None,
                 data_dir=ic15_train_data_dir):
        self.is_transform = is_transform

        self.img_size = img_size if \
            (img_size is None or isinstance(img_size, tuple)) \
            else (img_size, img_size)

        self.data_dir = data_dir

        self.img_file = os.listdir(self.data_dir)
        self.img_file = [os.path.join(self.data_dir,v) for v in self.img_file if v.endswith('.png')]


        self.channel_std = [10 for _ in range(192)]
        self.channel_std[0] = 200

        self.channel_mean = [0 for _ in range(192)]

        temp_file = []
        for file in self.img_file:
            try:
                img_png = cv2.imread(file,cv2.IMREAD_UNCHANGED)
                img = img_png[:, :, :3]

            except:
                continue
            temp_file.append(file)
        self.img_file = temp_file


    def __len__(self):
        return len(self.img_file)

    def _str2np(self, str):
        return np.array(json.loads(str))

    def __getitem__(self, index):



        try:
            img_png = cv2.imread(self.img_file[index],cv2.IMREAD_UNCHANGED)

            img = img_png[:,:,:3]
        except:
            debug_here = 0
            img_png = cv2.imread(self.img_file[index], cv2.IMREAD_UNCHANGED)

            img = img_png[:, :, :3]
        density = img_png[:,:,-1]


        if self.is_transform:
            # img = img_pad(img)
            # img=sp_noise(img)
            img = blur(img)
            img = add_noise(img)
            # img=gasuss_noise(img)

        if self.is_transform:
            imgs = [img, density]
            imgs = imgs_random_scale(imgs, self.img_size[0])
            imgs = random_crop(imgs, self.img_size)
            imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs)

            img, density = imgs[0],imgs[1]

        img = img[:, :, ::-1]
        img = Image.fromarray(img)

        if self.is_transform:

            img = transforms.ColorJitter(brightness=0.5, contrast=0.5,
                                         saturation=0.5, hue=0.3)(img)
            # img = gluon.data.vision.transforms.RandomColorJitter(
            #     brightness=0.5, contrast=0.5, saturation=0.3)(img)



        dct_size = self.img_size[0]//4
        dct_y, dct_cb, dct_cr = img2dct(np.asarray(img))

        dct_y = cv2.resize(dct_y, dsize=(dct_size,dct_size))
        dct_cb = cv2.resize(dct_cb, dsize=(dct_size, dct_size))
        dct_cr = cv2.resize(dct_cr, dsize=(dct_size, dct_size))

        dct = np.concatenate([dct_y,dct_cb,dct_cr],axis=-1)

        # dct_k = dct.reshape(-1,192)
        # dct_k_mean = np.mean(dct_k,0)
        # dct_std = np.std(dct_k,0)

        img = torch.from_numpy(dct.transpose((2, 0, 1))).float()

        img = transforms.Normalize(mean=self.channel_mean,
                                   std=self.channel_std)(img)

        # img = transforms.ToTensor()(dct)
        # img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)


        density = density.astype('float32')/255

        mask = np.where(density > 0.3, np.ones_like(density),
                        np.zeros_like(density))

        training_mask = np.ones_like(density)


        # img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        density = torch.from_numpy(density).float()
        training_mask = torch.from_numpy(training_mask).float()
        return img,mask, density, training_mask

# temp = IC15Loader(True,(1024,1024))
# img = temp[1]
# a = 1