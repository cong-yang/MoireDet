import json
import os
import random

import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils import data
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
from .tools.smoke_aug import get_big_aug_smoke
# from mxnet import gluon, nd

# ic15_root_dir = './data/'


train_background_dir = '/data/zhenyu/smoke/origin/train/background'
train_pure_smoke_dir = '/data/zhenyu/smoke/origin/train/pure_smoke'


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


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        x = F.conv2d(x.unsqueeze(0), self.weight, padding=2,
                     groups=self.channels)

class IC15Loader(data.Dataset):
    def __init__(self,is_transform=False, img_size=None,
                 background_dir=train_background_dir,pure_smoke_dir = train_pure_smoke_dir,train_val = 'train'):
        self.is_transform = is_transform

        self.train_val = train_val

        self.img_size = img_size if \
            (img_size is None or isinstance(img_size, tuple)) \
            else (img_size, img_size)

        self.background_dir = background_dir
        self.pure_smoke_dir = pure_smoke_dir

        self.img_file = os.listdir(self.background_dir)
        self.img_file = [os.path.join(self.background_dir,v) for v in self.img_file]

        self.smoke_file = os.listdir(self.pure_smoke_dir)
        self.smoke_file = [os.path.join(self.pure_smoke_dir,v) for v in self.smoke_file]


        self.read_img = lambda file_path :cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),cv2.IMREAD_UNCHANGED)

        self.guassian = GaussianBlurConv(channels=1)


    def __len__(self):
        return len(self.img_file)

    def _str2np(self, str):
        return np.array(json.loads(str))

    def __getitem__(self, index):

        random_index = random.randrange(0, len(self.smoke_file))

        smoke = self.read_img(self.smoke_file[random_index])
        env = cv2.imread(self.img_file[index])



        smoke = get_big_aug_smoke(smoke,self.img_size)

        density = smoke[:,:,-1].astype('float32')/255
        alpha = smoke[:,:,-1:].astype('float32')/255

        smoke = smoke[:, :, :3]


        if self.is_transform:
            # img = img_pad(img)
            # img=sp_noise(img)
            img = blur(env)
            img = add_noise(env)
            # img=gasuss_noise(img)

        if self.is_transform:
            imgs = [env]
            imgs = imgs_random_scale(imgs, self.img_size[0])
            imgs = random_crop(imgs, self.img_size)
            imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs)

            env = imgs[0]


        env = Image.fromarray(env)
        smoke = Image.fromarray(np.uint8(smoke))

        if self.is_transform:
            env = transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                         saturation=0.2, hue=0.1)(env)

            smoke = transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                         saturation=0.2, hue=0.1)(smoke)
            # img = gluon.data.vision.transforms.RandomColorJitter(
            #     brightness=0.5, contrast=0.5, saturation=0.3)(img)


        smoke_alpha = np.array(smoke)*alpha
        env_alpha = np.array(env)*(1-alpha)
        img =  env_alpha + smoke_alpha




        origin_env = np.array(env)
        origin_img = img.copy()

        # sim = np.abs(np.mean(smoke_alpha-env_alpha,2))
        # sim[density < 0.05] = 0
        # sim = sim/(density+0.000001)
        # ramdom_num = random.randint(0,1000)
        # cv2.imwrite('{}.jpg'.format(ramdom_num),np.uint8(img))
        # cv2.imwrite('{}_1.jpg'.format(ramdom_num), np.uint8(alpha*255))
        # cv2.imwrite('{}_2.jpg'.format(ramdom_num), np.uint8(sim))


        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        smoke_alpha = transforms.ToTensor()(smoke_alpha)

        env = transforms.ToTensor()(env)

        # eps = 1e-4
        # sim = (torch.sum(torch.pow(env*smoke_alpha,2),0)+eps)/(torch.sum(torch.pow(env,2),0)*torch.sum(torch.pow(smoke_alpha,2),0)+eps)
        # sim = torch.pow(sim,0.5)
        #
        #
        #
        # no_smoke_weight = 0.5
        # smoke_weight = 1.2
        #
        # no_smoke_mask = no_smoke_weight*torch.ones_like(sim)
        # smoke_mask = (1-sim)*smoke_weight

        no_smoke_mask = 0.2*torch.ones(self.img_size)
        smoke_mask = torch.ones(self.img_size)

        training_mask = torch.where(smoke_alpha[0] == 0,no_smoke_mask,smoke_mask)


        smoke_alpha = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(smoke_alpha)
        env = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(env)

        # img = torch.from_numpy(img).float()
        density = torch.from_numpy(density).float()
        density = density.unsqueeze(0)

        if self.train_val == 'val':
            origin_env = torch.from_numpy(origin_env).float()
            origin_img = torch.from_numpy(origin_img).float()
            return img,smoke_alpha,env,density, training_mask,origin_env,origin_img

        return img,smoke_alpha,env,density, training_mask
