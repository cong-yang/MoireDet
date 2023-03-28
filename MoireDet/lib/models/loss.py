# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:56
# @Author  : zhoujun

import sys
sys.path.append('./')
sys.path.insert(0,'/data/zhenyu.yang/modules')
sys.path.insert(0,'/home/users/zhenyu.yang/data/env/transformer_related/')

import itertools
import torch
from torch import nn
import numpy as np
import cv2
import torch.nn.functional as F


class wing_loss(nn.Module):
    def __init__(self,  w=20.0, epsilon=2.0):
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.constant = w - w * np.log(1 + w / epsilon)
        # print(self.w, self.epsilon, self.constant)

    def forward(self, prediction, gt):
        diff = torch.abs(prediction - gt)
        loss = torch.where(diff < self.w,
                           self.w * torch.log(1 + diff / self.epsilon),
                           diff - self.constant)

        loss = loss.view(-1)

        # diff = diff.view(-1)
        # loss = torch.norm(diff)

        return torch.mean(loss)


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



class direction_loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.kernel_size = 7
        filters = generate_filters(self.kernel_size )
        conv_weighs = [torch.FloatTensor(v).unsqueeze(0).unsqueeze(0) for v in filters]
        self.conv_weights = [nn.Parameter(data=v, requires_grad=False) for v in conv_weighs]

        # print(self.w, self.epsilon, self.constant)

        self.l1_loss = torch.nn.L1Loss()


    def forward(self, prediction, gt):
        loss_list = []
        for conv_weight in self.conv_weights:
            conv_weight = conv_weight.to(prediction.device)
            conv_pre = F.conv2d(prediction, conv_weight, padding=(self.kernel_size-1)//2)
            conv_gt = F.conv2d(gt, conv_weight, padding=(self.kernel_size-1)//2)

            temp_loss = self.l1_loss(conv_pre,conv_gt)


            loss_list.append(torch.mean(temp_loss))

        # loss = sum(loss_list)/len(loss_list)

        return sum(loss_list)/len(loss_list)


class tangent_loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.kernel_size = 7
        filters = generate_filters(self.kernel_size )
        conv_weighs = [torch.FloatTensor(v).unsqueeze(0).unsqueeze(0) for v in filters]
        self.conv_weights = [nn.Parameter(data=v, requires_grad=False) for v in conv_weighs]

        # print(self.w, self.epsilon, self.constant)

        self.l1_loss = torch.nn.L1Loss()


    def forward(self, prediction, gt):
        conv_pres = []
        conv_gts = []
        for conv_weight in self.conv_weights:
            conv_weight = conv_weight.to(prediction.device)
            conv_pre = F.conv2d(prediction, conv_weight, padding=(self.kernel_size-1)//2)
            conv_gt = F.conv2d(gt, conv_weight, padding=(self.kernel_size-1)//2)

            conv_pres.append(conv_pre)
            conv_gts.append(conv_gt)

        conv_pres = torch.stack(conv_pres,dim=-1).view(-1,len(self.conv_weights))
        conv_gts = torch.stack(conv_gts,dim=-1).view(-1,len(self.conv_weights))

        max_direcion = torch.argmax(conv_gts,1,keepdim = True)
        min_direction = torch.argmax(-conv_gts,1,keepdim = True)

        max_loss = self.l1_loss(torch.gather(conv_pres,1, max_direcion),torch.gather(conv_gts,1, max_direcion))
        min_loss = self.l1_loss(torch.gather(conv_pres,1, min_direction),torch.gather(conv_gts,1, min_direction))

        loss = torch.mean(max_loss) + torch.mean(min_loss)


        return loss


class coarse_tangent_loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.kernel_size = 7
        filters = generate_filters(self.kernel_size )
        conv_weighs = [torch.FloatTensor(v).unsqueeze(0).unsqueeze(0) for v in filters]
        self.conv_weights = [nn.Parameter(data=v, requires_grad=False) for v in conv_weighs]

        # print(self.w, self.epsilon, self.constant)

        self.l1_loss = torch.nn.L1Loss()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


    def forward(self, prediction, gt):
        conv_pres_max = []
        conv_pres_min = []
        conv_gts_max = []
        conv_gts_min = []

        for conv_weight in self.conv_weights:
            conv_weight = conv_weight.to(prediction.device)

            conv_pre = F.conv2d(prediction, conv_weight, padding=(self.kernel_size-1)//2)
            conv_gt = F.conv2d(gt, conv_weight, padding=(self.kernel_size-1)//2)

            conv_pres_max.append(self.maxpool(conv_pre))
            conv_pres_min.append(-self.maxpool(-conv_pre))
            conv_gts_max.append(self.maxpool(conv_gt))
            conv_gts_min.append(-self.maxpool(-conv_gt))

        conv_pres_max = torch.stack(conv_pres_max,dim=-1).view(-1,len(self.conv_weights))
        conv_pres_min = torch.stack(conv_pres_min,dim=-1).view(-1,len(self.conv_weights))
        conv_gts_max = torch.stack(conv_gts_max,dim=-1).view(-1,len(self.conv_weights))
        conv_gts_min = torch.stack(conv_gts_min,dim=-1).view(-1,len(self.conv_weights))



        max_direcion = torch.argmax(conv_gts_max,1,keepdim=True)
        min_direction = torch.argmax(-conv_gts_min,1,keepdim=True)

        max_loss = self.l1_loss(torch.gather(conv_pres_max, 1,max_direcion),torch.gather(conv_gts_max,1, max_direcion))
        min_loss = self.l1_loss(torch.gather(conv_pres_min, 1,min_direction),torch.gather(conv_gts_min,1, min_direction))

        loss = torch.mean(max_loss) + torch.mean(min_loss)


        return loss


class variance_loss(nn.Module):
    def __init__(self,kernel_size = 7,stride = 2):
        super().__init__()

        self.unfold = nn.Unfold(kernel_size=(kernel_size, kernel_size),padding=(kernel_size-1)//2,stride=stride)

        self.l1_loss = torch.nn.SmoothL1Loss()


    def forward(self, prediction, gt):
        eps = 0.1
        prediction = self.unfold(prediction)
        gt = self.unfold(gt)

        prediction = prediction - torch.mean(prediction,dim=1,keepdim = True)
        prediction = (torch.mean(prediction**2,dim=1)+eps)**0.5

        gt = gt - torch.mean(gt,dim=1,keepdim = True)
        gt = (torch.mean(gt**2,dim=1)+eps)**0.5

        loss = torch.mean(self.l1_loss(prediction,gt))
        return loss



class tangent_var_loss(nn.Module):
    def __init__(self,config=None):
        super().__init__()

        self.variance_loss = variance_loss()

        self.tangent_loss = tangent_loss()

    def forward(self, pred_moires,moires):
        weights = [1,1,0.4,0.5,0.6,0.7,0.8,0.9]
        weights = weights[:len(pred_moires)]
        all_loss = 0

        for weight,pred_moire in zip(weights,pred_moires):
            temp_loss = self.variance_loss(pred_moire,moires) + self.tangent_loss(pred_moire,moires)
            all_loss = temp_loss*weight + all_loss

        return all_loss


class coarse_tangent_var_loss(nn.Module):
    def __init__(self,config=None):
        super().__init__()

        self.variance_loss = variance_loss()

        self.tangent_loss = coarse_tangent_loss()

    def forward(self, pred_moires,moires):
        weights = [1,1,0.4,0.5,0.6,0.7,0.8,0.9]
        weights = weights[:len(pred_moires)]
        all_loss = 0

        for weight,pred_moire in zip(weights,pred_moires):
            temp_loss = self.variance_loss(pred_moire,moires) + self.tangent_loss(pred_moire,moires)
            all_loss = temp_loss*weight + all_loss

        return all_loss



class many_loss(nn.Module):
    def __init__(self,config=None):
        super().__init__()

        self.variance_loss = variance_loss()
        self.tangent_loss = coarse_tangent_loss()
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, pred_moires,moires):
        weights = [1,1,0.4,0.5,0.6,0.7,0.8,0.9]
        weights = weights[:len(pred_moires)]
        all_loss = 0

        for weight,pred_moire in zip(weights,pred_moires):
            temp_loss = self.variance_loss(pred_moire,moires) + self.tangent_loss(pred_moire,moires)
            temp_loss = torch.mean(self.l1_loss(pred_moire,moires)) + temp_loss*0.8
            all_loss = temp_loss*weight + all_loss

        return all_loss


class many_loss_var(nn.Module):
    def __init__(self,config=None):
        super().__init__()

        self.variance_loss = variance_loss()
        self.tangent_loss = coarse_tangent_loss()
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, pred_moires,moires):
        weights = [1,1,0.4,0.5,0.6,0.7,0.8,0.9]
        weights = weights[:len(pred_moires)]
        all_loss = 0

        for weight,pred_moire in zip(weights,pred_moires):
            temp_loss = self.variance_loss(pred_moire,moires)
            temp_loss = torch.mean(self.l1_loss(pred_moire,moires)) + temp_loss*0.8
            all_loss = temp_loss*weight + all_loss

        return all_loss


class many_loss_direction(nn.Module):
    def __init__(self,config=None):
        super().__init__()

        self.variance_loss = variance_loss()
        self.tangent_loss = coarse_tangent_loss()
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, pred_moires,moires):
        weights = [1,1,0.4,0.5,0.6,0.7,0.8,0.9]
        weights = weights[:len(pred_moires)]
        all_loss = 0

        for weight,pred_moire in zip(weights,pred_moires):
            temp_loss = self.tangent_loss(pred_moire,moires)
            temp_loss = torch.mean(self.l1_loss(pred_moire,moires)) + temp_loss*0.8
            all_loss = temp_loss*weight + all_loss

        return all_loss


class many_loss_l1(nn.Module):
    def __init__(self,config=None):
        super().__init__()

        self.variance_loss = variance_loss()
        self.tangent_loss = coarse_tangent_loss()
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, pred_moires,moires):
        weights = [1,1,0.4,0.5,0.6,0.7,0.8,0.9]
        weights = weights[:len(pred_moires)]
        all_loss = 0

        for weight,pred_moire in zip(weights,pred_moires):
            temp_loss = torch.mean(self.l1_loss(pred_moire,moires))
            all_loss = temp_loss*weight + all_loss

        return all_loss




class SimpleLoss(nn.Module):
    def __init__(self,config=None):
        super().__init__()

        self.l1_loss = torch.nn.L1Loss()

        self.direction_loss = direction_loss()

    def forward(self, pred_moires,moires):
        weights = [1,1,0.4,0.5,0.6,0.7,0.8,0.9]
        weights = weights[:len(pred_moires)]
        all_loss = 0

        for weight,pred_moire in zip(weights,pred_moires):
            temp_loss = torch.mean(self.l1_loss(pred_moire,moires)) + self.direction_loss(pred_moire,moires)
            all_loss = temp_loss*weight + all_loss

        return all_loss



if __name__ == '__main__':
    tangent_var = variance_loss()
    a = torch.zeros((3,1,300,300))
    b = torch.zeros((3, 1, 300, 300))
    print(tangent_var(a,b))
