# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:55
# @Author  : zhoujun
from .model import *
from .loss import *


def get_model(config,):
    try:
        model_name = config['arch']['model_name']
    except:
        model_name = 'DualBranch'
    model_config = config['arch']['args']
    model = eval(model_name)
    return model(model_config)


def get_loss(config):
    loss_config = config['loss']['args']
    try:
        loss_name = config['arch']['loss_name']
    except:
        loss_name = 'SimpleLoss'
    loss = eval(loss_name)
    return loss(loss_config)