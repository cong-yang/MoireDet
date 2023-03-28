# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:55
# @Author  : zhoujun
from .model import Model
from .loss import three_part_loss


def get_model(config):
    model_config = config['arch']['args']
    return Model(model_config)

def get_loss(config):
    weights = config['loss']['args']['weights']
    return three_part_loss(weights)
