# -*- coding: UTF-8 -*-
'''=================================================
@Author ：zhenyu.yang
@Date   ：2021/1/8 3:24 PM
=================================================='''
import sys
sys.path.insert(0,'/data/site-packages/site-packages')
sys.path.insert(0,'/home/zhenyu/env/pytorch/')
# sys.path.insert(0,'./')
sys.path.insert(0,'/home/zhenyu/env/transformer_related')


import torch
from performer_pytorch import Performer

model = Performer(
    dim = 512,
    depth = 3,
    heads = 8,
    causal = False
)

x = torch.randn(1024, 64, 512)
print(model(x).size())