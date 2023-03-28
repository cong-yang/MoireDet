# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:57
# @Author  : zhoujun

import torch
from torch import nn
import torch.nn.functional as F
from .modules import *

backbone_dict = {'resnet18': {'models': resnet18, 'out': [64, 128, 256, 512]},
                 'resnet34': {'models': resnet34, 'out': [64, 128, 256, 512]},
                 'resnet50': {'models': resnet50, 'out': [256, 512, 1024, 2048]},
                 'resnet101': {'models': resnet101, 'out': [256, 512, 1024, 2048]},
                 'resnet152': {'models': resnet152, 'out': [256, 512, 1024, 2048]},
                 'resnext50_32x4d': {'models': resnext50_32x4d, 'out': [256, 512, 1024, 2048]},
                 'resnext101_32x8d': {'models': resnext101_32x8d, 'out': [256, 512, 1024, 2048]},
                 'resnet18_dct': {'models': resnet18_dct, 'out': [64, 128, 256, 512]},
                 'resnet34_dct': {'models': resnet34_dct, 'out': [64, 128, 256, 512]}
                 }

segmentation_head_dict = {'FPN': FPN, 'FPEM_FFM': FPEM_FFM}


# 'MobileNetV3_Large': {'models': MobileNetV3_Large, 'out': [24, 40, 160, 160]},
# 'MobileNetV3_Small': {'models': MobileNetV3_Small, 'out': [16, 24, 48, 96]},
# 'shufflenetv2': {'models': shufflenet_v2_x1_0, 'out': [24, 116, 232, 464]}}

class TripleInterActionModule(nn.Module):

    def __init__(self, inplanes=7,channels = 64,outplanes = 1,norm_layer=None):
        super(TripleInterActionModule, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d


        self.conv1 = conv3x3(inplanes, channels)
        self.bn1 = norm_layer(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels, outplanes)
        self.bn2 = norm_layer(outplanes)


    def forward(self, x1,x2,x3):
        x = torch.cat([x1,x2,x3],dim=1)

        # identity = x1

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # out += identity
        out = self.relu(out)

        return out


class DualInterActionModule(nn.Module):

    def __init__(self, inplanes,norm_layer=None):
        super(DualInterActionModule, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        planes = inplanes
        self.conv1 = conv3x3(inplanes*2, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)


    def forward(self, x1,x2):
        x = torch.cat([x1,x2],dim=1)

        identity = x1

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class DualBranch(nn.Module):
    def __init__(self, model_config: dict):
        super().__init__()
        backbone = model_config['backbone']
        pretrained = model_config['pretrained']
        segmentation_head = model_config['segmentation_head']

        self.repeat_times = model_config.get('repeat_times',3)
        channels = model_config.get('channels', 128)
        fpem_repeat = model_config.get('fpem_repeat', 1)

        assert backbone in backbone_dict, 'backbone must in: {}'.format(backbone_dict)
        assert segmentation_head in segmentation_head_dict, 'segmentation_head must in: {}'.format(segmentation_head_dict)

        backbone_model, backbone_out = backbone_dict[backbone]['models'], backbone_dict[backbone]['out']


        self.moire_backbone = backbone_model(pretrained=False)
        self.moire_segmentation_head = FPEM_FFM(backbone_out,fpem_repeat=fpem_repeat,channels = channels)
        self.moire_interact = DualInterActionModule(channels)
        self.moire_head = nn.Sequential(conv3x3(channels,1))


        self.name = '{}_{}'.format(backbone, segmentation_head)

    def forward(self, x,pure_moire = None):
        _, _, H, W = x.size()


        moire_fea = self.moire_backbone(x)
        bi_fpn_fea = self.moire_segmentation_head(moire_fea)
        moire_density = self.moire_head(F.interpolate(bi_fpn_fea, size=(H, W), mode='bilinear', align_corners=True))
        moire_feas = moire_fea + [bi_fpn_fea]
        fea_loss = 0
        if pure_moire is not None:
            pure_moire_fea = self.moire_backbone(pure_moire)
            pure_bi_fpn_fea = self.moire_segmentation_head(pure_moire_fea)
            pure_moire_fea = pure_moire_fea + [pure_bi_fpn_fea]
            pure_moire_fea = [v.detach() for v in pure_moire_fea]
            pure_moire_density = self.moire_head(F.interpolate(pure_bi_fpn_fea, size=(H, W), mode='bilinear',align_corners=True))
            fea_loss = 0
            for i in range(len(moire_feas)):
                fea_loss = torch.mean(torch.abs(moire_feas[i]-pure_moire_fea[i])) * (i+1)**0.5 + fea_loss

            return [moire_density,pure_moire_density],fea_loss

        return [moire_density] ,fea_loss



class DualBranchWithImg(nn.Module):
    def __init__(self, model_config: dict):
        super().__init__()
        backbone = model_config['backbone']
        pretrained = model_config['pretrained']
        segmentation_head = model_config['segmentation_head']

        self.repeat_times = model_config.get('repeat_times',3)
        channels = model_config.get('channels', 128)
        fpem_repeat = model_config.get('fpem_repeat', 1)

        assert backbone in backbone_dict, 'backbone must in: {}'.format(backbone_dict)
        assert segmentation_head in segmentation_head_dict, 'segmentation_head must in: {}'.format(segmentation_head_dict)

        backbone_model, backbone_out = backbone_dict[backbone]['models'], backbone_dict[backbone]['out']


        ouput_channel = 1
        self.moire_backbone = backbone_model(pretrained=False)
        self.moire_segmentation_head = FPEM_FFM(backbone_out,fpem_repeat=fpem_repeat,channels = channels,ouput_channel=channels)
        self.moire_interact = TripleInterActionModule()
        self.moire_head = nn.Sequential(conv3x3(channels,1))


        ouput_channel = 3
        self.recover_backbone = backbone_model(pretrained=False)
        self.recover_segmentation_head = FPEM_FFM(backbone_out,fpem_repeat=fpem_repeat,channels = channels,ouput_channel=channels)
        self.recover_interact = TripleInterActionModule(outplanes=3)
        self.recover_head = nn.Sequential(conv3x3(channels,3))


        self.name = '{}_{}'.format(backbone, segmentation_head)

    def forward(self, x):
        _, _, H, W = x.size()

        moire = self.moire_head(self.moire_segmentation_head(self.moire_backbone(x)))
        recover = self.recover_head(self.recover_segmentation_head(self.recover_backbone(x)))

        moire = F.interpolate(moire, size=(H, W), mode='bilinear', align_corners=True)
        recover = F.interpolate(recover, size=(H, W), mode='bilinear', align_corners=True)

        moires = [moire]
        recovers = [recover]


        for i in range(self.repeat_times):
            temp_moire = self.moire_interact(moires[-1],recovers[-1],x)
            temp_recove = self.recover_interact(recovers[-1],moires[-1],x)

            moires.append(temp_moire)
            recovers.append(temp_recove)


        return moires,recovers



class Model(nn.Module):
    def __init__(self, model_config: dict):
        """
        PANnet
        :param model_config: 模型配置
        """
        super().__init__()
        backbone = model_config['backbone']
        pretrained = model_config['pretrained']

        self.is_dct = model_config['is_dct']

        segmentation_head = model_config['segmentation_head']

        assert backbone in backbone_dict, 'backbone must in: {}'.format(backbone_dict)
        assert segmentation_head in segmentation_head_dict, 'segmentation_head must in: {}'.format(
            segmentation_head_dict)

        backbone_model, backbone_out = backbone_dict[backbone]['models'], backbone_dict[backbone]['out']
        self.backbone = backbone_model(pretrained=pretrained)
        self.segmentation_head = segmentation_head_dict[segmentation_head](backbone_out, **model_config)
        self.name = '{}_{}'.format(backbone, segmentation_head)

    def forward(self, x):
        _, _, H, W = x.size()
        if self.is_dct :
            H = H*4
            W = W*4

        backbone_out = self.backbone(x)
        segmentation_head_out = self.segmentation_head(backbone_out)
        y = F.interpolate(segmentation_head_out, size=(H, W), mode='bilinear', align_corners=True)
        return y


if __name__ == '__main__':
    device = torch.device('cpu')
    x = torch.zeros(1, 3, 640, 640).to(device)

    model_config = {
        'backbone': 'resnet18',
        'fpem_repeat': 4,  # fpem模块重复的次数
        'pretrained': False,  # backbone 是否使用imagesnet的预训练模型
        'result_num':7,
        'segmentation_head': 'FPEM_FFM'  # 分割头，FPN or FPEM
    }
    model = Model(model_config=model_config).to(device)
    y = model(x)
    print(y.shape)
    print(model)
    # torch.save(model.state_dict(), 'PAN.pth')
