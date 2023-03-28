# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:57
# @Author  : zhoujun

import torch
from torch import nn
import torch.nn.functional as F
from .modules import *
from performer_pytorch import Performer

backbone_dict = {'resnet6': {'models': resnet6, 'out': [64]},
                 'resnet10': {'models': resnet10, 'out': [64, 128]},
                 'resnet18': {'models': resnet18, 'out': [64, 128, 256, 512]},
                 'resnet34': {'models': resnet34, 'out': [64, 128, 256, 512]},
                 'resnet50': {'models': resnet50,
                              'out': [256, 512, 1024, 2048]},
                 'resnet101': {'models': resnet101,
                               'out': [256, 512, 1024, 2048]},
                 'resnet152': {'models': resnet152,
                               'out': [256, 512, 1024, 2048]},
                 'resnext50_32x4d': {'models': resnext50_32x4d,
                                     'out': [256, 512, 1024, 2048]},
                 'resnext101_32x8d': {'models': resnext101_32x8d,
                                      'out': [256, 512, 1024, 2048]},
                 'resnet18_dct': {'models': resnet18_dct,
                                  'out': [64, 128, 256, 512]},
                 'resnet34_dct': {'models': resnet34_dct,
                                  'out': [64, 128, 256, 512]}
                 }

segmentation_head_dict = {'FPN': FPN, 'FPEM_FFM': FPEM_FFM}




def upsample_flow(self, flow, mask):
    """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
    N, _, H, W = flow.shape
    mask = mask.view(N, 1, 9, 8, 8, H, W)
    mask = torch.softmax(mask, dim=2)

    up_flow = F.unfold(8 * flow, [3, 3], padding=1)
    up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(N, 2, 8 * H, 8 * W)


#
# self.mask = nn.Sequential(
#     nn.Conv2d(128, 256, 3, padding=1),
#     nn.ReLU(inplace=True),
#     nn.Conv2d(256, 64 * 9, 1, padding=0))


class UpScale(nn.Module):
    def __init__(self, inchannels=128, feature_channels=128, outchannels=32,
                 fx=4, fy=4):
        super().__init__()

        self.fx = fx
        self.fy = fy
        self.outchannels = outchannels

        self.mask = nn.Sequential(
            nn.Conv2d(inchannels, feature_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, fx * fy * 9, 1, padding=0))

        self.feafusion = nn.Sequential(
            nn.Conv2d(inchannels, feature_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, outchannels, 1, padding=0))

    def forward(self, x):
        N, _, H, W = x.shape

        fea = self.feafusion(x)

        mask = self.mask(x)
        mask = mask.view(N, 1, 9, self.fy, self.fx, H, W)
        mask = torch.softmax(mask, dim=2)

        fea = F.unfold(fea, [3, 3], padding=1)
        fea = fea.view(N, self.outchannels, 9, 1, 1, H, W)

        fea = torch.sum(mask * fea, dim=2)
        fea = fea.permute(0, 1, 4, 2, 5, 3)
        return fea.reshape(N, self.outchannels, self.fy * H, self.fx * W)



class TripleBranchWithSpecificConv(nn.Module):
    def __init__(self, model_config: dict):
        super().__init__()
        backbone = model_config['backbone']
        pretrained = model_config['pretrained']
        segmentation_head = model_config['segmentation_head']

        self.repeat_times = model_config.get('repeat_times', 3)
        channels = model_config.get('channels', 128)
        ouput_channel = model_config.get('ouput_channel', 126)

        self.is_light = model_config.get('is_light', False)
        self.user_layers = (-1, -2)[self.is_light]

        self.ouput_channel = model_config.get('ouput_channel', 4)
        self.kernel_size = model_config.get('kernel_size', 5)

        moire_ouput_channel = self.kernel_size ** 2 * self.ouput_channel

        fpem_repeat = model_config.get('fpem_repeat', 1)

        assert backbone in backbone_dict, 'backbone must in: {}'.format(
            backbone_dict)
        assert segmentation_head in segmentation_head_dict, 'segmentation_head must in: {}'.format(
            segmentation_head_dict)

        backbone_model, backbone_out = backbone_dict[backbone]['models'], \
                                       backbone_dict[backbone]['out']

        self.moire_backbone = backbone_model(pretrained=False,
                                             replace_stride_with_dilation=[
                                                 True, True, False])

        self.unfold = nn.Unfold(
            kernel_size=(self.kernel_size, self.kernel_size),
            padding=(self.kernel_size - 1) // 2)

        self.moire_fusion = nn.Sequential(
            nn.Conv2d(backbone_out[self.user_layers], ouput_channel, (1, 1)),
            nn.BatchNorm2d(ouput_channel)
        )

        self.attention_backbone = backbone_model(pretrained=True)
        self.attention_head = FPEM_FFM(backbone_out, fpem_repeat=fpem_repeat,
                                       channels=channels, ouput_channel=1)

        self.with_pos_embedding = True
        nhead = 2
        ouput_channel = ouput_channel + 2
        # self.position_embedding = nn.Parameter(torch.rand(1000,nhead))

        self.performer = Performer(dim=ouput_channel, depth=3, heads=nhead,
                                   causal=True)

        self.upscale = UpScale(ouput_channel * 2 - 2, 128, 32, 4, 4)

        self.detail_conv = nn.Sequential(
            nn.ReLU(inplace=False),
            conv1x1(32, 64),
            nn.ReLU(inplace=True),
            conv1x1(64, moire_ouput_channel),
        )

        self.detail_bias = nn.Sequential(
            nn.ReLU(inplace=False),
            conv1x1(32, 64),
            nn.ReLU(inplace=True),
            conv1x1(64, self.ouput_channel * 3),
        )

        self.middle_fea = nn.Sequential(
            nn.ReLU(inplace=False),
            conv1x1(32, 64),
            nn.ReLU(inplace=True),
            conv1x1(64, self.ouput_channel * 3),
        )

        self.moire_head = nn.Sequential(
            nn.ReLU(inplace=True),
            conv1x1(self.ouput_channel * 6, 32),
            nn.ReLU(inplace=True),
            conv1x1(32, 1),
        )

        max_len = 100
        x_list = (torch.arange(max_len).float() / max_len).view(1, 1,
                                                                max_len).repeat(
            1, max_len, 1)
        y_list = (torch.arange(max_len).float() / max_len).view(1, max_len,
                                                                1).repeat(1, 1,
                                                                          max_len)
        self.pos = torch.cat([x_list, y_list], dim=0).view(1, 2, max_len,
                                                           max_len)

        self.name = '{}_{}'.format(backbone, segmentation_head)

    def forward(self, x, pure_moire=None):
        N, _, img_H, img_W = x.size()


        moire_fea = self.moire_backbone(x)[self.user_layers]
        moire_fea = self.moire_fusion(moire_fea)
        B, C, MH, MW = moire_fea.size()

        attention_fea = self.attention_backbone(x)
        attention_fea = self.attention_head(attention_fea)
        attention = torch.sigmoid(attention_fea)
        attention = F.interpolate(attention, size=(MH, MW), mode='bilinear',
                                  align_corners=True)

        moire_fea = moire_fea * attention
        B, C, H, W = moire_fea.size()

        # moire_fea_unfold = F.unfold(moire_fea,(10,10),stride=10)
        # moire_fea_unfold = moire_fea_unfold.view(B,C,100,H//10,W//10)
        # moire_fea_unfold = moire_fea_unfold.permute(2,0,3,4,1).contiguous()

        dst_size = (H // 2, W // 3)
        # if self.is_light:
        #     dst_size = (H // 4, W // 4)

        min_trans_fea = F.interpolate(moire_fea, size=dst_size,
                                      mode='bilinear', align_corners=True)
        pos = self.pos.to(x.device)
        pos = F.interpolate(pos, size=dst_size, mode='bilinear',
                            align_corners=True)
        min_trans_fea = torch.cat([min_trans_fea, pos.repeat(B, 1, 1, 1)],
                                  dim=1)

        B, C, minH, minW = min_trans_fea.size()
        min_trans_fea = min_trans_fea.view(B, C, -1).permute(2, 0,
                                                             1).contiguous()

        # if self.with_pos_embedding:
        #     S, B, C = trans_fea.size()
        #     trans_fea = torch.cat([trans_fea,self.position_embedding[:S].unsqueeze(1).repeat(1,B,1)],dim=2)

        min_trans_fea = self.performer(min_trans_fea).permute(1, 2,
                                                              0).contiguous().view(
            B, -1, minH, minW)
        min_trans_fea = F.interpolate(min_trans_fea, size=(H, W),
                                      mode='bicubic', align_corners=True)
        trans_fea = torch.cat([moire_fea, min_trans_fea], dim=1)

        trans_fea = self.upscale(trans_fea)

        _, _, h, w = trans_fea.size()

        if h != img_H or w != img_W:
            trans_fea = F.interpolate(trans_fea, size=(img_H, img_W),
                                      mode='bicubic', align_corners=True)

        detail_conv = self.detail_conv(trans_fea)

        detail_conv = detail_conv.view(B, 1, self.ouput_channel,self.kernel_size ** 2, img_H, img_W)
        unfold_img = self.unfold(x) # x is input image
        unfold_img = unfold_img.view(N, 3, 1, -1, img_H, img_W)
        detail_conv = torch.softmax(detail_conv, dim=3)

        # detail_conv = torch.tanh(detail_conv)
        detail_fea = torch.sum(unfold_img * detail_conv, dim=3).view(B,
                                                                     self.ouput_channel * 3,
                                                                     img_H,
                                                                     img_W)

        detail_bias = self.detail_bias(trans_fea)
        detail_bias = detail_bias.view(B, self.ouput_channel * 3, img_H, img_W)

        detail_fea = detail_fea + detail_bias

        middle_fea = self.middle_fea(trans_fea)

        fea = torch.cat([detail_fea, middle_fea], dim=1)
        moire_density = self.moire_head(fea)
        # moire_density = self.moire_head(F.interpolate(trans_fea, size=(img_H, img_W), mode='bicubic', align_corners=True))
        fea_loss = torch.mean(moire_density) * 0
        return [moire_density], fea_loss

        # return [moire_density,attention,detail_conv,detail_fea], fea_loss


class TripleBranchWithSpecificConvNoPer(nn.Module):
    def __init__(self, model_config: dict):
        super().__init__()
        backbone = model_config['backbone']
        pretrained = model_config['pretrained']
        segmentation_head = model_config['segmentation_head']

        self.repeat_times = model_config.get('repeat_times', 3)
        channels = model_config.get('channels', 128)
        ouput_channel = model_config.get('ouput_channel', 126)

        self.is_light = model_config.get('is_light', False)
        self.user_layers = (-1, -2)[self.is_light]

        self.ouput_channel = model_config.get('ouput_channel', 4)
        self.kernel_size = model_config.get('kernel_size', 5)

        moire_ouput_channel = self.kernel_size ** 2 * self.ouput_channel

        fpem_repeat = model_config.get('fpem_repeat', 1)

        assert backbone in backbone_dict, 'backbone must in: {}'.format(
            backbone_dict)
        assert segmentation_head in segmentation_head_dict, 'segmentation_head must in: {}'.format(
            segmentation_head_dict)

        backbone_model, backbone_out = backbone_dict[backbone]['models'], \
                                       backbone_dict[backbone]['out']

        self.moire_backbone = backbone_model(pretrained=False,
                                             replace_stride_with_dilation=[
                                                 True, True, False])

        self.unfold = nn.Unfold(
            kernel_size=(self.kernel_size, self.kernel_size),
            padding=(self.kernel_size - 1) // 2)

        self.moire_fusion = nn.Sequential(
            nn.Conv2d(backbone_out[self.user_layers], ouput_channel, (1, 1)),
            nn.BatchNorm2d(ouput_channel)
        )

        self.attention_backbone = backbone_model(pretrained=True)
        self.attention_head = FPEM_FFM(backbone_out, fpem_repeat=fpem_repeat,
                                       channels=channels, ouput_channel=1)



        self.upscale = UpScale(ouput_channel, 128, 32, 4, 4)

        self.detail_conv = nn.Sequential(
            nn.ReLU(inplace=False),
            conv1x1(32, 64),
            nn.ReLU(inplace=True),
            conv1x1(64, moire_ouput_channel),
        )

        self.detail_bias = nn.Sequential(
            nn.ReLU(inplace=False),
            conv1x1(32, 64),
            nn.ReLU(inplace=True),
            conv1x1(64, self.ouput_channel * 3),
        )

        self.middle_fea = nn.Sequential(
            nn.ReLU(inplace=False),
            conv1x1(32, 64),
            nn.ReLU(inplace=True),
            conv1x1(64, self.ouput_channel * 3),
        )

        self.moire_head = nn.Sequential(
            nn.ReLU(inplace=True),
            conv1x1(self.ouput_channel * 6, 32),
            nn.ReLU(inplace=True),
            conv1x1(32, 1),
        )

        max_len = 100
        x_list = (torch.arange(max_len).float() / max_len).view(1, 1,
                                                                max_len).repeat(
            1, max_len, 1)
        y_list = (torch.arange(max_len).float() / max_len).view(1, max_len,
                                                                1).repeat(1, 1,
                                                                          max_len)
        self.pos = torch.cat([x_list, y_list], dim=0).view(1, 2, max_len,
                                                           max_len)

        self.name = '{}_{}'.format(backbone, segmentation_head)

    def forward(self, x, pure_moire=None):
        N, _, img_H, img_W = x.size()


        moire_fea = self.moire_backbone(x)[self.user_layers]
        moire_fea = self.moire_fusion(moire_fea)
        B, C, MH, MW = moire_fea.size()

        attention_fea = self.attention_backbone(x)
        attention_fea = self.attention_head(attention_fea)
        attention = torch.sigmoid(attention_fea)
        attention = F.interpolate(attention, size=(MH, MW), mode='bilinear',
                                  align_corners=True)

        moire_fea = moire_fea * attention
        B, C, H, W = moire_fea.size()

        # moire_fea_unfold = F.unfold(moire_fea,(10,10),stride=10)
        # moire_fea_unfold = moire_fea_unfold.view(B,C,100,H//10,W//10)
        # moire_fea_unfold = moire_fea_unfold.permute(2,0,3,4,1).contiguous()

        trans_fea = self.upscale(moire_fea)

        _, _, h, w = trans_fea.size()

        if h != img_H or w != img_W:
            trans_fea = F.interpolate(trans_fea, size=(img_H, img_W),mode='bicubic', align_corners=True)

        detail_conv = self.detail_conv(trans_fea)

        detail_conv = detail_conv.view(B, 1, self.ouput_channel,self.kernel_size ** 2, img_H, img_W)
        unfold_img = self.unfold(x) # x is input image
        unfold_img = unfold_img.view(N, 3, 1, -1, img_H, img_W)
        detail_conv = torch.softmax(detail_conv, dim=3)

        # detail_conv = torch.tanh(detail_conv)
        detail_fea = torch.sum(unfold_img * detail_conv, dim=3).view(B,
                                                                     self.ouput_channel * 3,
                                                                     img_H,
                                                                     img_W)

        detail_bias = self.detail_bias(trans_fea)
        detail_bias = detail_bias.view(B, self.ouput_channel * 3, img_H, img_W)

        detail_fea = detail_fea + detail_bias

        middle_fea = self.middle_fea(trans_fea)

        fea = torch.cat([detail_fea, middle_fea], dim=1)
        moire_density = self.moire_head(fea)
        # moire_density = self.moire_head(F.interpolate(trans_fea, size=(img_H, img_W), mode='bicubic', align_corners=True))
        fea_loss = torch.mean(moire_density) * 0
        # return [moire_density], fea_loss

        return [moire_density,attention,detail_conv,detail_fea], fea_loss


class TripleBranchWithSpecificConvNoMiddle(nn.Module):
    def __init__(self, model_config: dict):
        super().__init__()
        backbone = model_config['backbone']
        pretrained = model_config['pretrained']
        segmentation_head = model_config['segmentation_head']

        self.repeat_times = model_config.get('repeat_times', 3)
        channels = model_config.get('channels', 128)
        ouput_channel = model_config.get('ouput_channel', 126)

        self.is_light = model_config.get('is_light', False)
        self.user_layers = (-1, -2)[self.is_light]

        self.ouput_channel = model_config.get('ouput_channel', 4)
        self.kernel_size = model_config.get('kernel_size', 5)

        moire_ouput_channel = self.kernel_size ** 2 * self.ouput_channel

        fpem_repeat = model_config.get('fpem_repeat', 1)

        assert backbone in backbone_dict, 'backbone must in: {}'.format(
            backbone_dict)
        assert segmentation_head in segmentation_head_dict, 'segmentation_head must in: {}'.format(
            segmentation_head_dict)

        backbone_model, backbone_out = backbone_dict[backbone]['models'], \
                                       backbone_dict[backbone]['out']



        self.unfold = nn.Unfold(
            kernel_size=(self.kernel_size, self.kernel_size),
            padding=(self.kernel_size - 1) // 2)


        self.attention_backbone = backbone_model(pretrained=True)
        self.attention_head = FPEM_FFM(backbone_out, fpem_repeat=fpem_repeat,
                                       channels=channels, ouput_channel=ouput_channel)

        self.with_pos_embedding = True
        nhead = 2
        ouput_channel = ouput_channel + 2
        # self.position_embedding = nn.Parameter(torch.rand(1000,nhead))

        self.performer = Performer(dim=ouput_channel, depth=3, heads=nhead,
                                   causal=True)

        self.upscale = UpScale(ouput_channel * 2 - 2, 128, 32, 4, 4)

        self.detail_conv = nn.Sequential(
            nn.ReLU(inplace=False),
            conv1x1(32, 64),
            nn.ReLU(inplace=True),
            conv1x1(64, moire_ouput_channel),
        )

        self.detail_bias = nn.Sequential(
            nn.ReLU(inplace=False),
            conv1x1(32, 64),
            nn.ReLU(inplace=True),
            conv1x1(64, self.ouput_channel * 3),
        )

        self.middle_fea = nn.Sequential(
            nn.ReLU(inplace=False),
            conv1x1(32, 64),
            nn.ReLU(inplace=True),
            conv1x1(64, self.ouput_channel * 3),
        )

        self.moire_head = nn.Sequential(
            nn.ReLU(inplace=True),
            conv1x1(self.ouput_channel * 6, 32),
            nn.ReLU(inplace=True),
            conv1x1(32, 1),
        )

        max_len = 100
        x_list = (torch.arange(max_len).float() / max_len).view(1, 1,
                                                                max_len).repeat(
            1, max_len, 1)
        y_list = (torch.arange(max_len).float() / max_len).view(1, max_len,
                                                                1).repeat(1, 1,
                                                                          max_len)
        self.pos = torch.cat([x_list, y_list], dim=0).view(1, 2, max_len,
                                                           max_len)

        self.name = '{}_{}'.format(backbone, segmentation_head)

    def forward(self, x, pure_moire=None):
        N, _, img_H, img_W = x.size()
        unfold_img = self.unfold(x)
        unfold_img = unfold_img.view(N, 3, 1, -1, img_H, img_W)



        moire_fea = self.attention_backbone(x)
        moire_fea = self.attention_head(moire_fea)


        B, C, H, W = moire_fea.size()

        # moire_fea_unfold = F.unfold(moire_fea,(10,10),stride=10)
        # moire_fea_unfold = moire_fea_unfold.view(B,C,100,H//10,W//10)
        # moire_fea_unfold = moire_fea_unfold.permute(2,0,3,4,1).contiguous()

        dst_size = (H // 2, W // 3)
        # if self.is_light:
        #     dst_size = (H // 4, W // 4)

        min_trans_fea = F.interpolate(moire_fea, size=dst_size,
                                      mode='bilinear', align_corners=True)
        pos = self.pos.to(x.device)
        pos = F.interpolate(pos, size=dst_size, mode='bilinear',
                            align_corners=True)
        min_trans_fea = torch.cat([min_trans_fea, pos.repeat(B, 1, 1, 1)],
                                  dim=1)

        B, C, minH, minW = min_trans_fea.size()
        min_trans_fea = min_trans_fea.view(B, C, -1).permute(2, 0,
                                                             1).contiguous()

        # if self.with_pos_embedding:
        #     S, B, C = trans_fea.size()
        #     trans_fea = torch.cat([trans_fea,self.position_embedding[:S].unsqueeze(1).repeat(1,B,1)],dim=2)

        min_trans_fea = self.performer(min_trans_fea).permute(1, 2,
                                                              0).contiguous().view(
            B, -1, minH, minW)
        min_trans_fea = F.interpolate(min_trans_fea, size=(H, W),
                                      mode='bicubic', align_corners=True)
        trans_fea = torch.cat([moire_fea, min_trans_fea], dim=1)

        trans_fea = self.upscale(trans_fea)

        _, _, h, w = trans_fea.size()

        if h != img_H or w != img_W:
            trans_fea = F.interpolate(trans_fea, size=(img_H, img_W),mode='bicubic', align_corners=True)


        detail_conv = self.detail_conv(trans_fea)
        detail_conv = detail_conv.view(B, 1, self.ouput_channel,
                                       self.kernel_size ** 2, img_H, img_W)
        # detail_conv = torch.softmax(detail_conv, dim=3)

        detail_conv = torch.tanh(detail_conv)

        detail_fea = torch.sum(unfold_img * detail_conv, dim=3).view(B,
                                                                     self.ouput_channel * 3,
                                                                     img_H,
                                                                     img_W)

        detail_bias = self.detail_bias(trans_fea)
        detail_bias = detail_bias.view(B, self.ouput_channel * 3, img_H, img_W)

        detail_fea = detail_fea + detail_bias

        middle_fea = self.middle_fea(trans_fea)

        fea = torch.cat([detail_fea, middle_fea], dim=1)
        moire_density = self.moire_head(fea)
        # moire_density = self.moire_head(F.interpolate(trans_fea, size=(img_H, img_W), mode='bicubic', align_corners=True))
        fea_loss = torch.mean(moire_density) * 0

        return [moire_density], fea_loss


class TripleBranchWithSpecificConv_H(nn.Module):
    def __init__(self, model_config: dict):
        super().__init__()
        backbone = model_config['backbone']
        pretrained = model_config['pretrained']
        segmentation_head = model_config['segmentation_head']

        self.repeat_times = model_config.get('repeat_times', 3)
        channels = model_config.get('channels', 128)
        ouput_channel = model_config.get('ouput_channel', 126)

        self.is_light = model_config.get('is_light', False)
        self.user_layers = (-1, -2)[self.is_light]

        self.ouput_channel = model_config.get('ouput_channel', 4)
        self.kernel_size = model_config.get('kernel_size', 5)

        moire_ouput_channel = self.kernel_size ** 2 * self.ouput_channel

        fpem_repeat = model_config.get('fpem_repeat', 1)

        assert backbone in backbone_dict, 'backbone must in: {}'.format(
            backbone_dict)
        assert segmentation_head in segmentation_head_dict, 'segmentation_head must in: {}'.format(
            segmentation_head_dict)

        backbone_model, backbone_out = backbone_dict[backbone]['models'], \
                                       backbone_dict[backbone]['out']




        self.attention_backbone = backbone_model(pretrained=True)
        self.attention_head = FPEM_FFM(backbone_out, fpem_repeat=fpem_repeat,
                                       channels=channels, ouput_channel=1)

        self.with_pos_embedding = True
        nhead = 2
        ouput_channel = ouput_channel + 2
        # self.position_embedding = nn.Parameter(torch.rand(1000,nhead))

        self.performer = Performer(dim=ouput_channel, depth=3, heads=nhead,
                                   causal=True)

        self.upscale = UpScale(ouput_channel * 2 - 2, 128, 32, 4, 4)

        self.detail_conv = nn.Sequential(
            nn.ReLU(inplace=False),
            conv1x1(32, 64),
            nn.ReLU(inplace=True),
            conv1x1(64, moire_ouput_channel),
        )

        self.detail_bias = nn.Sequential(
            nn.ReLU(inplace=False),
            conv1x1(32, 64),
            nn.ReLU(inplace=True),
            conv1x1(64, self.ouput_channel * 3),
        )

        self.middle_fea = nn.Sequential(
            nn.ReLU(inplace=False),
            conv1x1(32, 64),
            nn.ReLU(inplace=True),
            conv1x1(64, self.ouput_channel * 3),
        )

        self.moire_head = nn.Sequential(
            nn.ReLU(inplace=True),
            conv1x1(self.ouput_channel * 6, 32),
            nn.ReLU(inplace=True),
            conv1x1(32, 1),
        )

        max_len = 100
        x_list = (torch.arange(max_len).float() / max_len).view(1, 1,
                                                                max_len).repeat(
            1, max_len, 1)
        y_list = (torch.arange(max_len).float() / max_len).view(1, max_len,
                                                                1).repeat(1, 1,
                                                                          max_len)
        self.pos = torch.cat([x_list, y_list], dim=0).view(1, 2, max_len,
                                                           max_len)

        self.name = '{}_{}'.format(backbone, segmentation_head)

    def forward(self, x, pure_moire=None):
        N, _, img_H, img_W = x.size()


        moire_fea = self.attention_backbone(x)
        moire_fea = self.attention_head(moire_fea)


        trans_fea = F.interpolate(moire_fea, size=(img_H, img_W),mode='bicubic', align_corners=True)


        fea_loss = torch.mean(trans_fea) * 0

        return [trans_fea], fea_loss


class TripleBranchWithSpecificConv_HL(nn.Module):
    def __init__(self, model_config: dict):
        super().__init__()
        backbone = model_config['backbone']
        pretrained = model_config['pretrained']
        segmentation_head = model_config['segmentation_head']

        self.repeat_times = model_config.get('repeat_times', 3)
        channels = model_config.get('channels', 128)
        ouput_channel = model_config.get('ouput_channel', 126)

        self.is_light = model_config.get('is_light', False)
        self.user_layers = (-1, -2)[self.is_light]

        self.ouput_channel = model_config.get('ouput_channel', 4)
        self.kernel_size = model_config.get('kernel_size', 5)

        moire_ouput_channel = self.kernel_size ** 2 * self.ouput_channel

        fpem_repeat = model_config.get('fpem_repeat', 1)

        assert backbone in backbone_dict, 'backbone must in: {}'.format(
            backbone_dict)
        assert segmentation_head in segmentation_head_dict, 'segmentation_head must in: {}'.format(
            segmentation_head_dict)

        backbone_model, backbone_out = backbone_dict[backbone]['models'], \
                                       backbone_dict[backbone]['out']

        self.moire_backbone = backbone_model(pretrained=False,
                                             replace_stride_with_dilation=[
                                                 True, True, False])

        self.unfold = nn.Unfold(
            kernel_size=(self.kernel_size, self.kernel_size),
            padding=(self.kernel_size - 1) // 2)

        self.moire_fusion = nn.Sequential(
            nn.Conv2d(backbone_out[self.user_layers], ouput_channel, (1, 1)),
            nn.BatchNorm2d(ouput_channel)
        )

        self.attention_backbone = backbone_model(pretrained=True)
        self.attention_head = FPEM_FFM(backbone_out, fpem_repeat=fpem_repeat,
                                       channels=channels, ouput_channel=1)

        self.with_pos_embedding = True
        nhead = 2
        ouput_channel = ouput_channel + 2
        # self.position_embedding = nn.Parameter(torch.rand(1000,nhead))

        self.performer = Performer(dim=ouput_channel, depth=3, heads=nhead,
                                   causal=True)

        self.upscale = UpScale(ouput_channel * 2 - 2, 128, 32, 4, 4)

        self.detail_conv = nn.Sequential(
            nn.ReLU(inplace=False),
            conv1x1(32, 64),
            nn.ReLU(inplace=True),
            conv1x1(64, 1),
        )



        max_len = 100
        x_list = (torch.arange(max_len).float() / max_len).view(1, 1,
                                                                max_len).repeat(
            1, max_len, 1)
        y_list = (torch.arange(max_len).float() / max_len).view(1, max_len,
                                                                1).repeat(1, 1,
                                                                          max_len)
        self.pos = torch.cat([x_list, y_list], dim=0).view(1, 2, max_len,
                                                           max_len)

        self.name = '{}_{}'.format(backbone, segmentation_head)

    def forward(self, x, pure_moire=None):
        N, _, img_H, img_W = x.size()


        moire_fea = self.moire_backbone(x)[self.user_layers]
        moire_fea = self.moire_fusion(moire_fea)
        B, C, MH, MW = moire_fea.size()

        attention_fea = self.attention_backbone(x)
        attention_fea = self.attention_head(attention_fea)
        attention = torch.sigmoid(attention_fea)
        attention = F.interpolate(attention, size=(MH, MW), mode='bilinear',
                                  align_corners=True)

        moire_fea = moire_fea * attention
        B, C, H, W = moire_fea.size()

        # moire_fea_unfold = F.unfold(moire_fea,(10,10),stride=10)
        # moire_fea_unfold = moire_fea_unfold.view(B,C,100,H//10,W//10)
        # moire_fea_unfold = moire_fea_unfold.permute(2,0,3,4,1).contiguous()

        dst_size = (H // 2, W // 3)
        # if self.is_light:
        #     dst_size = (H // 4, W // 4)

        min_trans_fea = F.interpolate(moire_fea, size=dst_size,
                                      mode='bilinear', align_corners=True)
        pos = self.pos.to(x.device)
        pos = F.interpolate(pos, size=dst_size, mode='bilinear',
                            align_corners=True)
        min_trans_fea = torch.cat([min_trans_fea, pos.repeat(B, 1, 1, 1)],
                                  dim=1)

        B, C, minH, minW = min_trans_fea.size()
        min_trans_fea = min_trans_fea.view(B, C, -1).permute(2, 0,
                                                             1).contiguous()

        # if self.with_pos_embedding:
        #     S, B, C = trans_fea.size()
        #     trans_fea = torch.cat([trans_fea,self.position_embedding[:S].unsqueeze(1).repeat(1,B,1)],dim=2)

        min_trans_fea = self.performer(min_trans_fea).permute(1, 2,
                                                              0).contiguous().view(
            B, -1, minH, minW)
        min_trans_fea = F.interpolate(min_trans_fea, size=(H, W),
                                      mode='bicubic', align_corners=True)
        trans_fea = torch.cat([moire_fea, min_trans_fea], dim=1)

        trans_fea = self.upscale(trans_fea)

        _, _, h, w = trans_fea.size()

        if h != img_H or w != img_W:
            trans_fea = F.interpolate(trans_fea, size=(img_H, img_W),
                                      mode='bicubic', align_corners=True)

        detail_conv = self.detail_conv(trans_fea)



        fea_loss = torch.mean(detail_conv) * 0
        # return [moire_density], fea_loss

        return [detail_conv], fea_loss


class TripleBranchWithSpecificConv_HS(nn.Module):
    def __init__(self, model_config: dict):
        super().__init__()
        backbone = model_config['backbone']
        pretrained = model_config['pretrained']
        segmentation_head = model_config['segmentation_head']

        self.repeat_times = model_config.get('repeat_times', 3)
        channels = model_config.get('channels', 128)
        ouput_channel = model_config.get('ouput_channel', 126)

        self.is_light = model_config.get('is_light', False)
        self.user_layers = (-1, -2)[self.is_light]

        self.ouput_channel = model_config.get('ouput_channel', 4)
        self.kernel_size = model_config.get('kernel_size', 5)

        moire_ouput_channel = self.kernel_size ** 2 * self.ouput_channel

        fpem_repeat = model_config.get('fpem_repeat', 1)

        assert backbone in backbone_dict, 'backbone must in: {}'.format(
            backbone_dict)
        assert segmentation_head in segmentation_head_dict, 'segmentation_head must in: {}'.format(
            segmentation_head_dict)

        backbone_model, backbone_out = backbone_dict[backbone]['models'], \
                                       backbone_dict[backbone]['out']

        self.moire_backbone = backbone_model(pretrained=False,
                                             replace_stride_with_dilation=[
                                                 True, True, False])

        self.unfold = nn.Unfold(
            kernel_size=(self.kernel_size, self.kernel_size),
            padding=(self.kernel_size - 1) // 2)

        self.moire_fusion = nn.Sequential(
            nn.Conv2d(backbone_out[self.user_layers], ouput_channel, (1, 1)),
            nn.BatchNorm2d(ouput_channel)
        )

        self.attention_backbone = backbone_model(pretrained=True)
        self.attention_head = FPEM_FFM(backbone_out, fpem_repeat=fpem_repeat,
                                       channels=channels, ouput_channel=32)

        self.with_pos_embedding = True
        nhead = 2
        ouput_channel = ouput_channel + 2
        # self.position_embedding = nn.Parameter(torch.rand(1000,nhead))



        self.detail_conv = nn.Sequential(
            nn.ReLU(inplace=False),
            conv1x1(32, 64),
            nn.ReLU(inplace=True),
            conv1x1(64, moire_ouput_channel),
        )

        self.detail_bias = nn.Sequential(
            nn.ReLU(inplace=False),
            conv1x1(32, 64),
            nn.ReLU(inplace=True),
            conv1x1(64, self.ouput_channel * 3),
        )

        self.middle_fea = nn.Sequential(
            nn.ReLU(inplace=False),
            conv1x1(32, 64),
            nn.ReLU(inplace=True),
            conv1x1(64, self.ouput_channel * 3),
        )

        self.moire_head = nn.Sequential(
            nn.ReLU(inplace=True),
            conv1x1(self.ouput_channel * 6, 32),
            nn.ReLU(inplace=True),
            conv1x1(32, 1),
        )



        self.name = '{}_{}'.format(backbone, segmentation_head)

    def forward(self, x, pure_moire=None):
        N, _, img_H, img_W = x.size()


        moire_fea = self.moire_backbone(x)[self.user_layers]
        moire_fea = self.moire_fusion(moire_fea)
        B, C, MH, MW = moire_fea.size()

        attention_fea = self.attention_backbone(x)
        trans_fea = self.attention_head(attention_fea)



        _, _, h, w = trans_fea.size()

        if h != img_H or w != img_W:
            trans_fea = F.interpolate(trans_fea, size=(img_H, img_W),
                                      mode='bicubic', align_corners=True)


        detail_conv = self.detail_conv(trans_fea)

        detail_conv = detail_conv.view(B, 1, self.ouput_channel,self.kernel_size ** 2, img_H, img_W)
        unfold_img = self.unfold(x) # x is input image
        unfold_img = unfold_img.view(N, 3, 1, -1, img_H, img_W)
        detail_conv = torch.softmax(detail_conv, dim=3)

        # detail_conv = torch.tanh(detail_conv)
        detail_fea = torch.sum(unfold_img * detail_conv, dim=3).view(B,
                                                                     self.ouput_channel * 3,
                                                                     img_H,
                                                                     img_W)

        detail_bias = self.detail_bias(trans_fea)
        detail_bias = detail_bias.view(B, self.ouput_channel * 3, img_H, img_W)

        detail_fea = detail_fea + detail_bias

        middle_fea = self.middle_fea(trans_fea)

        fea = torch.cat([detail_fea, middle_fea], dim=1)
        moire_density = self.moire_head(fea)
        # moire_density = self.moire_head(F.interpolate(trans_fea, size=(img_H, img_W), mode='bicubic', align_corners=True))
        fea_loss = torch.mean(moire_density) * 0
        # return [moire_density], fea_loss

        return [moire_density], fea_loss


class TripleBranchWithSpecificConv_HP(nn.Module):
    def __init__(self, model_config: dict):
        super().__init__()
        backbone = model_config['backbone']
        pretrained = model_config['pretrained']
        segmentation_head = model_config['segmentation_head']

        self.repeat_times = model_config.get('repeat_times', 3)
        channels = model_config.get('channels', 128)
        ouput_channel = model_config.get('ouput_channel', 126)

        self.is_light = model_config.get('is_light', False)
        self.user_layers = (-1, -2)[self.is_light]

        self.ouput_channel = model_config.get('ouput_channel', 4)
        self.kernel_size = model_config.get('kernel_size', 5)

        moire_ouput_channel = self.kernel_size ** 2 * self.ouput_channel

        fpem_repeat = model_config.get('fpem_repeat', 1)

        assert backbone in backbone_dict, 'backbone must in: {}'.format(
            backbone_dict)
        assert segmentation_head in segmentation_head_dict, 'segmentation_head must in: {}'.format(
            segmentation_head_dict)

        backbone_model, backbone_out = backbone_dict[backbone]['models'], \
                                       backbone_dict[backbone]['out']



        self.unfold = nn.Unfold(
            kernel_size=(self.kernel_size, self.kernel_size),
            padding=(self.kernel_size - 1) // 2)


        self.attention_backbone = backbone_model(pretrained=True)
        self.attention_head = FPEM_FFM(backbone_out, fpem_repeat=fpem_repeat,
                                       channels=channels, ouput_channel=ouput_channel)

        self.with_pos_embedding = True
        nhead = 2
        ouput_channel = ouput_channel + 2
        # self.position_embedding = nn.Parameter(torch.rand(1000,nhead))

        self.performer = Performer(dim=ouput_channel, depth=3, heads=nhead,
                                   causal=True)

        self.upscale = UpScale(ouput_channel * 2 - 2, 128, 32, 4, 4)

        self.detail_conv = nn.Sequential(
            nn.ReLU(inplace=False),
            conv1x1(32, 64),
            nn.ReLU(inplace=True),
            conv1x1(64, 1),
        )


        max_len = 100
        x_list = (torch.arange(max_len).float() / max_len).view(1, 1,
                                                                max_len).repeat(
            1, max_len, 1)
        y_list = (torch.arange(max_len).float() / max_len).view(1, max_len,
                                                                1).repeat(1, 1,
                                                                          max_len)
        self.pos = torch.cat([x_list, y_list], dim=0).view(1, 2, max_len,
                                                           max_len)

        self.name = '{}_{}'.format(backbone, segmentation_head)

    def forward(self, x, pure_moire=None):
        N, _, img_H, img_W = x.size()



        moire_fea = self.attention_backbone(x)
        moire_fea = self.attention_head(moire_fea)


        B, C, H, W = moire_fea.size()

        # moire_fea_unfold = F.unfold(moire_fea,(10,10),stride=10)
        # moire_fea_unfold = moire_fea_unfold.view(B,C,100,H//10,W//10)
        # moire_fea_unfold = moire_fea_unfold.permute(2,0,3,4,1).contiguous()

        dst_size = (H // 2, W // 3)
        # if self.is_light:
        #     dst_size = (H // 4, W // 4)

        min_trans_fea = F.interpolate(moire_fea, size=dst_size,
                                      mode='bilinear', align_corners=True)
        pos = self.pos.to(x.device)
        pos = F.interpolate(pos, size=dst_size, mode='bilinear',
                            align_corners=True)
        min_trans_fea = torch.cat([min_trans_fea, pos.repeat(B, 1, 1, 1)],
                                  dim=1)

        B, C, minH, minW = min_trans_fea.size()
        min_trans_fea = min_trans_fea.view(B, C, -1).permute(2, 0,
                                                             1).contiguous()

        # if self.with_pos_embedding:
        #     S, B, C = trans_fea.size()
        #     trans_fea = torch.cat([trans_fea,self.position_embedding[:S].unsqueeze(1).repeat(1,B,1)],dim=2)

        min_trans_fea = self.performer(min_trans_fea).permute(1, 2,
                                                              0).contiguous().view(
            B, -1, minH, minW)
        min_trans_fea = F.interpolate(min_trans_fea, size=(H, W),
                                      mode='bicubic', align_corners=True)
        trans_fea = torch.cat([moire_fea, min_trans_fea], dim=1)

        trans_fea = self.upscale(trans_fea)

        _, _, h, w = trans_fea.size()

        if h != img_H or w != img_W:
            trans_fea = F.interpolate(trans_fea, size=(img_H, img_W),mode='bicubic', align_corners=True)


        detail_conv = self.detail_conv(trans_fea)


        # moire_density = self.moire_head(F.interpolate(trans_fea, size=(img_H, img_W), mode='bicubic', align_corners=True))
        fea_loss = torch.mean(detail_conv) * 0

        return [detail_conv], fea_loss



class TripleBranchWithSpecificConv_HLNoP(nn.Module):
    def __init__(self, model_config: dict):
        super().__init__()
        backbone = model_config['backbone']
        pretrained = model_config['pretrained']
        segmentation_head = model_config['segmentation_head']

        self.repeat_times = model_config.get('repeat_times', 3)
        channels = model_config.get('channels', 128)
        ouput_channel = model_config.get('ouput_channel', 126)

        self.is_light = model_config.get('is_light', False)
        self.user_layers = (-1, -2)[self.is_light]

        self.ouput_channel = model_config.get('ouput_channel', 4)
        self.kernel_size = model_config.get('kernel_size', 5)

        moire_ouput_channel = self.kernel_size ** 2 * self.ouput_channel

        fpem_repeat = model_config.get('fpem_repeat', 1)

        assert backbone in backbone_dict, 'backbone must in: {}'.format(
            backbone_dict)
        assert segmentation_head in segmentation_head_dict, 'segmentation_head must in: {}'.format(
            segmentation_head_dict)

        backbone_model, backbone_out = backbone_dict[backbone]['models'], \
                                       backbone_dict[backbone]['out']

        self.moire_backbone = backbone_model(pretrained=False,
                                             replace_stride_with_dilation=[
                                                 True, True, False])



        self.moire_fusion = nn.Sequential(
            nn.Conv2d(backbone_out[self.user_layers], ouput_channel, (1, 1)),
            nn.BatchNorm2d(ouput_channel)
        )

        self.attention_backbone = backbone_model(pretrained=True)
        self.attention_head = FPEM_FFM(backbone_out, fpem_repeat=fpem_repeat,
                                       channels=channels, ouput_channel=1)



        self.detail_conv = nn.Sequential(
            nn.ReLU(inplace=False),
            conv1x1(ouput_channel, 64),
            nn.ReLU(inplace=True),
            conv1x1(64, 1),
        )





        self.name = '{}_{}'.format(backbone, segmentation_head)

    def forward(self, x, pure_moire=None):
        N, _, img_H, img_W = x.size()


        moire_fea = self.moire_backbone(x)[self.user_layers]
        moire_fea = self.moire_fusion(moire_fea)
        B, C, MH, MW = moire_fea.size()

        attention_fea = self.attention_backbone(x)
        attention_fea = self.attention_head(attention_fea)
        attention = torch.sigmoid(attention_fea)
        attention = F.interpolate(attention, size=(MH, MW), mode='bilinear',
                                  align_corners=True)

        moire_fea = moire_fea * attention


        _, _, h, w = moire_fea.size()

        if h != img_H or w != img_W:
            moire_fea = F.interpolate(moire_fea, size=(img_H, img_W),
                                      mode='bicubic', align_corners=True)

        detail_conv = self.detail_conv(moire_fea)



        fea_loss = torch.mean(detail_conv) * 0
        # return [moire_density], fea_loss

        return [detail_conv], fea_loss


if __name__ == '__main__':
    device = torch.device('cpu')
    x = torch.zeros(1, 3, 640, 640).to(device)

    model_config = {
        'backbone': 'resnet18',
        'fpem_repeat': 4,  # fpem模块重复的次数
        'pretrained': False,  # backbone 是否使用imagesnet的预训练模型
        'result_num': 7,
        'segmentation_head': 'FPEM_FFM'  # 分割头，FPN or FPEM
    }
    model = Model(model_config=model_config).to(device)
    y = model(x)
    print(y.shape)
    print(model)
    # torch.save(model.state_dict(), 'PAN.pth')
