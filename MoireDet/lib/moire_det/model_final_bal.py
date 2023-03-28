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


# 'MobileNetV3_Large': {'models': MobileNetV3_Large, 'out': [24, 40, 160, 160]},
# 'MobileNetV3_Small': {'models': MobileNetV3_Small, 'out': [16, 24, 48, 96]},
# 'shufflenetv2': {'models': shufflenet_v2_x1_0, 'out': [24, 116, 232, 464]}}

class TripleInterActionModule(nn.Module):

    def __init__(self, inplanes=7, channels=64, outplanes=1, norm_layer=None):
        super(TripleInterActionModule, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes, channels)
        self.bn1 = norm_layer(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels, outplanes)
        self.bn2 = norm_layer(outplanes)

    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3], dim=1)

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

    def __init__(self, inplanes, norm_layer=None):
        super(DualInterActionModule, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        planes = inplanes
        self.conv1 = conv3x3(inplanes * 2, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)

        identity = x1

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class PositionEmbedding(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(2, num_pos_feats)
        self.col_embed = nn.Embedding(2, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        _, _, h, w = x.size()
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


class DualBranch(nn.Module):
    def __init__(self, model_config: dict):
        super().__init__()
        backbone = model_config['backbone']
        pretrained = model_config['pretrained']
        segmentation_head = model_config['segmentation_head']

        self.repeat_times = model_config.get('repeat_times', 3)
        channels = model_config.get('channels', 128)
        ouput_channel = model_config.get('ouput_channel', 126)

        fpem_repeat = model_config.get('fpem_repeat', 1)

        assert backbone in backbone_dict, 'backbone must in: {}'.format(
            backbone_dict)
        assert segmentation_head in segmentation_head_dict, 'segmentation_head must in: {}'.format(
            segmentation_head_dict)

        backbone_model, backbone_out = backbone_dict[backbone]['models'], \
                                       backbone_dict[backbone]['out']

        self.moire_backbone = backbone_model(pretrained=False)
        self.moire_segmentation_head = FPEM_FFM(backbone_out,
                                                fpem_repeat=fpem_repeat,
                                                channels=channels,
                                                ouput_channel=ouput_channel)

        self.with_pos_embedding = True
        nhead = 8
        if self.with_pos_embedding:
            ouput_channel = ouput_channel + 2
            # self.position_embedding = nn.Parameter(torch.rand(1000,nhead))

        self.performer = Performer(dim=ouput_channel, depth=3, heads=nhead,
                                   causal=True)
        self.moire_head = nn.Sequential(conv3x3(ouput_channel, 1))

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
        _, _, img_H, img_W = x.size()

        moire_fea = self.moire_backbone(x)
        bi_fpn_fea = self.moire_segmentation_head(moire_fea)
        B, C, H, W = bi_fpn_fea.size()

        trans_fea = F.interpolate(bi_fpn_fea, size=(H // 8, W // 8),
                                  mode='bilinear', align_corners=True)
        pos = self.pos.to(x.device)
        pos = F.interpolate(pos, size=(H // 8, W // 8), mode='bilinear',
                            align_corners=True)
        if self.with_pos_embedding:
            trans_fea = torch.cat([trans_fea, pos.repeat(B, 1, 1, 1)], dim=1)

        B, C, H, W = trans_fea.size()
        trans_fea = trans_fea.view(B, C, -1).permute(2, 0, 1).contiguous()

        # if self.with_pos_embedding:
        #     S, B, C = trans_fea.size()
        #     trans_fea = torch.cat([trans_fea,self.position_embedding[:S].unsqueeze(1).repeat(1,B,1)],dim=2)

        trans_fea = self.performer(trans_fea).permute(1, 2,
                                                      0).contiguous().view(B,
                                                                           -1,
                                                                           H,
                                                                           W)

        moire_density = self.moire_head(
            F.interpolate(trans_fea, size=(img_H, img_W), mode='bilinear',
                          align_corners=True))
        moire_feas = moire_fea + [bi_fpn_fea]
        fea_loss = torch.mean(moire_density) * 0

        if pure_moire is not None and pure_moire is not 0:
            pure_moire_fea = self.moire_backbone(pure_moire)
            pure_bi_fpn_fea = self.moire_segmentation_head(pure_moire_fea)
            pure_moire_fea = pure_moire_fea + [pure_bi_fpn_fea]
            pure_moire_fea = [v.detach() for v in pure_moire_fea]
            pure_moire_density = self.moire_head(
                F.interpolate(pure_bi_fpn_fea, size=(H, W), mode='bilinear',
                              align_corners=True))
            fea_loss = 0
            for i in range(len(moire_feas)):
                fea_loss = torch.mean(
                    torch.abs(moire_feas[i] - pure_moire_fea[i])) * (
                                       i + 1) ** 0.5 + fea_loss

            return [moire_density, pure_moire_density], fea_loss

        return [moire_density], fea_loss


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


class DualBranchWithAttention(nn.Module):
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

        self.with_pos_embedding = True
        nhead = 2
        if self.with_pos_embedding:
            ouput_channel = ouput_channel + 2
            # self.position_embedding = nn.Parameter(torch.rand(1000,nhead))

        self.performer = Performer(dim=ouput_channel, depth=3, heads=nhead,
                                   causal=True)
        self.moire_head = nn.Sequential(conv3x3(ouput_channel, 1))

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
        _, _, img_H, img_W = x.size()

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
        trans_fea = F.interpolate(moire_fea, size=dst_size, mode='bilinear',
                                  align_corners=True)
        pos = self.pos.to(x.device)
        pos = F.interpolate(pos, size=dst_size, mode='bilinear',
                            align_corners=True)
        if self.with_pos_embedding:
            trans_fea = torch.cat([trans_fea, pos.repeat(B, 1, 1, 1)], dim=1)

        B, C, H, W = trans_fea.size()
        trans_fea = trans_fea.view(B, C, -1).permute(2, 0, 1).contiguous()

        # if self.with_pos_embedding:
        #     S, B, C = trans_fea.size()
        #     trans_fea = torch.cat([trans_fea,self.position_embedding[:S].unsqueeze(1).repeat(1,B,1)],dim=2)

        trans_fea = self.performer(trans_fea).permute(1, 2,
                                                      0).contiguous().view(B,
                                                                           -1,
                                                                           H,
                                                                           W)

        moire_density = self.moire_head(
            F.interpolate(trans_fea, size=(img_H, img_W), mode='bicubic',
                          align_corners=True))
        fea_loss = torch.mean(moire_density) * 0

        return [moire_density], fea_loss


class DualBranchWithAttentionUpScale(nn.Module):
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

        self.with_pos_embedding = True
        nhead = 2
        ouput_channel = ouput_channel + 2
        # self.position_embedding = nn.Parameter(torch.rand(1000,nhead))

        self.performer = Performer(dim=ouput_channel, depth=3, heads=nhead,
                                   causal=True)

        self.upscale = UpScale(ouput_channel * 2 - 2, 128, 32, 4, 4)

        self.moire_head = nn.Sequential(
            nn.ReLU(inplace=True),
            conv3x3(32, 1)
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
        _, _, img_H, img_W = x.size()

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
            moire_density = self.moire_head(
                F.interpolate(trans_fea, size=(img_H, img_W), mode='bicubic',
                              align_corners=True))
        else:
            moire_density = self.moire_head(trans_fea)
        # moire_density = self.moire_head(F.interpolate(trans_fea, size=(img_H, img_W), mode='bicubic', align_corners=True))
        fea_loss = torch.mean(moire_density) * 0

        return [moire_density], fea_loss


class DualBranchWithFineAttentionUpScale(nn.Module):
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

        self.detail = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3), padding=1),
            nn.BatchNorm2d(32)
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

        self.moire_head = nn.Sequential(
            nn.ReLU(inplace=True),
            conv1x1(64, 64),
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
        _, _, img_H, img_W = x.size()

        detail_fea = self.detail(x)

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
                                      mode='bicubic',
                                      align_corners=True)

        fea = torch.cat([detail_fea, trans_fea], dim=1)
        moire_density = self.moire_head(fea)
        # moire_density = self.moire_head(F.interpolate(trans_fea, size=(img_H, img_W), mode='bicubic', align_corners=True))
        fea_loss = torch.mean(moire_density) * 0

        return [moire_density], fea_loss


class TripleBranchWithSpecificConv_bak(nn.Module):
    def __init__(self, model_config: dict):
        super().__init__()
        backbone = model_config['backbone']
        pretrained = model_config['pretrained']
        segmentation_head = model_config['segmentation_head']

        self.repeat_times = model_config.get('repeat_times', 3)
        channels = model_config.get('channels', 128)

        self.ouput_channel = model_config.get('ouput_channel', 32)
        self.kernel_size = model_config.get('kernel_size', 5)

        moire_ouput_channel = self.kernel_size ** 2 * self.ouput_channel * 3

        self.unfold = nn.Unfold(
            kernel_size=(self.kernel_size, self.kernel_size),
            padding=(self.kernel_size - 1) // 2, stride=4)

        fpem_repeat = model_config.get('fpem_repeat', 1)

        assert backbone in backbone_dict, 'backbone must in: {}'.format(
            backbone_dict)
        assert segmentation_head in segmentation_head_dict, 'segmentation_head must in: {}'.format(
            segmentation_head_dict)

        backbone_model, backbone_out = backbone_dict[backbone]['models'], \
                                       backbone_dict[backbone]['out']

        self.moire_backbone = resnet10(pretrained=False,
                                       replace_stride_with_dilation=[True,
                                                                     True])
        self.moire_fusion = nn.Sequential(
            nn.Conv2d(128, moire_ouput_channel, (1, 1)),
            nn.BatchNorm2d(moire_ouput_channel)
        )

        self.attention_backbone = backbone_model(pretrained=True)
        self.attention_head = FPEM_FFM(backbone_out, fpem_repeat=fpem_repeat,
                                       channels=channels, ouput_channel=1)

        self.with_pos_embedding = True
        nhead = 2
        ouput_channel = self.ouput_channel + 2
        # self.position_embedding = nn.Parameter(torch.rand(1000,nhead))

        self.performer = Performer(dim=ouput_channel, depth=3, heads=nhead,
                                   causal=True)
        self.moire_head = nn.Sequential(conv3x3(ouput_channel, 1))

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
        unfold_img = unfold_img.view(N, 1, -1, img_H // 4, img_W // 4)

        moire_fea = self.moire_backbone(x)[-1]
        moire_fea = self.moire_fusion(moire_fea)
        B, C, MH, MW = moire_fea.size()

        moire_fea = moire_fea.view(B, self.ouput_channel,
                                   self.kernel_size ** 2 * 3, MH, MW)
        moire_fea = torch.softmax(moire_fea, dim=2)

        attention_fea = self.attention_backbone(x)
        attention_fea = self.attention_head(attention_fea)
        attention = torch.sigmoid(attention_fea)
        attention = F.interpolate(attention, size=(MH, MW), mode='bilinear',
                                  align_corners=True)

        attention = attention.unsqueeze(1)
        moire_fea = moire_fea * attention

        moire_fea = torch.sum(unfold_img * moire_fea, dim=2)

        B, C, H, W = moire_fea.size()

        # moire_fea_unfold = F.unfold(moire_fea,(10,10),stride=10)
        # moire_fea_unfold = moire_fea_unfold.view(B,C,100,H//10,W//10)
        # moire_fea_unfold = moire_fea_unfold.permute(2,0,3,4,1).contiguous()

        dst_size = (H // 2, W // 3)
        # if self.is_light:
        #     dst_size = (H // 4, W // 4)
        trans_fea = F.interpolate(moire_fea, size=dst_size, mode='bilinear',
                                  align_corners=True)
        pos = self.pos.to(x.device)
        pos = F.interpolate(pos, size=dst_size, mode='bilinear',
                            align_corners=True)
        if self.with_pos_embedding:
            trans_fea = torch.cat([trans_fea, pos.repeat(B, 1, 1, 1)], dim=1)

        B, C, H, W = trans_fea.size()
        trans_fea = trans_fea.view(B, C, -1).permute(2, 0, 1).contiguous()

        # if self.with_pos_embedding:
        #     S, B, C = trans_fea.size()
        #     trans_fea = torch.cat([trans_fea,self.position_embedding[:S].unsqueeze(1).repeat(1,B,1)],dim=2)

        trans_fea = self.performer(trans_fea).permute(1, 2,
                                                      0).contiguous().view(B,
                                                                           -1,
                                                                           H,
                                                                           W)

        moire_density = self.moire_head(
            F.interpolate(trans_fea, size=(img_H, img_W), mode='bicubic',
                          align_corners=True))
        fea_loss = torch.mean(moire_density) * 0

        return [moire_density], fea_loss


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
        # return [moire_density], fea_loss

        return [moire_density,attention,detail_conv,detail_fea], fea_loss


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



class TripleBranchWithSpecificConvWithAttention(nn.Module):
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
        unfold_img = self.unfold(x)
        unfold_img = unfold_img.view(N, 3, 1, -1, img_H, img_W)

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
        detail_conv = detail_conv.view(B, 1, self.ouput_channel,
                                       self.kernel_size ** 2, img_H, img_W)
        detail_conv = torch.softmax(detail_conv, dim=3)
        detail_fea = torch.sum(unfold_img * detail_conv, dim=3).view(B,
                                                                     self.ouput_channel * 3,
                                                                     img_H,
                                                                     img_W)

        detail_bias = self.detail_bias(trans_fea)
        detail_bias = torch.sigmoid(
            detail_bias.view(B, self.ouput_channel * 3, img_H, img_W))

        detail_fea = detail_fea * detail_bias

        middle_fea = self.middle_fea(trans_fea)

        fea = torch.cat([detail_fea, middle_fea], dim=1)
        moire_density = self.moire_head(fea)
        # moire_density = self.moire_head(F.interpolate(trans_fea, size=(img_H, img_W), mode='bicubic', align_corners=True))
        fea_loss = torch.mean(moire_density) * 0

        return [moire_density], fea_loss

class HeavyAttention(nn.Module):
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

        self.detail = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), padding=1),
            nn.BatchNorm2d(64)
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

        self.upscale = UpScale(ouput_channel * 2 - 2, 128, 64, 4, 4)

        self.detail_attention = nn.Sequential(
            nn.ReLU(inplace=False),
            conv1x1(64, 32),
            nn.ReLU(inplace=True),
            conv1x1(32, 64),
        )

        self.middle_fea = nn.Sequential(
            nn.ReLU(inplace=False),
            conv1x1(64, 128),
            nn.ReLU(inplace=True),
            conv1x1(128, 64),
        )

        self.moire_head = nn.Sequential(
            nn.ReLU(inplace=True),
            conv1x1(128, 64),
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
        _, _, img_H, img_W = x.size()

        detail_fea = self.detail(x)

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

        detail_attention = torch.sigmoid(self.detail_attention(trans_fea))
        detail_fea_with_attention = detail_attention * detail_fea

        middle_fea = self.middle_fea(trans_fea)

        fea = torch.cat([detail_fea_with_attention, middle_fea], dim=1)
        moire_density = self.moire_head(fea)
        # moire_density = self.moire_head(F.interpolate(trans_fea, size=(img_H, img_W), mode='bicubic', align_corners=True))
        fea_loss = torch.mean(moire_density) * 0

        return [moire_density], fea_loss


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
