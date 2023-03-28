# -*- coding: utf-8 -*-
# @Time    : 2019/9/13 10:29
# @Author  : zhoujun
import torch
from torch import nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, backbone_out_channels, **kwargs):
        """
        :param backbone_out_channels: 基础网络输出的维度
        :param kwargs:
        """
        super().__init__()
        result_num = kwargs.get('result_num', 6)
        inplace = True
        conv_out = 256
        # Top layer
        self.toplayer = nn.Conv2d(backbone_out_channels[3], conv_out, kernel_size=1, stride=1,
                                  padding=0)  # Reduce channels
        # Lateral layers
        self.latlayer1 = nn.Conv2d(backbone_out_channels[2], conv_out, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(backbone_out_channels[1], conv_out, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(backbone_out_channels[0], conv_out, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1)

        self.conv = nn.Sequential(
            nn.Conv2d(conv_out * 4, conv_out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(conv_out),
            nn.ReLU(inplace=inplace)
        )
        self.out_conv = nn.Conv2d(conv_out, result_num, kernel_size=1, stride=1)

    def forward(self, x):
        c2, c3, c4, c5 = x
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        x = self._upsample_cat(p2, p3, p4, p5)
        x = self.conv(x)
        x = self.out_conv(x)
        return x

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:], mode='bilinear', align_corners=False) + y

    def _upsample_cat(self, p2, p3, p4, p5):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w), mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, size=(h, w), mode='bilinear', align_corners=False)
        p5 = F.interpolate(p5, size=(h, w), mode='bilinear', align_corners=False)
        return torch.cat([p2, p3, p4, p5], dim=1)


class FPEM_FFM(nn.Module):
    def __init__(self, backbone_out_channels, **kwargs):
        """
        PANnet
        :param backbone_out_channels: 基础网络输出的维度
        """
        super().__init__()
        fpem_repeat = kwargs.get('fpem_repeat', 2)
        self.conv_c2 = nn.Conv2d(in_channels=backbone_out_channels[0], out_channels=128, kernel_size=1)
        self.conv_c3 = nn.Conv2d(in_channels=backbone_out_channels[1], out_channels=128, kernel_size=1)
        self.conv_c4 = nn.Conv2d(in_channels=backbone_out_channels[2], out_channels=128, kernel_size=1)
        self.conv_c5 = nn.Conv2d(in_channels=backbone_out_channels[3], out_channels=128, kernel_size=1)
        self.fpems = nn.ModuleList()
        for i in range(fpem_repeat):
            self.fpems.append(FPEM(128))
        self.out_conv = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1)

    def forward(self, x):
        c2, c3, c4, c5 = x
        # reduce channel
        c2 = self.conv_c2(c2)
        c3 = self.conv_c3(c3)
        c4 = self.conv_c4(c4)
        c5 = self.conv_c5(c5)
        c2_ffm = c2
        c3_ffm = c3
        c4_ffm = c4
        c5_ffm = c5

        c2_list = []
        # FPEM
        for fpem in self.fpems:
            c2, c3, c4, c5 = fpem(c2, c3, c4, c5)

            c2_list.append(c2)

            c2_ffm += c2
            c3_ffm += c3
            c4_ffm += c4
            c5_ffm += c5

        # FFM
        c5 = F.interpolate(c5_ffm, c2_ffm.size()[-2:], mode='bilinear', align_corners=True)
        c4 = F.interpolate(c4_ffm, c2_ffm.size()[-2:], mode='bilinear', align_corners=True)
        c3 = F.interpolate(c3_ffm, c2_ffm.size()[-2:], mode='bilinear', align_corners=True)
        Fy = torch.cat([c2_ffm, c3, c4, c5], dim=1)
        y = self.out_conv(Fy)
        return y,c2_list[0]


class FPEM(nn.Module):
    def __init__(self, in_channel=128):
        super().__init__()
        self.add_up_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, groups=in_channel),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.add_up_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, groups=in_channel),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.add_up_3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, groups=in_channel),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.add_down_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, groups=in_channel,
                      stride=2),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.add_down_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, groups=in_channel,
                      stride=2),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.add_down_3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, groups=in_channel,
                      stride=2),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )

    def forward(self, c2, c3, c4, c5):
        # up阶段
        c4 = self.add_up_1(c4 + F.interpolate(c5, c4.size()[-2:], mode='bilinear', align_corners=True))
        c3 = self.add_up_2(c3 + F.interpolate(c4, c3.size()[-2:], mode='bilinear', align_corners=True))
        c2 = self.add_up_3(c2 + F.interpolate(c3, c2.size()[-2:], mode='bilinear', align_corners=True))

        # down 阶段
        c3 = self.add_down_1(c2 + F.interpolate(c3, c2.size()[-2:], mode='bilinear', align_corners=True))
        c4 = self.add_down_2(c3 + F.interpolate(c4, c3.size()[-2:], mode='bilinear', align_corners=True))
        c5 = self.add_down_3(c4 + F.interpolate(c5, c4.size()[-2:], mode='bilinear', align_corners=True))
        return c2, c3, c4, c5
