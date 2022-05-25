import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable

def cbr_layer(in_channels, out_channels, kernel_size, groups=1, stride=1, activation=True):
    if activation:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=int(kernel_size / 2),
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=int(kernel_size / 2),
                      groups=groups,
                      bias=False),
            nn.BatchNorm2d(out_channels, affine=True, eps=1e-5, momentum=0.1))


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, exp, stride):
        super().__init__()
        intermed_planes = in_planes*exp
        self.residual = (in_planes == out_planes) and (stride == 1)
        self.output = nn.Sequential(cbr_layer(in_planes, intermed_planes, kernel_size=1),
                                     cbr_layer(intermed_planes, intermed_planes, kernel_size=3, stride=stride, groups=intermed_planes),
                                     cbr_layer(intermed_planes, out_planes, kernel_size=1, activation=False))

    def forward(self, x):
        out = self.output(x)
        if self.residual:
            return (out+x)
        else:
            return out

class Mobilenet_backbone(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet_config = [[1, 16, 1, 1],
                            [6, 24, 2, 2],
                            [6, 32, 3, 2],
                            [6, 64, 4, 2],
                            [6, 96, 3, 1],
                            [6, 160, 3, 2],
                            [6, 320, 1, 1]]

        self.in_channels = 32
        self.layer1 = cbr_layer(3, self.in_channels, kernel_size=3, stride=2)
        layer_count = 2
        for t, c, n, s in mobilenet_config:
            layers = []
            for i in range(n):
                layers.append(InvertedResidualBlock(self.in_channels, c, t, stride=s if i==0 else 1))
                self.in_channels = c
            setattr(self, "layer{}".format(layer_count), nn.Sequential(*layers))
            layer_count += 1

    def forward(self,x):
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)  # 24, x / 4
        l4 = self.layer4(l3)  # 32, x / 8
        l5 = self.layer5(l4)  # 64, x / 16
        l6 = self.layer6(l5)  # 96, x / 16
        l7 = self.layer7(l6)  # 160, x / 32
        l8 = self.layer8(l7)  # 320, x / 32

        return l3, l4, l5, l6, l7, l8
class CRPBlock(nn.Module):
    def __init__(self, in_planes, out_planes, num_stages, groups=False):
        super().__init__()
        for i in range(num_stages):
            setattr(self, '{}_{}'.format(i+1, 'outvar_dimred'),
                    nn.Conv2d(in_planes if (i == 0) else out_planes,
                              out_planes, kernel_size=1, stride=1,
                              padding=0, bias=False, groups=in_planes if groups else 1))
        self.stride = 1
        self.n_stages = num_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self,x):
        out=x
        for i in range(self.n_stages):
            out = self.maxpool(out)
            out = getattr(self, '{}_{}'.format(i+1, 'outvar_dimred'))(out)
            x = out+x
        return x

class RefineNetDecoder(nn.Module):
    def __init__(self, tasks, num_classes, num_instances):
        super().__init__()
        self.tasks = tasks
        self.num_classes = num_classes
        self.num_instances = num_instances
        self.conv8 = nn.Conv2d(320, 256, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv7 = nn.Conv2d(160, 256, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv6 = nn.Conv2d(96, 256, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv5 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv4 = nn.Conv2d(32, 256, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv3 = nn.Conv2d(24, 256, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.crp4 = self.make_crp(256, 256, 4, groups=False)
        self.crp3 = self.make_crp(256, 256, 4, groups=False)
        self.crp2 = self.make_crp(256, 256, 4, groups=False)
        self.crp1 = self.make_crp(256, 256, 4, groups=True)
        self.conv_adapt4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv_adapt3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv_adapt2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.relu = nn.ReLU6(inplace=True)
        self.pre_depth = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, groups=256, bias=False)
        self.depth = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.pre_segm = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, groups=256, bias=False)
        self.segm = nn.Conv2d(256, self.num_classes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.pre_insegm = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, groups=256, bias=False)
        if self.tasks[2]:
	        self.insegm = nn.Conv2d(256, self.num_instances, kernel_size=1, stride=1)
        
    def make_crp(self, in_planes, out_planes, num_stages, groups=False):
        layers = [CRPBlock(in_planes, out_planes, num_stages, groups=groups)]
        return nn.Sequential(*layers)

    def forward(self, l3, l4, l5, l6, l7, l8):
        l8 = self.conv8(l8)
        l7 = self.conv7(l7)
        l7 = self.relu(l8+l7)
        l7 = self.crp4(l7)
        l7 = self.conv_adapt4(l7)
        l7 = nn.Upsample(size = l6.size()[2:],mode='bilinear', align_corners=False)(l7)

        l6 = self.conv6(l6)
        l5 = self.conv5(l5)
        l5 = self.relu(l5+l6+l7)
        l5 = self.crp3(l5)
        l5 = self.conv_adapt3(l5)
        l5 = nn.Upsample(size = l4.size()[2:],mode='bilinear', align_corners=False)(l5)
        l4 = self.conv4(l4)
        l4 = self.relu(l5+l4)
        l4 = self.crp2(l4)
        l4 = self.conv_adapt2(l4)
        l4 = nn.Upsample(size = l3.size()[2:],mode='bilinear', align_corners=False)(l4)

        l3 = self.conv3(l3)
        l3 = self.relu(l3+l4)
        l3 = self.crp1(l3)

        out_insegm = None
        out_segm = None
        out_depth = None
        if self.tasks[2]:
            #Instance Segmentation
            out_insegm = self.pre_insegm(l3)
            out_insegm = self.relu(out_insegm)
            out_insegm = self.pre_insegm(out_insegm)
            out_insegm = self.relu(out_insegm)
            out_insegm = self.insegm(out_insegm)
        if self.tasks[1]:
            # Semantic Segmentation
            out_segm = self.pre_segm(l3)
            out_segm = self.relu(out_segm)
            out_segm = self.pre_segm(out_segm)
            out_segm = self.relu(out_segm)
            out_segm = self.segm(out_segm)
        if self.tasks[0]:
            # Depth Estimation
            out_depth = self.pre_depth(l3)
            out_depth = self.relu(out_depth)
            out_depth = self.pre_depth(out_depth)
            out_depth = self.relu(out_depth)
            out_depth = self.depth(out_depth)

        return out_depth, out_segm, out_insegm

class MNET(nn.Module):
    def __init__(self, tasks=[True, False, False], num_classes=None, num_instances=None):
        super().__init__()
        self.n_classes = num_classes
        self.tasks = tasks  #[Depth Estimation, Semantic Segmentation, Instance Segmentation]
        self.n_instances = num_instances
        self.enc = Mobilenet_backbone()
        self.dec = RefineNetDecoder(self.tasks, self.n_classes, self.n_instances)

    def forward(self, x):
        l3, l4, l5, l6, l7, l8 = self.enc(x)
        out_depth, out_segm, out_insegm = self.dec(l3, l4, l5, l6, l7, l8)
        return out_depth, out_segm, out_insegm