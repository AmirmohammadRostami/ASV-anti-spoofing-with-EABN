import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class AttentionNet(nn.Module):
    def __init__(self,input_shape):
        super(AttentionNet, self).__init__()
        num_classes = 2
        block = BasicBlock
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.middle_layers = nn.Sequential(self._make_layer(block, 2, 4),
                                           self._make_layer(block, 4, 8),
                                           self._make_layer(block, 8, 12),
                                           self._make_layer(block, 12, 16))
        self.att_conv = nn.Conv2d(16, 16, kernel_size=1, padding=0,
                                  bias=False)
        self.bn_att2 = nn.BatchNorm2d(16)
        self.att_conv2 = nn.Conv2d(16, num_classes, kernel_size=1, bias=False)
        self.att_conv3 = nn.Conv2d(16, 1, kernel_size=1, bias=False)
        self.bn_att3 = nn.BatchNorm2d(1)
        self.att_gap = nn.AvgPool2d(input_shape)
        self.sigmoid = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, stride=1):
        layers = []
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes),
        )
        layers.append(block(inplanes, planes, stride, downsample))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.middle_layers(x)
        ax = self.relu(self.bn_att2(self.att_conv(x)))
        attention_map = self.sigmoid(self.bn_att3(self.att_conv3(ax)))
        ax = self.att_conv2(ax)
        output = self.att_gap(ax)
        output = output.view(output.size(0), -1)
        return attention_map, self.logsoftmax(output)