import torch
from torch import nn
from efficientnet_pytorch.model import EfficientNet as effNet
import torch.nn.functional as F


class EfficientNet(nn.Module):
    def __init__(self, input_shape, width_coefficient, depth_coefficient):
        super().__init__()
        print(input_shape)
        dic = {'num_classes': 256,
                'width_coefficient': width_coefficient,
                'depth_coefficient': depth_coefficient,
                'image_size': input_shape,
                'dropout_rate': 0.4,
                'drop_connect_rate': 0.5}
        self.feature = effNet.from_name('efficientnet-b0', in_channels=1, **dic)
        self.cls_layer = nn.Linear(256, 2)

    def forward(self, x):
        feat = self.feature(x)
        out = self.cls_layer(feat)
        return F.log_softmax(out, dim=-1), feat
