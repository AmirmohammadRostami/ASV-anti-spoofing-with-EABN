import torch.nn as nn
from src.efficientnet import EfficientNet
from src.attentionnet import AttentionNet
from src import resnet_models

class AttentionBranchNetwork(nn.Module):
    def __init__(self,input_shape = (257, 400), width_coefficient = 0.25, depth_coefficient = 0.2, perc_type = 'eff'):
        super().__init__()
        self.attention_branch = AttentionNet(input_shape)
        if perc_type == 'eff':
            self.perception_branch = EfficientNet(input_shape,width_coefficient,depth_coefficient)
        elif perc_type == 'seres':
            self.perception_branch = resnet_models.se_res2net50_v1b(num_classes=2, KaimingInit=True)
        else:
            raise NotImplementedError
    def forward(self, x, y=None):
        batch_size = x.size(0)
        attention_maps, ab_output = self.attention_branch(x)
        x = (1 + attention_maps) * x
        pb_output, feats = self.perception_branch(x)
        return (feats,pb_output), ab_output, attention_maps