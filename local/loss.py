import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from torch import cdist

class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''

    def __init__(self, gamma=2, alpha=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logpt, target):
        """
        input: [N, C]
        target: [N, ]
        """
        # logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        if self.alpha is not None:
            loss = F.nll_loss(logpt, target, torch.tensor(self.alpha).to(
                target.device), ignore_index=self.ignore_index)
        else:
            loss = F.nll_loss(logpt, target, ignore_index=self.ignore_index)
        return loss

# class BinaryTripletCenterLoss(nn.Module):
#     def __init__(self,weight, margin=32, num_classes=2, feat_dim=256):
#         super(BinaryTripletCenterLoss, self).__init__()
#         self.margin = margin
#         self.ranking_loss = nn.MarginRankingLoss(margin=margin)
#         self.centers = nn.Parameter(torch.randn(
#             num_classes, feat_dim))
#         self.class_weight = weight

#     def forward(self, inputs, targets):
#         batch_size = inputs.size(0)
#         dist = cdist(inputs, self.centers)
#         loss = 0
#         mask = targets.eq(0)
#         loss += ((dist[mask][:, 0] - dist[mask][:, 1] +
#                    self.margin).clamp(min=0)*(self.class_weight[0]/(self.class_weight[0]+self.class_weight[1]))).sum()
#         mask = targets.eq(1)
#         loss += ((dist[mask][:, 1] - dist[mask][:, 0] +
#                    self.margin).clamp(min=0)*(self.class_weight[1]/(self.class_weight[0]+self.class_weight[1]))).sum()
#         return loss.clamp(min=0, max=1e+12)/batch_size

class BinaryTripletCenterLoss(nn.Module):
    def __init__(self, weight,margin=32, num_classes=2, feat_dim=256):
        super(BinaryTripletCenterLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.centers = nn.Parameter(torch.randn(
            num_classes, feat_dim))
        self.class_weight = weight

    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        dist = cdist(inputs, self.centers)
        loss = 0
        mask = targets.eq(0)
        loss += ((dist[mask][:, 0] - dist[mask][:, 1] +
                   self.margin).clamp(min=0).sum())*self.class_weight[0]
        mask = targets.eq(1)
        loss += ((dist[mask][:, 1] - dist[mask][:, 0] +
                   self.margin).clamp(min=0).sum())*self.class_weight[1]
        return loss.clamp(min=0, max=1e+12)/batch_size

class customBTCwithFLoss(nn.Module):
    def __init__(self, weights=[1., 1.], lam=0.005, feat_dim = 256):
        super(customBTCwithFLoss, self).__init__()
        print(weights)
        self.weights = weights
        self.lam = lam
        self.focal_loss = FocalLoss(alpha=weights)
        self.btc_loss = BinaryTripletCenterLoss(weight=weights, feat_dim = feat_dim)

    def forward(self, outputs, targets):
        feats, output = outputs
        L1 = self.focal_loss(output, targets)
        L2 = self.btc_loss(feats, targets)
        return L1+L2*self.lam

    # def load_state_dict(self, state_dict):
    #     print(state_dict['btc_loss.centers'])
    #     self.btc_loss.centers = state_dict['btc_loss.centers']
