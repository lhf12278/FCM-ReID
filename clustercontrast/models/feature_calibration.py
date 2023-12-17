# -*- coding: UTF-8 -*-
'''
@Project ：TransReID-SSL-main-v3
@File ：feature_calibration.py
@Author ：棋子
@Date ：2023/7/12 19:02
@Software: PyCharm
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCM(nn.Module):
    def __init__(self, temperature=1):
        super(FCM, self).__init__()
        self.temperature = temperature

    def forward(self, feats, k=16, w=1):
        print('======== fcm  input feats ==========')
        scores = torch.matmul(feats, feats.transpose(-2, -1)) / self.temperature
        top_k_values, top_k_indices = torch.topk(scores, k=k, dim=1)

        result = torch.full_like(scores, float("-inf"))
        result.scatter_(1, top_k_indices, top_k_values)

        attn_weights = nn.Softmax(dim=-1)(result)
        feats_att = torch.matmul(attn_weights, feats)
        feats_fcm = feats * w + feats_att
        feats_norm = F.normalize(feats_fcm, p=2, dim=1)

        return feats_norm

class FCM1(nn.Module):
    def __init__(self, temperature=1):
        super(FCM1, self).__init__()
        self.temperature = temperature

    def forward(self, feats, k=16, w=1):
        print('======== fcm  input feats ==========')
        scores = torch.matmul(feats, feats.transpose(-2, -1)) / self.temperature     # [256, 256]

        top_k_values, top_k_indices = torch.topk(scores, k=k + 1, dim=1)
        top_k_values, top_k_indices = top_k_values[:, 1:], top_k_indices[:, 1:]

        result = torch.full_like(scores, float("-inf"))
        result.scatter_(1, top_k_indices, top_k_values)

        attn_weights = nn.Softmax(dim=-1)(result)
        feats_att = torch.matmul(attn_weights, feats)
        feats_fcm = feats * w + feats_att * (1 - w)
        feats_norm = F.normalize(feats_fcm, p=2, dim=1)

        return feats_norm