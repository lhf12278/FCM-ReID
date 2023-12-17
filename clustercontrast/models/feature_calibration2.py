# -*- coding: UTF-8 -*-
'''
@Project ：TransReID-SSL-main-v3
@File ：feature_calibration.py
@Author ：棋子
@Date ：2023/7/12 19:02
@Software: PyCharm
'''
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

class FCM(nn.Module):
    def __init__(self, t=1):
        super(FCM, self).__init__()
        self.temperature = t

    def forward(self, feats):
        print('======== fcm  input feats ==========')
        k = 16
        scores = torch.matmul(feats, feats.transpose(-2, -1)) / self.temperature  # # torch.Size([32621, 32621])
        mask = torch.full_like(scores, float("-inf"))
        mask.fill_diagonal_(1)

        # Compute pairwise distances
        distances = torch.cdist(feats, feats)

        # Initialize a list to store mutual k-neighbors
        # mutual_k_neighbors = []

        _, indices = torch.topk(distances, k + 1, largest=False)  # +1 to include itself
        knn_indices = indices[:, 1:k + 1]

        for i in range(len(feats)):
            i_knn = knn_indices[i]
            mutual_knn = torch.tensor([j.item() for j in i_knn if i in knn_indices[j]])

            if len(mutual_knn) > 0:
                mask[i, mutual_knn] = scores[i, mutual_knn]

            # mutual_k_neighbors.append(mutual_knn)

        attn_weights = nn.Softmax(dim=-1)(mask)
        feats_att = torch.matmul(attn_weights, feats)
        feats_fcm = feats + feats_att
        feats_norm = F.normalize(feats_fcm, p=2, dim=1)

        return feats_norm

if __name__ == '__main__':
    fcm = FCM()
    feats, labels = torch.load('../../samples.pt')
    print(feats.shape)
    feats = fcm(feats)
    print(feats.shape)






