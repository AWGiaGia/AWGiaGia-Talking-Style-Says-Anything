from inspect import Parameter
from typing import Iterator
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, device="cuda:1"):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.device = device

    def forward(self, distance, label):
        label = label.to(self.device)

        distance = distance.to(self.device)

        # # label==1: fake; label==0: real
        # # 余弦相似度损失
        # loss_contrastive = torch.mean(
        #     (1 - label) * (1 - distance)
        #     + label * torch.clamp(distance - self.margin, min=0.0)
        # )

        # label==1: fake; label==0: real
        loss_contrastive = torch.mean(
            (1 - label) * distance
            + label * torch.clamp(self.margin - distance, min=0.0)
        )
        return loss_contrastive


class AlterationLoss(nn.Module):
    def __init__(
        self,
        epsilon=0.9,
        k=10,
        window_len=5,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(AlterationLoss, self).__init__()
        self.window_len = window_len
        self.device = device
        self.k = k
        self.epsilon = nn.Parameter(
            torch.tensor(epsilon, dtype=torch.float32), requires_grad=True
        )

    def forward(self, similarities):
        # 找到低相似度的图像对n
        mask = similarities < self.epsilon
        bias = self.epsilon - similarities[mask]
        n = bias.size(0)
        T = self.window_len
        return n / T * self.k * torch.sum(torch.exp(bias))
