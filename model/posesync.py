import numpy as np
import torch
import torch.nn as nn
from torchvision.models import wide_resnet50_2
from model.clip_model import clip
import torch.nn.functional as F
import math

class Posesync(nn.Module):
    def __init__(
        self,
        window_len=5,
        fix_backbone=True,
        device="cuda:1",
        face_backbone="wide_resnet50_2",
    ):
        super(Posesync, self).__init__()
        self.window_len = window_len
        self.device = device
        self.face_backbone = face_backbone

        if face_backbone == "ViT-L/14":
            # 将嘴唇图像编码为长度为768的一维向量
            self.face_embedder, _ = clip.load("ViT-L/14", device=device)
            first_dimension = 768
        elif face_backbone == "wide_resnet50_2":
            # 将嘴唇图像编码为长度为1000的一维向量
            self.face_embedder = wide_resnet50_2(pretrained=True, progress=True)
            first_dimension = 1000

        self.first_dimension = first_dimension

        # 将WINDOW_LEN个一维向量拼接作为输入，输出一个128维的定长向量
        self.face_encoder = nn.Sequential(
            nn.Linear(first_dimension * window_len, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        self.landmark_encoder = nn.Sequential(
            nn.Linear(window_len * 68 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        
        self.discriminator = nn.Sequential(
            nn.Linear(128*2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        

        if fix_backbone:
            for param in self.face_embedder.parameters():
                param.requires_grad = True


    def forward(self, face, landmark):
        encodered_face = self.encode_face(face)
        encodered_landmark = self.encode_landmark(landmark)
        return self.discriminator(torch.concat((encodered_face,encodered_landmark), 1))

    def encode_face(self, face):
        embedded_face = None
        for i in range(self.window_len):
            single_face = face[i].to(self.device).float()

            if self.face_backbone.startswith("wide_resnet"):
                tmp = self.face_embedder(single_face)
            elif self.face_backbone.startswith("ViT"):
                tmp = self.face_embedder.encode_image(single_face).float()

            if i == 0:
                embedded_face = tmp
            else:
                embedded_face = torch.concat((embedded_face, tmp), 1)

        # print(f'Here embedded face is of:{embedded_face.shape}')
        # print(f'The type is: {embedded_face[0].type}')
        return self.face_encoder(embedded_face)

    def encode_landmark(self, landmark):
        landmark = landmark.to(self.device).float()
        return self.landmark_encoder(landmark)


