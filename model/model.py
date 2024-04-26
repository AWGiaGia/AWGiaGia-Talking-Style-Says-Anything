import numpy as np
import torch
import torch.nn as nn
from torchvision.models import wide_resnet50_2
from model.clip_model import clip
import torch.nn.functional as F
import cv2,os



class Temporal(nn.Module):
    def __init__(
        self,
        window_len=5,
        fix_backbone=True,
        device="cuda:1",
        face_backbone="wide_resnet50_2",
        not_linear=True,
        scale_error=False
    ):
        super(Temporal, self).__init__()
        self.window_len = window_len
        self.device = device
        self.face_backbone = face_backbone
        self.not_linear = not_linear
        self.scale_error = scale_error
        if scale_error:
            self.proj_len = self.window_len - 1 + 2
        else:
            self.proj_len = self.window_len - 1

        if face_backbone == "ViT-L/14":
            # 将嘴唇图像编码为长度为768的一维向量
            self.face_embedder, _ = clip.load("ViT-L/14", device=device)
            first_dimension = 768
        elif face_backbone == "wide_resnet50_2":
            # 将嘴唇图像编码为长度为1000的一维向量
            self.face_embedder = wide_resnet50_2(pretrained=True, progress=True)
            first_dimension = 1000

        self.first_dimension = first_dimension
        
        # 时序注意力模块
        self.temporal_attention = Action(
            in_channels=3,
            window_len=window_len,
            device=device
        )
        
        
        # 输出头
        if not_linear:
            self.proj = nn.Sequential(
                nn.Linear(self.proj_len,1),
                nn.Sigmoid()
            )
        else: #linear模式仅用于combine模型
            self.proj = nn.Linear(self.proj_len,1)
        
        
        if fix_backbone:
            for param in self.face_embedder.parameters():
                param.requires_grad = False
        
    def forward(self,faces):
        faces = self.temporal_attention(faces)
        # faces is of: torch.tensor([batch_size,window_len,channel,h,w])
        
        features = list()
        similarites = None
        
        for i in range(self.window_len):
            single_face = faces[:,i,:,:,:].to(self.device).squeeze(1)
            if self.face_backbone.startswith("wide_resnet"):
                tmp = self.face_embedder(single_face)
            elif self.face_backbone.startswith("ViT"):
                tmp = self.face_embedder.encode_image(single_face).float()
            
            features.append(tmp)
            
            # print(f'Here tmp is of: {tmp.shape}')
            # raise ValueError('out')
            if i > 0:
                if similarites == None:
                    similarites = self.distance(features[-2],features[-1]).unsqueeze(1)
                else:
                    similarites = torch.concat((similarites, self.distance(features[-2],features[-1]).unsqueeze(1)), 1)
        
        eps = 1.0e-10
        if self.scale_error:
            # normalized = F.normalize(similarites,dim=-1)
            d_mean = torch.mean(similarites,dim=-1).unsqueeze(-1)
            d_std = torch.std(similarites,dim=-1,unbiased=False).unsqueeze(-1)
            normalized = (similarites - d_mean) / (d_std + eps)
            # print(f'Here similarites: f{similarites}')
            # print(f'Here normed: f{normalized}')
            # print(f'Here mean is: f{d_mean}')
            # print(f'Here std is: {d_std}')
            # # raise ValueError(f'out')
            morphs = torch.concat((normalized,d_mean,d_std),axis=-1)
            
            # raise ValueError(f'morphs is of: {morphs.shape}')
        else:
            # # 从大到小排序
            morphs, _ = torch.sort(similarites,dim=1,descending=True)   


        # 输出
        score = self.proj(morphs)
        # if self.not_linear:
        #     score = self.proj(morphs)
        # else:
        #     score = torch.clamp(self.proj(morphs), min = 0.0)
        
        return score
          
                  
    def distance(self, a, b):
        # # 标准化张量 a 和 b
        # a_1 = F.normalize(a, dim=1)
        # b_1 = F.normalize(b, dim=1)
        # cos_sim = F.cosine_similarity(a_1, b_1, dim=1)
        # return cos_sim
        return F.pairwise_distance(a,b,p=2,eps=1e-6) / a.size(-1)


class Contrastive(nn.Module):
    def __init__(
        self,
        window_len=5,
        fix_backbone=True,
        device="cuda:1",
        face_backbone="wide_resnet50_2",
    ):
        super(Contrastive, self).__init__()
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

        # self.distance = nn.CosineSimilarity(dim=-1, eps=1e-6)  # landmark和嘴唇的距离度量器

        if fix_backbone:
            for param in self.face_embedder.parameters():
                param.requires_grad = False

    def distance(self, a, b):
        # # 标准化张量 a 和 b
        # a_1 = F.normalize(a, dim=1)
        # b_1 = F.normalize(b, dim=1)
        # cos_sim = F.cosine_similarity(a_1, b_1, dim=1)
        # return cos_sim
        return F.pairwise_distance(a,b,p=2,eps=1e-6) / a.size(-1)
    

    def forward(self, face, landmark):
        encodered_face = self.encode_face(face)
        encodered_landmark = self.encode_landmark(landmark)
        return self.distance(encodered_face, encodered_landmark)

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


class Action(nn.Module):
    def __init__(self, in_channels=3, window_len=5, shift_div=8,device='cuda:1'):
        super(Action, self).__init__()
        self.window_len = window_len
        self.in_channels = in_channels
        self.reduced_channels = in_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fold = self.in_channels // shift_div
        self.device = device

        # shifting
        self.action_shift = nn.Conv1d(
                                    self.in_channels, self.in_channels,
                                    kernel_size=3, padding=1, groups=self.in_channels,
                                    bias=False)      
        self.action_shift.weight.requires_grad = True
        self.action_shift.weight.data.zero_()
        self.action_shift.weight.data[:self.fold, 0, 2] = 1 # shift left
        self.action_shift.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 # shift right  
        self.action_shift.to(self.device) 


        if 2*self.fold < self.in_channels:
            self.action_shift.weight.data[2 * self.fold:, 0, 1] = 1 # fixed


        # # spatial temporal excitation
        self.action_p1_conv1 = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), 
                                    stride=(1, 1 ,1), bias=False, padding=(1, 1, 1))     
        self.action_p1_conv1.to(self.device)  


        # # channel excitation
        self.action_p2_squeeze = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        self.action_p2_conv1 = nn.Conv1d(self.reduced_channels, self.reduced_channels, kernel_size=3, stride=1, bias=False, padding=1, 
                                       groups=1)
        self.action_p2_expand = nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        
        self.action_p2_squeeze.to(self.device)
        self.action_p2_conv1.to(self.device)
        self.action_p2_expand.to(self.device)
    


        # motion excitation
        self.pad = (0,0,0,0,0,0,0,1)
        self.action_p3_squeeze = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        self.action_p3_bn1 = nn.BatchNorm2d(self.reduced_channels)
        self.action_p3_conv1 = nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(3, 3), 
                                    stride=(1 ,1), bias=False, padding=(1, 1), groups=self.reduced_channels)
        self.action_p3_expand = nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        
        self.action_p3_squeeze.to(self.device)
        self.action_p3_bn1.to(self.device)
        self.action_p3_conv1.to(self.device)
        self.action_p3_expand.to(self.device)
        
        print('=> Using ACTION')


    def forward(self, face_list):
        # nt, c, h, w = x.size()
        # n_batch = nt // self.window_len

        # x_shift = x.view(n_batch, self.window_len, c, h, w)
        
        
        # get the x_shift of: (n_batch, self.window_len, c, h, w)
        x_shift = None
        for i in range(self.window_len):
            single_face = face_list[i].to(self.device).float().unsqueeze_(1)

            if i == 0:
                x_shift = single_face

            else:
                x_shift = torch.concat((x_shift, single_face),1)

        n_batch, nt, c, h, w = x_shift.shape[0], x_shift.shape[0] * x_shift.shape[1], x_shift.shape[2], x_shift.shape[3], x_shift.shape[4]
        
        x_shift = x_shift.permute([0, 3, 4, 2, 1])  # (n_batch, h, w, c, window_len)
        x_shift = x_shift.contiguous().view(n_batch*h*w, c, self.window_len)
        x_shift = self.action_shift(x_shift)  # (n_batch*h*w, c, window_len)
        x_shift = x_shift.view(n_batch, h, w, c, self.window_len)
        x_shift = x_shift.permute([0, 4, 3, 1, 2])  # (n_batch, window_len, c, h, w)
        # x_shift = x_shift.contiguous().view(nt, c, h, w)
        x_shift = x_shift.contiguous()
        

        

        # 3D convolution: c*T*h*w, spatial temporal excitation
        x_p1 = x_shift.view(n_batch, self.window_len, c, h, w).transpose(2,1).contiguous()
        x_p1 = x_p1.mean(1, keepdim=True)
        x_p1 = self.action_p1_conv1(x_p1)
        # x_p1 = x_p1.transpose(2,1).contiguous().view(nt, 1, h, w)
        x_p1 = x_p1.transpose(2,1).contiguous()
        x_p1_w = self.sigmoid(x_p1)
        x_p1 = x_shift * x_p1_w + x_shift


        # 2D convolution: c*T*1*1, channel excitation
        # n,t,c,h,w --> n,t,c,1,1 --> nt,c,1,1
        x_p2 = self.avg_pool(x_shift).contiguous().view(nt, c, 1, 1)
        # nt,c,1,1
        x_p2 = self.action_p2_squeeze(x_p2)
        # nt,c,1,1 --> n,c,t
        x_p2 = x_p2.view(n_batch, self.window_len, c, 1, 1).squeeze(-1).squeeze(-1).transpose(2,1).contiguous()
        # n,c,t --> n,c,t
        x_p2 = self.action_p2_conv1(x_p2)
        x_p2 = self.relu(x_p2)
        # n,c,t --> nt,c,1,1
        x_p2 = x_p2.transpose(2,1).contiguous().view(-1, c, 1, 1)
        x_p2 = self.action_p2_expand(x_p2)
        # nt,c,1,1 --> n,t,c,1,1
        x_p2 = x_p2.view(n_batch, self.window_len, c, 1, 1)
        x_p2_w = self.sigmoid(x_p2)
        x_p2 = x_shift * x_p2_w + x_shift
        

        # # 2D convolution: motion excitation
        x_shift_clone = x_shift.clone().view(nt, c, h, w)
        # nt,c,h,w
        x3 = self.action_p3_squeeze(x_shift_clone)
        #nt,c,h,w
        x3 = self.action_p3_bn1(x3)
        # x3_p0: n,t-1,c,h,w
        x3_plus0, _ = x3.view(n_batch, self.window_len, c, h, w).split([self.window_len-1, 1], dim=1)
        # x3_p1: nt,c,h,w
        x3_plus1 = self.action_p3_conv1(x3)
        # x3_p1: n,t-1,c,h,w --> n,t-1,c,h,w
        _ , x3_plus1 = x3_plus1.view(n_batch, self.window_len, c, h, w).split([1, self.window_len-1], dim=1)
        x_p3 = x3_plus1 - x3_plus0
        # n,t,c,h,w
        x_p3 = F.pad(x_p3, self.pad, mode="constant", value=0)
        # n,t,c,h,w --> nt,c,h,w
        x_p3 = self.avg_pool(x_p3.view(nt, c, h, w))
        # nt,c,h,w
        x_p3 = self.action_p3_expand(x_p3)
        # nt,c,h,w --> n,t,c,h,w
        x_p3 = x_p3.view(n_batch, self.window_len, c, 1, 1)
        x_p3_w = self.sigmoid(x_p3)
        x_p3 = x_shift * x_p3_w + x_shift

        out = x_p1 + x_p2 + x_p3
        
        # 仅在attention可视化时有效
        ###############################################   
        # global COUNT
        # print(f'Here x_shift: {x_shift.shape}')
        # if not os.path.exists('./attention'):
        #     os.mkdir('./attention')
        
        # for index in range(self.window_len):
        #     img = x_shift[0,index,:,:,:].permute(1,2,0).cpu().numpy()
        #     if np.max(img) > 1.5:            
        #         img = img / 255.
        #         img = np.where(img > 1, 1, img)
            
            
        #     # spatial_temporal_mask = x_p1_w[0,index,:,:,:].permute(1,2,0).cpu().numpy()
        #     spatial_temporal_mask = np.absolute(x_p1_w[0,index,:,:,:].permute(1,2,0).cpu().numpy())
        #     spatial_temporal_mask = (spatial_temporal_mask - np.min(spatial_temporal_mask)) / (np.max(spatial_temporal_mask) - np.min(spatial_temporal_mask))


            
        #     # channel_mask = x_p2_w[0,index,:,:,:].permute(1,2,0).cpu().numpy()
        #     # channel_mask = (channel_mask - np.min(channel_mask)) / (np.max(channel_mask) - np.min(channel_mask))
            
        #     # motion_mask = x_p3_w[0,index,:,:,:].permute(1,2,0).cpu().numpy()
        #     # motion_mask = (motion_mask - np.min(motion_mask)) / (np.max(motion_mask) - np.min(motion_mask))
            
        #     # whole_mask = (spatial_temporal_mask + channel_mask + motion_mask) / 3.
            
        #     # print(f'Here spatial_temporal_mask: {spatial_temporal_mask}')
        #     # raise ValueError('out')
        #     cv2.imwrite(f'./attention/src_{COUNT}_{index}.png',img * 255.0)
        #     show_cam_on_image(img,spatial_temporal_mask,f'./attention/spatial_temporal_{COUNT}_{index}.png')
        #     # show_cam_on_image(img,channel_mask,f'./attention/channel_{COUNT}_{index}.png')
        #     # show_cam_on_image(img,motion_mask,f'./attention/motion_{COUNT}_{index}.png')
        #     # show_cam_on_image(img,whole_mask,f'./attention/whole_{COUNT}_{index}.png')
        
        # # raise ValueError('out')
        # COUNT += 1
        ###############################################
        
        
        
        return out


class Alteration(nn.Module):
    def __init__(
        self,
        arch="ViT-L/14",
        shreshold=0,
        window_len=5,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(Alteration, self).__init__()
        self.window_len = window_len
        self.device = device
        self.encoder, preprocess = clip.load(arch, device=device)
        # self.fc = nn.Linear(window_len - 1, 1)  # 基于偏差值做分类, 寄
        # self.shreshold = nn.Parameter(    # 和阈值比较， 寄
        #     torch.tensor(shreshold, requires_grad=True, dtype=torch.float32)
        # )
        self.fc = nn.Linear(768, 1) # 直接合成

    def forward(self, lips):
        features = list()
        similarities = list()
        for i in range(self.window_len):
            features.append(self.encoder.encode_image(lips[i].to(self.device)))

            def cosine_similarity(a, b):
                # 标准化张量 a 和 b
                a_1 = F.normalize(a, dim=1)
                b_1 = F.normalize(b, dim=1)
                cos_sim = F.cosine_similarity(a_1, b_1, dim=1)
                return cos_sim

            if i > 0:
                similarities.append(cosine_similarity(features[-2], features[-1]))

        similarities = torch.stack(similarities, dim=0)
        features = torch.stack(features, dim=0)
        features = features.sum(dim=0).to(torch.float32)

        # 直接用相似度, 寄了
        # pred = self.fc(-similarities.T).sigmoid().flatten()

        # 和阈值比较， 寄了
        # pred = (torch.sum(-similarities, dim=0) / (self.window_len - 1)).sigmoid().flatten()
        # pred = torch.where(pred >= self.shreshold, 1.0, 0.0)

        # 合成
        pred = self.fc(features).sigmoid().flatten()
        return features, similarities, pred


class Combine(nn.Module):
    def __init__(
        self,
        window_len=5,
        fix_backbone=True,
        device="cuda:1",
        face_backbone="wide_resnet50_2",
        scale_error=False,
        Contrastive_ckpt = None,
        Temporal_ckpt = None,
        proj_ckpt = None,
        fix_contrastive = True,
        fix_temporal = False,
        fix_proj = False
    ):
        super(Combine, self).__init__()
        self.contrastive = Contrastive(
            window_len=window_len,
            fix_backbone=fix_backbone,
            device=device,
            face_backbone=face_backbone
        )
        self.contrastive.to(device)
        self.contrastive.load_state_dict(torch.load(Contrastive_ckpt,map_location=torch.device(device)))
        
        self.temporal = Temporal(
            window_len=window_len,
            fix_backbone=fix_backbone,
            device=device,
            face_backbone=face_backbone,
            not_linear=False,
            scale_error=scale_error
        )
        self.temporal.to(device)
        # 不加载最后的sigmoid层
        self.temporal.load_state_dict(torch.load(Temporal_ckpt,map_location=torch.device(device)),strict=False)
        
        
        
        self.proj = nn.Sequential(
            nn.Linear(2,1,bias=False),
            nn.Sigmoid()
        )
        if proj_ckpt != None and proj_ckpt != False:
            self.proj.load_state_dict(torch.load(proj_ckpt,map_location=torch.device(device)))
        
        
        if fix_contrastive:
            for param in self.contrastive.parameters():
                param.requires_grad = False
        
        if fix_temporal:
            for param in self.temporal.parameters():
                param.requires_grad = False
            
        if fix_proj:
            for param in self.proj.parameters():
                param.requires_grad = False
        
        
        
    def forward(self,faces,lips,landmarks):
        out1 = self.contrastive.forward(lips,landmarks).unsqueeze(1)
        out2 = self.temporal.forward(faces)
        
        out = self.proj(torch.concat((out1,out2),-1))
        return out


def show_cam_on_image(img, mask,tgt):
    # heatmap = mask / np.max(mask)
    # heatmap = np.uint8(255 * heatmap)
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    
    # # heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    # # heatmap = np.float32(heatmap) / 255
    # cam = heatmap*0.9 + np.float32(img)
    # cam = cam / np.max(cam)
    
    cam = heat_cam(img, mask, use_rgb=True)
    cv2.imwrite(tgt, cam)
    # raise ValueError('out')
    

def heat_cam(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255


    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    threshold = np.max(heatmap) * 0.95
    cam = np.where(heatmap>threshold, (1 - image_weight) * heatmap + image_weight * img, img)

    # cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
    
COUNT = 0