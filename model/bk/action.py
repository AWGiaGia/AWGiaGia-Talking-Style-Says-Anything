import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb


class Action(nn.Module):
    def __init__(self, net, window_len=5, shift_div=8,device='cpu'):
        super(Action, self).__init__()
        self.device = device
        self.net = net
        self.window_len = window_len
        self.in_channels = self.net.in_channels
        self.out_channels = self.net.out_channels
        self.kernel_size = self.net.kernel_size
        self.stride = self.net.stride
        self.padding = self.net.padding
        self.reduced_channels = self.in_channels//16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fold = self.in_channels // shift_div

        # shifting
        self.action_shift = nn.Conv1d(
                                    self.in_channels, self.in_channels,
                                    kernel_size=3, padding=1, groups=self.in_channels,
                                    bias=False)      
        self.action_shift.weight.requires_grad = True
        self.action_shift.weight.data.zero_()
        self.action_shift.weight.data[:self.fold, 0, 2] = 1 # shift left
        self.action_shift.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 # shift right  


        if 2*self.fold < self.in_channels:
            self.action_shift.weight.data[2 * self.fold:, 0, 1] = 1 # fixed


        # # spatial temporal excitation
        self.action_p1_conv1 = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), 
                                    stride=(1, 1 ,1), bias=False, padding=(1, 1, 1))       




    def forward(self, face_list):
        # 这里改一下       
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
        # nt, c, h, w = x_shift.size()
        # n,t,c,h,w --> n,c,t,h,w
        x_p1 = x_shift.view(n_batch, self.window_len, c, h, w).transpose(2,1).contiguous()
        # n,c,t,h,w --> n,1,t,h,w
        x_p1 = x_p1.mean(1, keepdim=True)
        # n,1,t,h,w -- > n,1,t,h,w
        x_p1 = self.action_p1_conv1(x_p1)
        # n,1,t,h,w --> n,t,1,h,w --> nt,1,h,w
        # x_p1 = x_p1.transpose(2,1).contiguous().view(nt, 1, h, w)
        x_p1 = x_p1.transpose(2,1).contiguous()
        x_p1 = self.sigmoid(x_p1)
        x_p1 = x_shift * x_p1 + x_shift

        out = x_p1
        return out





# class TemporalPool(nn.Module):
#     def __init__(self, net, window_len):
#         super(TemporalPool, self).__init__()
#         self.net = net
#         self.window_len = window_len

#     def forward(self, x):
#         x = self.temporal_pool(x, window_len=self.window_len)
#         return self.net(x)

#     @staticmethod
#     def temporal_pool(x, window_len):
#         nt, c, h, w = x.size()
#         n_batch = nt // window_len
#         x = x.view(n_batch, window_len, c, h, w).transpose(1, 2)  # n, c, t, h, w
#         x = F.max_pool3d(x, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
#         x = x.transpose(1, 2).contiguous().view(nt // 2, c, h, w)
#         return x


# def make_temporal_shift(net, window_len, n_div=8, place='blockres', temporal_pool=False):
#     if temporal_pool:
#         n_segment_list = [window_len, window_len // 2, window_len // 2, window_len // 2]
#     else:
#         n_segment_list = [window_len] * 4
#     assert n_segment_list[-1] > 0
#     print('=> window_len per stage: {}'.format(n_segment_list))


#     # pdb.set_trace()
#     import torchvision
#     if isinstance(net, torchvision.models.ResNet):
#         if place == 'block':
#             def make_block_temporal(stage, this_segment):
#                 blocks = list(stage.children())
#                 print('=> Processing stage with {} blocks'.format(len(blocks)))
#                 for i, b in enumerate(blocks):
#                     blocks[i].conv1 = Action(b.conv1, window_len=this_segment, shift_div = n_div)
#                 return nn.Sequential(*(blocks))

#             pdb.set_trace()
#             net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
#             net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
#             net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
#             net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

#         elif 'blockres' in place:
#             n_round = 1
#             if len(list(net.layer3.children())) >= 23:
#                 n_round = 2
#                 print('=> Using n_round {} to insert temporal shift'.format(n_round))

#             def make_block_temporal(stage, this_segment):
#                 blocks = list(stage.children())
#                 print('=> Processing stage with {} blocks residual'.format(len(blocks)))
#                 for i, b in enumerate(blocks):
#                     if i % n_round == 0:
#                         blocks[i].conv1 = Action(b.conv1, window_len=this_segment, shift_div = n_div)
#                         # pdb.set_trace()
#                 return nn.Sequential(*blocks)

#             # pdb.set_trace()
#             net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
#             net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
#             net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
#             net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])
             
#     else:
#         raise NotImplementedError(place)


# def make_temporal_pool(net, window_len):
#     import torchvision
#     if isinstance(net, torchvision.models.ResNet):
#         print('=> Injecting nonlocal pooling')
#         net.layer2 = TemporalPool(net.layer2, window_len)
#     else:
#         raise NotImplementedError






