import os
import random

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, root=None, real_sample_only=False, is_train=True, pos_neg_rate=10, window_len=5,n_extracts=10):
        self.root = root
        
        real_sample = os.listdir(os.path.join(root, 'faces', '0_real'))
        fake_sample = os.listdir(os.path.join(root, 'faces', '1_fake'))
        
        self.real_faces = [os.path.join(root, 'faces', '0_real', i) for i in real_sample ]
        self.fake_faces = [os.path.join(root, 'faces', '1_fake', i) for i in fake_sample]
        self.real_landmarks = [os.path.join(root, 'landmarks', '0_real', i.replace('png','npy')) for i in real_sample]
        self.fake_landmarks = [os.path.join(root, 'landmarks', '1_fake', i.replace('png','npy')) for i in fake_sample]


        self.faces_label = dict()
        # self.landmarks_label = dict()
        self.real_sample_only = real_sample_only
        self.is_train = is_train
        self.window_len = window_len
        self.n_extracts = n_extracts

        self.pos_neg_rate = pos_neg_rate

        if is_train and real_sample_only:
            self.total_faces = self.real_faces
            self.total_landmarks = self.real_landmarks

        else:
            self.total_faces = self.real_faces + self.fake_faces
            self.total_landmarks = self.real_landmarks + self.fake_landmarks

        for i in self.real_faces:
            self.faces_label[i] = 0.0
        for i in self.fake_faces:
            self.faces_label[i] = 1.0

            # for i in self.real_landmarks:
            #     self.landmarks_label[i] = 0.0
            # for i in self.fake_landmarks:
            #     self.landmarks_label[i] = 1.0

        # print(self.faces_label)
        # raise ValueError('out')

        if len(self.total_faces) != len(self.total_landmarks):
            print(f'lenght of faces are: {len(self.total_faces)}')
            print(f'lenght of landmarks are: {len(self.total_landmarks)}')

            raise ValueError(f'Numbers not equal error!')
        

    def __len__(self):
        return len(self.total_faces)

    def __getitem__(self, idx):
        index1 = idx
        index2 = idx
        
        if self.faces_label[self.total_faces[index1]] > 0.5:
            face_label = 1.0 # fake
        else:
            face_label = 0.0 # real

        # on training
        if self.is_train:
            if self.real_sample_only:
                if random.random() < self.pos_neg_rate:
                    index2 = idx
                    label = 0.0 # real
                else:
                    index2 = -1
                    label = 1.0 # fake
            else:
                if self.faces_label[self.total_faces[index1]] > 0.5:  # 防止浮点数取等误差，>0.5即==1.0
                    label = 1.0  # fake
                else:
                    if random.random() < self.pos_neg_rate:
                        index2 = idx
                        label = 0.0  # real
                    else:
                        index2 = -1
                        label = 1.0 # fake
        # on testing
        else:
            if self.faces_label[self.total_faces[index1]] > 0.5:
                label = 1.0  # fake
            else:
                label = 0.0  # true
        
        face, lip = self.process_face(self.total_faces[index1])
        
        if index2 != -1:
            landmark_path = self.total_landmarks[index2]
        else:
            landmark_path = self.select_from_group(index1)
        
        landmark = self.process_landmark(landmark_path)
        
        
        
        return face, lip, landmark, label, face_label
        # return face, landmark, label, face_label


    def process_face(self, img_path):
        img = torch.tensor(cv2.imread(img_path), dtype=torch.float32)
        img = img.permute(2, 0, 1)
        crops = img
        # crops = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
        #                              std=[0.26862954, 0.26130258, 0.27577711])(img)
        # crop images
        # crops[0]: 1.0x, crops[1]: 0.65x, crops[2]: 0.45x
        crops = [[transforms.Resize((224, 224))(img[:, 500:, i*500:i*500 + 500]) for i in range(self.window_len)], [], []]
        
        # print(f'Here crops0 is of: {crops[0].shape}')
        # raise ValueError('out')
        
        crop_idx = [(28, 196), (61, 163)]
        for i in range(len(crops[0])):
            crops[1].append(transforms.Resize((224, 224))
                            (crops[0][i][:, crop_idx[0][0]:crop_idx[0][1], crop_idx[0][0]:crop_idx[0][1]]))
            crops[2].append(transforms.Resize((224, 224))
                            (crops[0][i][:, crop_idx[1][0]:crop_idx[1][1], crop_idx[1][0]:crop_idx[1][1]]))
        img = transforms.Resize((1120, 1120))(img)

        return crops[1],crops[2]

    def process_landmark(self, landmark_path):
        landmark = np.load(landmark_path)
        landmark = landmark.flatten()

        # 加上位置编码
        position_encoding = self.positional_encoding(landmark)
        landmark_with_position = landmark + position_encoding.flatten()

        return landmark_with_position

    def positional_encoding(self, data):
        # 余弦编码
        position = np.arange(0, data.shape[0], 2)  # 使用 shape[0] 获取一维数组的长度
        div_term = np.exp(np.arange(0, data.shape[0], 2) * -(np.log(10000.0) / data.shape[0]))
        position_encoding = np.zeros(data.shape)
        position_encoding[0::2] = np.sin(position * div_term)
        position_encoding[1::2] = np.cos(position * div_term)
        return position_encoding


    def select_from_group(self,idx):
        path = self.total_faces[idx].split(os.sep)
        filename = path[-1]
        try:
            group = int(filename.split('_')[0])
            idx1 = int(filename.split('_')[1].split('.')[0])
        except:
            print(f'error filepath is: {self.total_faces[idx]}')
            raise ValueError(f'out')
        
        idx2 = random.randint(0, self.n_extracts - 1)
        while idx2 == idx1:
            idx2 = random.randint(0, self.n_extracts - 1)
        
        path[0] = '//'
        path[-1] = f'{group}_{idx2}.npy'
        path[-3] = 'landmarks'
        landmark_path = os.path.join(*path)
        # landmark_path = os.path.join(os.path.split(self.total_faces[idx])[0],f'{group}_{idx2}.png')
        return landmark_path
        
   
    
if __name__ == '__main__':
    # set = MyDataset(root='/ljw/Main/PoseSync/datasets/sample')
    # loader = DataLoader(set,batch_size=1,shuffle=True,num_workers=0)
    # for _,(face, landmark, label, face_label,face_path,landmark_path) in enumerate(loader):
    #     print(f'{type(face)}')
    #     print(f'{type(landmark)}')
    #     print(f'label is: {True if label < 0.5 else False}')
    #     print(f'face_label is: {True if face_label < 0.5 else False}')
    #     print(face_path)
    #     print(landmark_path)
    # print(f'out')
    index = 0
    for _,(face, lip, landmark, label, face_label) in enumerate(loader):
        
        face_sample = (face[1] - face[0]).squeeze(0).permute(1,2,0).cpu().numpy()
        cv2.imwrite(f'residual{index}.png',face_sample)
        index += 1
        if index >= 1:
            break
        
        # lip_sample = lip[0].squeeze(0).permute(1,2,0).cpu().numpy()
        # cv2.imwrite('lip.png',lip_sample)
        
        # print(f'{type(face)}')
        # print(f'{type(landmark)}')
        # print(f'label is: {True if label < 0.5 else False}')
        # print(f'face_label is: {True if face_label < 0.5 else False}')
    print(f'out')
    # index = 0
    
    # set = MyDataset(root='/ljw/Main/PoseSync/datasets/sample',
    #                 real_sample_only=True,
    #                 is_train=True,
    #                 pos_neg_rate=0.5)
    # loader = DataLoader(set,batch_size=1,shuffle=True,num_workers=0)
    # for _,(face, lip, landmark, label, face_label,face_path,landmark_path) in enumerate(loader):
        
    #     print(f'({index}) face_path is: {face_path}')
    #     print(f'({index}) landmark_path is: {landmark_path}')
    #     print(f'({index}) label is: {label}')
    #     index += 1
    # print(f'out')
    
