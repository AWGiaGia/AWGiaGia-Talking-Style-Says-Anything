import os
import warnings
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import *


from dataset.dataset import MyDataset
from model.model import Contrastive
from model.loss import ContrastiveLoss

warnings.filterwarnings("ignore")

import yaml


def seed_anything(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

    torch.manual_seed(seed_value)     # 为CPU设置随机种子
    torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
    torch.cuda.manual_seed_all(seed_value)   # 为所有GPU设置随机种子（多块GPU）

    torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    config_path = './config/csl/train.yaml'
    with open(config_path,'r',encoding='utf-8') as f:
        configs = yaml.load(f,Loader=yaml.FullLoader)

    print(f'EXP Settings: ')
    for k,v in configs.items():
        print(f'{k}: {v}')
    print(f'*'*30)

    seed_anything(configs['seed'])
    
    # 创建模型保存路径
    if not os.path.exists(os.path.join('./weight','csl',configs['name'])):
        os.mkdir(os.path.join('./weight','csl',configs['name']))
    # 创建结果保存路径
    if not os.path.exists(os.path.join('./output','csl','train',configs['name'])):
        os.mkdir(os.path.join('./output','csl','train',configs['name']))
    
    # 将配置文件保存到模型保存路径
    with open(os.path.join('./weight','csl',configs['name'],'train.yaml'),'w') as f:
        f.write(yaml.dump(configs,allow_unicode=True))
    
    # 初始化
    train_dataset = MyDataset(root=configs['train_data_toot'], 
                              real_sample_only=configs['real_sample_only'], 
                              is_train=configs['is_train'], 
                              pos_neg_rate=configs['pos_neg_rate'], 
                              window_len=configs['window_len'], 
                              n_extracts=configs['n_extracts'])
    train_loader = DataLoader(train_dataset, batch_size = configs['batch_size'], shuffle=True, num_workers = configs['num_workers'])
    
    val_dataset = MyDataset(root=configs['val_data_root'], 
                              real_sample_only=False, 
                              is_train=False, 
                              pos_neg_rate=10, 
                              window_len=configs['window_len'], 
                              n_extracts=configs['n_extracts'])
    val_loader = DataLoader(val_dataset, batch_size = configs['batch_size'], shuffle=True, num_workers = configs['num_workers'])
    
    model = Contrastive(window_len=configs['window_len'],
                        fix_backbone=configs['fix_backbone'], 
                        device=configs['device'], 
                        face_backbone=configs['face_backbone'])
    model.to(configs['device'])
    
    if configs['pretrained_ckpt'] != None and configs['pretrained_ckpt'] != False:
        model.load_state_dict(torch.load(configs['pretrained_ckpt']))
    
    
    criterion = ContrastiveLoss(margin=configs['margin'],device=configs['device'])
    opt = optim.Adam(model.parameters(),lr=configs['lr'])
    
    train_loss_curve = []
    for epoch in range(configs['epochs']):
        total_loss = 0.0
        model.train()
        for _,(face, lip, landmark, label, face_label) in tqdm(enumerate(train_loader),desc = f'on training epoch {epoch}'):
            opt.zero_grad()
            d = model(lip,landmark)
            loss = criterion(d,label)
            
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
        train_loss_curve.append(total_loss)
        print(f"Epoch {epoch + 1}/{configs['epochs']}, Loss: {total_loss / len(train_loader)}")
    
        if epoch % configs['val_gap'] == 0 or epoch >= configs['epochs'] - 1:
            model.eval()
            label_list = []
            distance_list = []
            with torch.no_grad():
                for _,(face, lip, landmark, label, face_label) in tqdm(enumerate(val_loader),desc = f'on testing epoch'):
                    label = label.float().to(configs['device'])
                    distance = model(lip,landmark)
                    
                    label_list += label.cpu().float().tolist()
                    distance_list += distance.cpu().float().tolist()
            
            # draw distances
            real_distance = []
            fake_distance = []
            for i in range(len(label_list)):
                if label_list[i] > 0.5:
                    fake_distance.append(distance_list[i])
                else:
                    real_distance.append(distance_list[i])

            real_distance = np.array(real_distance)
            fake_distance = np.array(fake_distance)
            x1 = np.array([i for i in range(len(real_distance))])
            x2 = np.array([i for i in range(len(fake_distance))])
                
            plt.scatter(x1,real_distance,c='r')
            plt.scatter(x2,fake_distance,c='b')
            plt.legend(['real_d','fake_d'])
            plt.savefig(os.path.join('./output','csl','train',configs['name'],f'ep{epoch}_distance.png'))
            # plt.savefig(f'{name}_distance.png')
            plt.clf()
            
            # 保存模型
            torch.save(model.state_dict(),os.path.join('./weight','csl',configs['name'],f'ep{epoch}.pth'))
            
    plt.plot(train_loss_curve)
    plt.savefig(os.path.join('./output','csl','train',configs['name'],f'loss_curve.png'))
    plt.clf()
            
    
                