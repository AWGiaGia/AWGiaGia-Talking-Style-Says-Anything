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
from utils.metric import csl_metric,metric

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
    config_path = './config/csl/test.yaml'
    with open(config_path,'r',encoding='utf-8') as f:
        configs = yaml.load(f,Loader=yaml.FullLoader)

    print(f'EXP Settings: ')
    for k,v in configs.items():
        print(f'{k}: {v}')
    print(f'*'*30)

    seed_anything(configs['seed'])
    
    # 创建结果保存路径
    if not os.path.exists(os.path.join('./output','csl','test',configs['name'])):
        os.mkdir(os.path.join('./output','csl','test',configs['name']))
    
    # 初始化
    test_dataset = MyDataset(
        root = configs['test_data_root'],
        real_sample_only=False,
        is_train=False,
        pos_neg_rate=10,
        window_len=configs['window_len'],
        n_extracts=configs['n_extracts']
    )
    test_loader = DataLoader(test_dataset,batch_size=configs['batch_size'],shuffle=True,num_workers=configs['num_workers'])
    
    model = Contrastive(window_len=configs['window_len'],
                        fix_backbone=True,
                        device=configs['device'], 
                        face_backbone=configs['face_backbone'])
    model.load_state_dict(torch.load(configs['checkpoint'], map_location=torch.device(configs['device'])))
    model.to(configs['device'])
    model.eval()
    
    confidence = configs['confidence'] # i = fake if distance[i] >= confidence else i = real
    
    logits = []
    preds = []
    with torch.no_grad():
        for _,(face, lip, landmark, label, face_label) in tqdm(enumerate(test_loader),desc = f'testing'):
            label = label.float().to(configs['device'])
            distance = model(lip,landmark)
            
            
            # pred = torch.where(distance > confidence, 1.0, 0.0)
            
            
            # print(f'Here pred: {pred}')
            # print(f'here logits: {label}')
            # print(f'shape of pred: {pred.shape}')
            # print(f'shape of logits: {label.shape}')
            
            preds += distance.cpu().float().tolist()
            logits += label.cpu().float().tolist()
    
    

    
    preds = np.array(preds)
    logits = np.array(logits)
    
    # print(f'Here preds are: {preds}')
    # print(f'Here logits are: {logits}')
    # print(f'shape of preds: {preds.shape}')
    # print(f'shape of logits: {logits.shape}')
    # # raise ValueError('out')
    
    acc,fnr,fpr,ap,roc = metric(preds,logits,confidence=confidence)
    
    print(f'fnr is: {fnr}')
    print(f'fpr is: {fpr}')
    print(f'acc is: {acc}')
    print(f'ap is: {ap}')
    print(f'roc is: {roc}')
    
    
