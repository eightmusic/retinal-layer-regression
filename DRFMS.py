from scipy.io import loadmat
import numpy as np
from glob import glob
from os import path
from matplotlib import colors
import os
import torch
from torch.utils.data import TensorDataset,Dataset,DataLoader,random_split
from einops import rearrange,reduce,repeat
import time
import copy
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
import sys
sys.path.append("..")
from IPython import display
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore") # 忽略警告
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch import optim, nn
import cv2
import random
from torch.autograd import Variable
from sklearn.utils import shuffle
# from relay_net import ReLayNet
from ReLay import Unet
from trans_reg import TransFoot
from resmlp import Resmlp
from train_model import train_model
from hroct import hr
from henetoct import get_seg_model
from sklearn.model_selection import train_test_split
import json
from train_model import evaluation
# from transformer import MixVisionTransformer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# mat_fps = glob(path.join(r'B:\eee\cv\oct\2015_BOE_Chiu\2015_BOE_Chiu/', '*.mat'))
# lab_path='B:\eee\Downloads\oct_preprocess-master\oct_preprocess-master\label/'
# img_path='B:\eee\Downloads\oct_preprocess-master\oct_preprocess-master\image/'
lab_path='B:\eee\Downloads\oct_preprocess\hc\label/'
img_path='B:\eee\Downloads\oct_preprocess\hc\image/'
def read_directory(directory_name):
    array_of_img = []
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(directory_name):
        # print(filename) #just for test
        #img is used to store the image data
        img = cv2.imread(directory_name + "/" + filename,0)
        array_of_img.append(img)
        #print(img)
#         print(array_of_img)
    return np.array(array_of_img,dtype=np.uint8)
img=read_directory(img_path)   #bhw
def read_lab(directory_name):
    array_of_lay = []
    array_of_flu = []
    for filename in os.listdir(directory_name):
        with open(directory_name + "/" + filename) as f:
            # print(filename)
            js = f.read()
            dic = json.loads(js)
            f.close()
        a=dic.get('bds')
        b=dic.get('lesion')
        array_of_lay.append(a)
        array_of_flu.append(b)
    return np.array(array_of_lay,dtype=np.float32),array_of_flu
llay,flu=read_lab(lab_path)    #bnw

def split(data):
    hc_test = data[:392]
    hc_valid = data[392:441]
    hc_train = data[441:686]
    ms_test = data[686:1274]
    ms_valid = data[1274:1372]
    ms_train = data[1372:1715]
    train = np.concatenate((hc_train, ms_train), axis=0)
    valid = np.concatenate((hc_valid, ms_valid), axis=0)
    # test = np.concatenate((hc_test, ms_test), axis=0)
    test=hc_test
    return train, valid, test
def data_img_seg():
    img0=img.reshape(1715,1,128,1024)
    img_train, img_valid, img_test = split(img0)
    lay_train, lay_valid, lay_test = split(llay)
    img_train = rearrange(img_train, 'b c h (w a) ->(b a) c h w', w=64)
    img_valid = rearrange(img_valid, 'b c h (w a) ->(b a) c h w', w=64)
    img_test = rearrange(img_test, 'b c h (w a) ->(b a) c h w', w=1024)
    lay_train = rearrange(lay_train, 'b n (w a)-> (b a) n w ', w=64)
    lay_valid = rearrange(lay_valid , 'b n (w a)-> (b a) n w ', w=64)
    lay_test = rearrange(lay_test , 'b n (w a)-> (b a) n w ', w=1024)
    img_train,lay_train=shuffle(img_train,lay_train)
    # test_img,test_lay=shuffle(test_img,test_lay)
    # print(type(train_img[0,0,0,0]),type(train_lay[0,0,0]))
    # is_na(torch.tensor(train_lay))
    # is_na(torch.tensor(test_lay))
    # print(2)
    train_data = TensorDataset(torch.FloatTensor(img_train[:, :, :, :]), torch.tensor(lay_train))
    valid_data = TensorDataset(torch.FloatTensor(img_valid[:, :, :, :]), torch.tensor(lay_valid))
    test_data = TensorDataset(torch.FloatTensor(img_test[:, :, :, :]), torch.tensor(lay_test))

    return train_data, valid_data,test_data

ds_train,ds_valid,ds_test=data_img_seg()
ba=8
train_loader,val_loader,test_loader=DataLoader(ds_train,batch_size=ba),DataLoader(ds_valid,batch_size=ba),DataLoader(ds_test,batch_size=ba)


if __name__ == '__main__':
    T_path='B:/eee/demo/pycharm/oct/weights/transformer/end/DRFMSce.pkl'
    # M_path='B:/eee/demo/pycharm/oct/weights/mlp/DRFMS.pkl'
    # relaynet_model = TransFoot(image_size=(128, 64), patch_size=4, dim=4, trans_depth=8, heads=8, mlp_dim=16,dim_head=16,num_classes=9,
    #                            channels=1).cuda() #1.218 4 4 8 8 16 16  pool 1.11-1.425-1316m 2  |1126m 3 bs*2 效果差点
    #conv 1.19-1.55
    relaynet_model = TransFoot(image_size=(128, 64), patch_size=4, dim=4, trans_depth=8, heads=8, mlp_dim=16,dim_head=16,num_classes=9,
                               channels=1).cuda()

    # relaynet_model = Resmlp(128, 64, 64, 9).cuda()  #   1.005-1.305  #mlp pool-jiewei 0.98 1.27
    # train_model(relaynet_model, 11, train_loader, val_loader,test_loader,PATH=T_path,n_class=9,l=0.001)

    # relaynet_model.load_state_dict(torch.load('B:/eee/demo/pycharm/oct/weights/transformer/end/DRFMSok.pkl'))
    relaynet_model.load_state_dict(torch.load('B:/eee/demo/pycharm/oct/weights/transformer/end/DRFMSce.pkl'))
    evaluation(relaynet_model,test_loader)

