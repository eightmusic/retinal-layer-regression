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
# from ReLay import Unet
from models.trans_reg import TransFoot
from models.resmlp import Resmlp
from models.train_model import train_model
# from hroct import hr
# from henetoct import get_seg_model
from sklearn.model_selection import train_test_split
import json
from models.train_model import evaluation
# from transformer import MixVisionTransformer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# mat_fps = glob(path.join(r'B:\eee\cv\oct\2015_BOE_Chiu\2015_BOE_Chiu/', '*.mat'))
lab_path='B:\eee\Downloads\oct_preprocess\label1/'
img_path='B:\eee\Downloads\oct_preprocess\image1/'

def get_valid_idx(manualLayer):  #取出里面的[10,15,20,25,28,30,32,35,40,45,50]这些索引
    idx = []
    for i in range(0,61):
        temp = manualLayer[:,:,i]
        if np.sum(temp !=0) != 0:
            idx.append(i)
            #print(np.sum(temp !=0))
    return idx


# def fillna(layer):
#     for n in range(layer.shape[0]):
#         #         print('n',n)
#         for b in range(layer.shape[2]):
#             idx1, idx2 = None, None
#             #             print('b',b)
#             for idx, i in enumerate(layer[n, :, b]):
#                 #     print(type(i)==float)
#                 #                 if idx==0 and not(layer[n,:,b][idx+1]>=0 and layer[n,:,b][idx+1]<310):
#                 #                     idx1=0
#                 if idx + 1 == 768:
#                     break
#
#                 if ((i >= 0 and i < 310) and not (layer[n, :, b][idx + 1] >= 0 and layer[n, :, b][idx + 1] < 310)):  # \
#                     #                 or (idx==0 and not(layer[n,:,b][idx+1]>=0 and layer[n,:,b][idx+1]<310)):
#                     idx1 = idx
#                 #                     print('jin',idx1)
#                 if not (i >= 0 and i < 310) and (
#                         layer[n, :, b][idx + 1] >= 0 and layer[n, :, b][idx + 1] < 310) and idx1:
#                     #                     print('jin')
#                     idx2 = idx + 1
#                 if idx1 and idx2:
#                     il = np.linspace(layer[n, :, b][idx1], layer[n, :, b][idx2], idx2 - idx1 + 1)[1:-1]
#                     layer[n, :, b][idx1 + 1:idx2] = il
#                     idx1, idx2 = None, None
#                     #                     print('succes',idx1,idx2,il,layer[n,:,b][idx1+1:idx2])
#                     # break
#     #     print(mat['manualLayers1'][4,300:330,18])
#     #     print(layer[4,300:330,18])
#     return layer
def fillna(layer):
    # y = layer.shape[1]
    # print(layer.shape)
    na_num=0
    for b in range(layer.shape[2]):
        for n in range(layer.shape[0]):
            idx1, idx2 = None, None
            for idx, i in enumerate(layer[n, :, b]):
                if idx + 1 == 500:
                    break
                if (idx==0) and np.isnan(layer[n, :, b][idx]):
                    idx1=-1
                if (not (np.isnan(i))) and np.isnan(layer[n, :, b][idx+1]):
                    # print('idx1')
                    idx1 = idx
                if np.isnan(i) and not(np.isnan(layer[n, :, b][idx+1])):
                    # print('idx1')
                    idx2 = idx + 1
                if idx==498 and np.isnan(layer[n, :, b][idx+1]):
                    idx2=500
                if (not (idx1 is None)) and (not (idx2 is None)):
                    if (idx1==-1) :
                        # print(layer[n, :, b][idx+1:idx2])
                        layer[n, :, b][idx1+1:idx2] = layer[n, :, b][idx2]
                        # print(layer[n, :, b][idx+1:idx2])
                        idx1, idx2 = None, None
                        # na_num+=1
                        continue
                    if idx2==500:
                        # if na_num==5:
                            # print(layer[n, :, b][idx + 1:idx2])
                        layer[n, :, b][idx1+1:idx2] = layer[n, :, b][idx1]
                        # print(layer[n, :, b][idx + 1:idx2])
                        idx1, idx2 = None, None
                        # na_num += 1
                        continue
                    if (idx1>=0) and (idx2<=499) :
                        il = np.linspace(layer[n, :, b][idx1], layer[n, :, b][idx2], idx2 - idx1 + 1)[1:-1]
                        layer[n, :, b][idx1 + 1:idx2] = il
                        idx1, idx2 = None, None
                        # na_num += 1
                        continue
                        #                     print('succes',idx1,idx2,il,layer[n,:,b][idx1+1:idx2])
                        # break
                # idx1, idx2 = None, None
            # if (idx1) or (idx2):
            #     print(b,n,idx,'error')


    #     print(mat['manualLayers1'][4,300:330,18])
    #     print(layer[4,300:330,18])
    # print(na_num)
    return layer

def data_shrink(img_old,label_old):
    img,label=img_old,label_old
    n_b=224
    for i in range(img.shape[0]):
        # print(i,img.shape)
        for factor in [0.6,0.7,0.8,0.9]:
            h=int(n_b*factor) #factor=0.7-0.9
            # print(h,img[i,0].shape,factor,type(img[i,0,0,0]))
            # plt.imshow(img[i,0],cmap='gray')
            # plt.show()
            img_test = cv2.resize(img[i,0], (500,h))
            # print('sucessful')
            label_test=label[i]*factor
            factor2=(1-factor)
            factor3=random.uniform(0, factor2)
            pad_zero_up=np.random.randint(0,50,(int(factor3*n_b),500))
            pad_zero_down=np.random.randint(0,50,(n_b-h-int(factor3*n_b),500))

            img_s=np.concatenate((pad_zero_up,img_test,pad_zero_down),axis=0)
            label_s=label_test+factor3*n_b
            # print(img_s.shape,label_s.shape)
            re_h,re_w=img_s.shape
            la_h,la_w=label_s.shape
            img_s=np.reshape(img_s,(1,1,re_h,re_w))
            label_s=np.reshape(label_s,(1,la_h,la_w))
            # img_s=rearrange(img_s,'h w->b c h w',b=1,c=1)
            # label_s=rearrange(label_s,'n w->b n w',b=1)
            img_old=np.concatenate((img_old,img_s),axis=0)
            label_old=np.concatenate((label_old,label_s),axis=0)
            # plt.imshow(img_test)
            # plt.show()
            # break
        # break
    return img_old.astype(np.uint8),label_old#.astype(np.float32)
def read_directory(directory_name):
    array_of_img = []
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(directory_name):
        # print(filename) #just for test`
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
    return np.array(array_of_lay,dtype=np.float32),np.array(array_of_flu,dtype=np.uint8)
llay,flu=read_lab(lab_path)    #bnw
def is_na(lay):
    num=0
    for a in range(lay.shape[0]):
        for b in range(lay.shape[1]):
            for c in range(lay.shape[2]):

                if (np.isnan(lay[a,b,c])) :#and (lay[a,b,c]!=0)
                    print(a,b,c)
                    num+=1
                    # break
    print('error:',num)


def data_img_seg():
    # for i in range(10):
    #     mat = loadmat(mat_fps[i])
    #     manualLayer = fillna(mat['manualLayers1'])
    #     manualLayer = np.array(manualLayer, dtype=np.uint16)
    #     manualLayer = manualLayer.astype(np.float32)
    #     img = np.array(mat['images'], dtype=np.uint8)
    #     valid_idx = get_valid_idx(manualLayer)
    #     manualLayer = manualLayer[:, :, valid_idx]
    #     img = img[:360, :, valid_idx]
    #
    #     max_col = -100
    #     min_col = 900
    #     for b_scan_idx in range(0, 11):  # 5
    #         # print(b_scan_idx)
    #         for col in range(768):  # 296
    #             cur_col = manualLayer[:, col, b_scan_idx]
    #             if np.sum(cur_col) == 0:  # 列的和是否为零
    #                 continue
    #                 # 这处col为第一次不为0，即为流体出现的第一列
    #             max_col = max(max_col, col)  # 这处col为最后一此不为0，即为流体出现的最后一列
    #             min_col = min(min_col, col)  # 这处col为第一次不为0，
    #     manualLayer = manualLayer[:, min_col:min_col + 540, :]  # n,w,b
    #     img = img[:, min_col:min_col + 540, :]
    #
    #     if i == 0:
    #         img_all = img
    #         layer_all = manualLayer
    #         continue
    #     img_all = np.concatenate((img_all, img), axis=2)
    #     layer_all = np.concatenate((layer_all, manualLayer), axis=2)

    lay=rearrange(llay,'b n w->n w b')
    # print(lay.shape)
    # print(llay[1, 1, 1])
    # print(llay[1,1])
    #
    # print('**************')
    # is_na(lay)
    lay=fillna(lay)
    # lay=fillna(lay)
    # print (lay)
    layer_all = rearrange(lay, 'n w b -> b n w ')
    img_all = rearrange(img, '(b c) h w -> b c h w ', c=1)
    # train_img, test_img, train_lay, test_lay = train_test_split(img_all, layer_all, test_size=0.5, shuffle=True)
    train_img, test_img,train_lay, test_lay=img_all[55:],img_all[:55],layer_all[55:],layer_all[:55]#105,99,94,88/,83,77,72,66,61,55
    # train_img, test_img,train_lay, test_lay=img_all[:55],img_all[55:],layer_all[:55],layer_all[55:] #mlp 2.48  #transformer 2.81

    is_na(layer_all)
    #############################
    s_img, s_lay = data_shrink(train_img, train_lay)
    train_img=np.concatenate((train_img,s_img),axis=0)
    train_lay=np.concatenate((train_lay,s_lay),axis=0)
    #############################
    # print(train_img.shape, train_lay.shape, type(train_img[0, 0, 0, 0]), type(train_lay[0, 0, 0]))
    # train_lay, test_lay=lay_all[:55,:,:],layer_all[:55,:,:]
    train_img = rearrange(train_img, 'b c h (w a) ->(b a) c h w', w=100)
    test_img = rearrange(test_img, 'b c h (w a) ->(b a) c h w', w=100)
    train_lay = rearrange(train_lay, 'b n (w a)-> (b a) n w ', w=100)
    test_lay = rearrange(test_lay, 'b n (w a)-> (b a) n w ', w=100)
    # re_img,re_layer=train_img[:,:,:,::-1],train_lay[:,:,::-1]
    # train_img=np.concatenate((train_img, re_img), axis=0)
    # train_lay=np.concatenate((train_lay, re_layer), axis=0)

    train_img,train_lay=shuffle(train_img,train_lay)
    # test_img,test_lay=shuffle(test_img,test_lay)
    # print(type(train_img[0,0,0,0]),type(train_lay[0,0,0]))
    # is_na(torch.tensor(train_lay))
    # is_na(torch.tensor(test_lay))
    # print(2)
    train_data = TensorDataset(torch.FloatTensor(train_img[:, :, :, :]), torch.tensor(train_lay))
    test_data = TensorDataset(torch.FloatTensor(test_img[:, :, :, :]), torch.tensor(test_lay))

    return train_data, test_data

ds_train,ds_valid=data_img_seg()
train_loader,val_loader=DataLoader(ds_train,batch_size=12),DataLoader(ds_valid,batch_size=12)

if __name__ == '__main__':
    param ={
            'num_channels':1,
            'num_filters':64,
            'kernel_h':7,
            'kernel_w':3,
            'kernel_c': 1,
            'stride_conv':1,
            'pool':2,
            'stride_pool':1,
            'num_class':8
        }
    # relaynet_model = ReLayNet(param).cuda()
    # relaynet_model = Unet(1,8).cuda()
    # relaynet_model = TransFoot(image_size=(350, 30), patch_size=5, dim=300, trans_depth=4, heads=6, mlp_dim=600,
    #                 channels=1).cuda()    #dim:5->300可能有些高了，c*p_s->dim,他们的算法一般不变或者略微长点，mlp_dim一般是dim的四倍

                               # patch_size=8, dim=4, trans_depth=4, heads=4, mlp_dim=32,dim_head=4,

    # relaynet_model = TransFoot(image_size=(224, 50), patch_size=8, dim=4, trans_depth=4, heads=4, mlp_dim=32,
    #                            channels=1).cuda()
    # relaynet_model = TransFoot(image_size=(224, 50), patch_size=7, dim=10, trans_depth=6, heads=8, mlp_dim=40,
    #                            channels=1).cuda()
    #100-0.004,#adam 3.875 adamw3.82375  #relu+adamw3.7655
    # relaynet_model = TransFoot(image_size=(360, 30), patch_size=5, dim=10, trans_depth=6, heads=8, mlp_dim=40,
    #                            channels=1).cuda()
    # relaynet_model = TransFoot(image_size=(360, 30), patch_size=5, dim=360, trans_depth=6, heads=8, mlp_dim=1440,
    #                            channels=1).cuda()

    # relaynet_model =Resmlp(360,180,180,8).cuda()#180 8.81925
    # relaynet_model =Resmlp(224,112,112,8).cuda()#0.879
    # relaynet_model=hr().cuda()

    # relaynet_model=get_seg_model().cuda()


    # def weight_init(m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.xavier_normal_(m.weight)
    #         # nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.Conv2d):
    #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #     elif isinstance(m, nn.BatchNorm2d):
    #         nn.init.constant_(m.weight, 1)
    #         nn.init.constant_(m.bias, 0)
    #
    #
    # relaynet_model.apply(weight_init)
    # relaynet_model=MixVisionTransformer(img_size=224, patch_size=16, in_chans=1, num_classes=8, embed_dims=[32,64, 128],
    #              num_heads=[2, 4, 8], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
    #              attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
    #              depths=[3,2,2], sr_ratios=[2, 2, 1],str=[2,2,1]).cuda()#depths=[3, 4, 6, 3]
    # T_path='B:/eee/demo/pycharm/oct/weights/transformer/end/DED.pkl'#训练好的
    # T_path = 'B:/eee/demo/pycharm/oct/weights/transformer/end/DED1.pkl'#61-123
    # T_path = 'B:/eee/demo/pycharm/oct/weights/transformer/end/DED2.pkl'#66
    # T_path = 'B:/eee/demo/pycharm/oct/weights/transformer/end/DED3.pkl'  #
    # T_path = 'B:/eee/demo/pycharm/oct/weights/transformer/end/DED4.pkl'
    # T_path = 'B:/eee/demo/pycharm/oct/weights/transformer/end/DED5.pkl'
    # T_path = 'B:/eee/demo/pycharm/oct/weights/transformer/end/DED6.pkl'
    # T_path = 'B:/eee/demo/pycharm/oct/weights/transformer/end/DED7.pkl'
    T_path = 'B:/eee/demo/pycharm/oct/weights/transformer/end/DEDce.pkl'
    # T_path = 'B:/eee/demo/pycharm/oct/weights/transformer/end/DED9.pkl'
    M_path='B:/eee/demo/pycharm/oct/weights/mlp/DED1.pkl'

    relaynet_model = TransFoot(image_size=(224, 100), patch_size=4, dim=4, trans_depth=8, heads=8, mlp_dim=16,dim_head=16,num_classes=8,
                               channels=1).cuda() #3.692 7 4 8 8 16 16
    #3.150-4.01875 4,4,8,8,16,16  3.25
    # relaynet_model =Resmlp(224,112,112,8).cuda()#0.879
    # train_model(relaynet_model,300,train_loader,val_loader,PATH=T_path,n_class=8,l=0.0008)
    # relaynet_model.load_state_dict(torch.load('B:/eee/demo/pycharm/oct/weights/transformer/end/DED1.pkl'))
    # relaynet_model.load_state_dict(torch.load('B:/eee/demo/pycharm/oct/weights/mlp/DED.pkl'))
    relaynet_model.load_state_dict(torch.load('B:/eee/demo/pycharm/oct/weights/transformer/end/DED9.pkl'))
    # relaynet_model.load_state_dict(torch.load('B:/eee/demo/pycharm/oct/weights/transformer/DED-a-1.0.pkl'))
    evaluation(relaynet_model,val_loader)

    #t-p atch_size=2, dim=2, trans_depth=8, heads=8, mlp_dim=8,dim_head=8,num_classes=8,
    # imq=cv2.imread('B:\eee\Downloads\oct_preprocess\image1\Subject_01_1.png',0)
    # out=relaynet_model()