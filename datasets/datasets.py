import os
import torch
from torch.utils.data import TensorDataset,Dataset,DataLoader,random_split
from einops import rearrange,reduce,repeat
from sklearn.utils import shuffle
import numpy as np
import cv2
import random
import json

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