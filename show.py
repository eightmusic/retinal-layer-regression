import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import PIL
import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from models import TransFoot
import scipy
from scipy.signal import savgol_filter

#B:\eee\Downloads\oct_preprocess\hc\image\hc01_spectralis_macula_v1_s1_R_1.png
#B:\eee\Downloads\oct_preprocess\hc\label\hc01_spectralis_macula_v1_s1_R_1.txt

# imt = cv2.imread('B:\eee\Downloads\oct_preprocess\image1\Subject_01_2.png', 0)
imt = cv2.imread('B:\eee\Downloads\oct_preprocess\hc\image\hc01_spectralis_macula_v1_s1_R_1.png', 0)
print(imt.shape)
#128,1024
# plt.imshow(imt)
# plt.show()
# imq=torch.Tensor(np.array(imt).reshape(1,1,224,500))#.cuda()
imq=torch.Tensor(np.array(imt).reshape(1,1,128,1024))#.cuda()
relaynet_model = TransFoot(image_size=(128, 1024), patch_size=4, dim=4, trans_depth=8, heads=8, mlp_dim=16, dim_head=16,
                           num_classes=9,
                           channels=1)#.cuda()  # 3.692 7 4 8 8 16 16
# relaynet_model.load_state_dict(torch.load('B:/eee/demo/pycharm/oct/weights/transformer/DED-a-1.0.pkl'))
# relaynet_model.load_state_dict(torch.load('B:/eee/demo/pycharm/oct/weights/transformer/DED.pkl'))
relaynet_model.load_state_dict(torch.load('B:/eee/demo/pycharm/oct/weights/transformer/end/DRFMSce.pkl'))
out= relaynet_model(imq)
print(out.shape)
# with open('B:\eee\Downloads\oct_preprocess\label1\Subject_01_2.txt') as f:
with open('B:\eee\Downloads\oct_preprocess\hc\label\hc01_spectralis_macula_v1_s1_R_1.txt') as f:
    # print(filename)
    js = f.read()
    dic = json.loads(js)
    f.close()
a = np.array(dic.get('bds'),dtype=np.float32)
b = dic.get('lesion')
# print(a.shape)
# print(a[:,1])
def draw(lay):
    x=range(1024)
    print(lay.shape)
    plt.plot(x,(lay[0,:]),color='cornflowerblue')
    plt.plot(x,(lay[1,:]),color='orange')
    plt.plot(x,(lay[2,:]),color='green')
    plt.plot(x,(lay[3,:]),color='red')
    plt.plot(x,(lay[4,:]),color='mediumslateblue')
    plt.plot(x,(lay[5,:]),color='chocolate')
    plt.plot(x,(lay[6,:]),color='hotpink')
    plt.plot(x,(lay[7,:]),color='darkblue')
    plt.plot(x, (lay[8, :]),color='yellow')
    # plt.plot(x,llay[0,8,:])
    plt.imshow(imt,cmap='gray')
    plt.show()
def sm(line):
    tmp_smooth = savgol_filter(line, 47, 4)  #53,3
    return tmp_smooth
def smoothdraw(lay):
    x=range(1024)
    print(lay.shape)
    plt.plot(x,sm(lay[0,:]),color='cornflowerblue')
    plt.plot(x,sm(lay[1,:]),color='orange')
    plt.plot(x,sm(lay[2,:]),color='green')
    plt.plot(x,sm(lay[3,:]),color='red')
    plt.plot(x,sm(lay[4,:]),color='mediumslateblue')
    plt.plot(x,sm(lay[5,:]),color='chocolate')
    plt.plot(x,sm(lay[6,:]),color='hotpink')
    plt.plot(x,sm(lay[7,:]),color='darkblue')
    plt.plot(x, sm(lay[8, :]),color='yellow')
    # plt.plot(x,llay[0,8,:])
    plt.imshow(imt,cmap='gray')
    plt.show()
ooupt=out[0,0].detach().numpy()
# ooupt= savgol_filter(ooupt, 45,3)
draw(a)
draw(ooupt)
smoothdraw(ooupt)

def ze(pred,true):
    # pred=torch.round(pred)
    diff=pred[1:]-pred[:-1]
    diff=diff[1:]-diff[:-1]
    diff1=true[1:]-true[:-1]
    diff1=diff1[1:]-diff1[:-1]
    diff0=diff1-diff
    out=torch.mean(diff0**2)
    print(diff1,diff)
    return out
############################################################
# a=np.array([2,3,4,5,4,3,2])
# b=np.array([2.1,2.9,4.1,4.9,4.1,2.9,3.1,1.9])

# a1=ze(torch.Tensor(a))
# b1=ze(torch.Tensor(b))
# print(a1,b1)
# a=torch.Tensor(a[0])
# b=torch.Tensor(ooupt[0])
# # qq=ze(a)
# qq1=ze(b,a)
# print(qq1)

# m = nn.ReplicationPad2d((1,1,0,0))
# input = torch.arange(9).reshape(1, 1, 3, 3).float()
# print(m(input))
# k=torch.Tensor([[[[1/3,1/3,1/3]]]])
# print(k.shape)
# kk=torch.repeat_interleave(k,5,0)
# print(kk.shape)
# print(kk)
# # x=torch.randn((2,1,15,50))
# # x = nn.ReplicationPad2d((1, 1, 0, 0))(x)
# # print(x.shape)
# # k = torch.Tensor([[[[1 / 3, 1 / 3, 1 / 3]]]])
# # k = torch.repeat_interleave(k, 5, 0)
# # out = F.conv2d(x, k, stride=1)
# # print(out.shape)
# l=torch.Tensor([[1/3,1/3,1/3]])
# print('l',l.shape)


#
# import numpy as np
# import matplotlib.pyplot as plt
#
# x = np.linspace(0,2*np.pi,100)
# y = np.sin(x) + np.random.random(100) * 0.2
# yhat = savitzky_golay(y, 51, 3) # window size 51, polynomial order 3
#
# plt.plot(x,y)
# plt.plot(x,yhat, color='red')
# plt.show()
# score=torch.Tensor([[0.,0.5],[1.,2.]])
# print(score)
# # # if score[0,1]>0.1:
# # #     print(score[0,0])
# # # print(type(score[0,0]))
# s=torch.where(score>1,score,torch.tensor(0.))
#
# print(s)
# a = torch.tensor([[0.0349,  0.0670, -0.0612, 0.0280, -0.0222,  0.0422],
#          [-1.6719,  0.1242, -0.6488, 0.3313, -1.3965, -0.0682],
#          [-1.3419,  0.4485, -0.6589, 0.1420, -0.3260, -0.4795]])
# b = torch.tensor([[-0.0658, -0.1490, -0.1684, 0.7188,  0.3129, -0.1116],
#          [-0.2098, -0.2980,  0.1126, 0.9666, -0.0178,  0.1222],
#          [ 0.1179, -0.4622, -0.2112, 1.1151,  0.1846,  0.4283]])
# cc = torch.where(a>0,a,b)     #合并a,b两个tensor，如果a中元素大于0，则c中与a对应的位置取a的值，否则取b的值
# print(cc)
# k = torch.Tensor([[[[1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]]]])
# k.requires_grad=True
# print(k.requires_grad)
# x1=torch.randn((1,1,7,7))
# out=F.conv2d(x1,k,stride=1)
# print(out.requires_grad)