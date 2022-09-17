import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error
import torch
from torch import nn
import math
from einops import rearrange,reduce,repeat
import torch.nn.functional as F
import copy
from scipy.signal import savgol_filter

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss


def acc(logits, labels):
    # print('lab',labels.shape)
    #     print(logits.shape,labels.shape)
    #     if logits.shape[0] !=labels.shape[0] :
    #         print(logits.shape,labels.shape)
    #         print('shape error')
    #     print(torch.sum(logits.data==labels.data),labels.shape[0]*labels.shape[1]*labels.shape[2])
    a = torch.sum(logits.data == labels.data) / (labels.shape[1] * labels.shape[2])
    return a


def rmse(y_pred, y_true):
    y_pred, y_true = y_pred.cpu().detach().numpy(), y_true.cpu().detach().numpy()
    y_pred = rearrange(y_pred, 'b c h w ->(b c) h w')
    rmse = [0.0 for i in range(y_pred.shape[1])]
    for i in range(y_pred.shape[0]):
        for j in range(y_pred.shape[1]):
            # print(type(y_true[i, j]))
            # print(y_true[i, j])
            # print(y_true[i, j,0:6])
            # y_true[i, j,0:5]=[0,0,0,0,0]
            # print(y_true[i, j, 0:6])
            # is_na(y_true[i, j])
            # print('---------------------------')
            rmse[j] += np.sqrt(mean_squared_error(y_pred[i, j], y_true[i, j]))
            # try:
            #     rmse[j]+=np.sqrt(mean_squared_error(y_pred[i,j],y_true[i,j]))
            # except ValueError:
            #     print('pred',y_pred[i,j])
            #     print('----------------------------------------')
            #     print('true',y_true[i,j])
            #     print(i,j)
            # "Error: 没有找到文件或读取文件失败"

            # rmse[j]+=np.sqrt(mean_squared_error(y_pred[i,j],y_true[i,j]))
    return [x / y_pred.shape[0] for x in rmse]


#     return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_pred, y_true):
    y_pred, y_true = y_pred.cpu().detach().numpy(), y_true.cpu().detach().numpy()
    y_pred = rearrange(y_pred, 'b c h w ->(b c) h w')
    #     print(y_pred.shape,y_true.shape)
    mae = [0.0 for i in range(y_pred.shape[1])]
    for i in range(y_pred.shape[0]):
        for j in range(y_pred.shape[1]):
            # is_na(y_true[i,j])
            mae[j] += mean_absolute_error(y_pred[i, j], y_true[i, j])

    return [x / y_pred.shape[0] for x in mae]
    # return [x for x in mae]


def per_class_dice(y_pred, y_true, num_class):
    avg_dice = 0
    y_pred = y_pred.data.cpu().numpy()
    y_true = y_true.data.cpu().numpy()
    for i in range(num_class):
        GT = y_true == (i + 1)
        Pred = y_pred == (i + 1)
        inter = np.sum(np.matmul(GT, Pred)) + 0.0001
        union = np.sum(GT) + np.sum(Pred) + 0.0001
        t = 2 * inter / union
        avg_dice = avg_dice + (t / num_class)
    return avg_dice
def criteon(logit ,lab ,epoch ,n_classes=10):
    c1 =nn.CrossEntropyLoss(weight=torch.tensor((0.25 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,0.25 ,1)).cuda())
    c2 =DiceLoss(n_classes)
    # c3=FocalLoss()
    # c4=nn.MSELoss()
    # c5=F.smooth_l1_loss(logit,lab)
    logit = rearrange(logit, 'b c h w ->(b c) h w')
    c6 = nn.KLDivLoss(size_average=True, reduce=True)  # KL散度
    # if epoch<3:
    # loss=loss_compu(logit,lab,c4)
    # else:
    #     loss = loss_compu1(lab,logit,c6)

    # return 0.5*c1(logit,lab.long())+0.5*c2(logit,lab)#+c3(logit,lab.long())
    #     return 0.5*c2(logit,lab)+0.5*c3(logit,lab.long())
    #     return 0.5*c2(logit,lab)+0.5*c1(logit,lab.long())
    #     return loss#c5
    #     return c4(logit.float(),lab)
    #     return c6(logit,lab)
    #     loss=loss_compu(logit.float(), lab, c4)
    #     return #loss_compu(logit.float(),lab,c4)
    loss =loss_compu(logit ,lab)
    # print(type(loss))
    return loss
    # return c6(logit,lab)

def loss_compu1(y_pred ,y_true):
    #     y_pred=rearrange(y_pred,'b c h w ->(b c) h w')
    #     loss=0.0
    #     for i in range(y_pred.shape[0]):
    #         for j in range(y_pred.shape[1]):
    #             loss+=F.smooth_l1_loss(y_pred[i,j],y_true[i,j])   #这么计算和直接计算的区别
    # #             loss+=c6(y_pred[i,j],y_true[i,j])
    #     return loss/y_pred.shape[0]

    y_pred =rearrange(y_pred ,'b c h w ->(b c) h w')
    loss =0.0
    for i in range(y_pred.shape[0]):
        for j in range(y_pred.shape[1]):
            # print(y_pred[i,:,j].shape,y_true[i,:,j].shape)
            loss+=F.smooth_l1_loss(y_pred[i ,j ,:] ,y_true[i ,j ,:])
    #             loss+=c6(y_pred[i,j],y_true[i,j])
    return loss /y_pred.shape[0]

def ze(pred,true):
    # pred=torch.round(pred)
    diff=torch.sum(torch.abs(pred[1:]-pred[:-1]))
    # ju=torch.mean(torch.abs(diff))
    # l=torch.where(diff>1,diff,torch.tensor(0.).cuda())
    # ju1=torch.mean(l**2)
    # diff=diff[1:]-diff[:-1]
    diff1=torch.sum(torch.abs(true[1:]-true[:-1]))

    # diff1=diff1[1:]-diff1[:-1]
    diff0=diff-diff1
    diff0=diff0**2
    # out=torch.mean(diff0**2)+ju1
    return diff0*0.001#ju1##out
def loss_compu(y_pred,y_true):
    # y_pred=rearrange(y_pred,'b c h w ->(b c) h w')
    loss=0.0
    out=0.0
    for i in range(y_pred.shape[0]):
        for j in range(y_pred.shape[1]):
            loss+=F.smooth_l1_loss(y_pred[i,j],y_true[i,j])
            out+=ze(y_pred[i,j],y_true[i,j])/y_pred.shape[1]
            # print('out',out)
            # loss+=c6(y_pred[i,j],y_true[i,j])
    # print(out)
    return loss/y_pred.shape[0]#+out/(y_pred.shape[0])
    # return loss/y_pred.shape[0]+3*out/(y_pred.shape[0])
