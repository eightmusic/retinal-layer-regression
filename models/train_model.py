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

# def is_na(lay):
#     for i in lay:
#         if (i is None) :
#             print ('error')
#             break
    # print(lay.shape)
    # for a in range(lay.shape[0]):
    #             if ((lay[a] is None)) :#and (lay[a,b,c]!=0)
    #                 print('error')
    #                 break

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
def criteon(logit,lab,epoch,n_classes=10):
    c1=nn.CrossEntropyLoss(weight=torch.tensor((0.25,1,1,1,1,1,1,1,0.25,1)).cuda())
    c2=DiceLoss(n_classes)
    # c3=FocalLoss()
    # c4=nn.MSELoss()
    # c5=F.smooth_l1_loss(logit,lab)
    logit = rearrange(logit, 'b c h w ->(b c) h w')
    c6 = nn.KLDivLoss(size_average=True, reduce=True)  #KL散度
    # if epoch<3:
    # loss=loss_compu(logit,lab,c4)
    # else:
    #     loss = loss_compu1(lab,logit,c6)
    

    #return 0.5*c1(logit,lab.long())+0.5*c2(logit,lab)#+c3(logit,lab.long())
#     return 0.5*c2(logit,lab)+0.5*c3(logit,lab.long())
#     return 0.5*c2(logit,lab)+0.5*c1(logit,lab.long())
#     return loss#c5
#     return c4(logit.float(),lab)
#     return c6(logit,lab)
#     loss=loss_compu(logit.float(), lab, c4)
#     return #loss_compu(logit.float(),lab,c4)
    loss=loss_compu(logit,lab)
    # print(type(loss))
    return loss
    # return c6(logit,lab)

def loss_compu1(y_pred,y_true):
#     y_pred=rearrange(y_pred,'b c h w ->(b c) h w')
#     loss=0.0
#     for i in range(y_pred.shape[0]):
#         for j in range(y_pred.shape[1]):
#             loss+=F.smooth_l1_loss(y_pred[i,j],y_true[i,j])   #这么计算和直接计算的区别
# #             loss+=c6(y_pred[i,j],y_true[i,j])
#     return loss/y_pred.shape[0]

    y_pred=rearrange(y_pred,'b c h w ->(b c) h w')
    loss=0.0
    for i in range(y_pred.shape[0]):
        for j in range(y_pred.shape[1]):
            # print(y_pred[i,:,j].shape,y_true[i,:,j].shape)
            loss+=F.smooth_l1_loss(y_pred[i,j,:],y_true[i,j,:])
#             loss+=c6(y_pred[i,j],y_true[i,j])
    return loss/y_pred.shape[0]

# def ze(pred):
#     # pred=torch.round(pred)
#     diff=pred[1:]-pred[:-1]
#     diff=diff[1:]-diff[:-1]
#     out=torch.mean(diff**2)
#     return out                   #梯度相似，梯度变化相似
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


# def accuracy(y_pred,y_true):
#     print(y_pred.shape,y_true.shape)
#     y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred),dim=1).data
#     print(y_pred.shape,y_pred_cls.shape,y_true.shape)
#     return accuracy_score(y_true,y_pred_cls)
# def accuracy(y_pred,y_true):
# #     y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred),dim=1).data
#     return accuracy_score(y_true,y_pred)
def acc(logits,labels):
    # print('lab',labels.shape)
#     print(logits.shape,labels.shape)
#     if logits.shape[0] !=labels.shape[0] :
#         print(logits.shape,labels.shape)
#         print('shape error')
#     print(torch.sum(logits.data==labels.data),labels.shape[0]*labels.shape[1]*labels.shape[2])
    a=torch.sum(logits.data==labels.data)/(labels.shape[1]*labels.shape[2])
    return a

def rmse(y_pred,y_true):
    y_pred,y_true=y_pred.cpu().detach().numpy(),y_true.cpu().detach().numpy()
    y_pred=rearrange(y_pred,'b c h w ->(b c) h w')
    rmse=[0.0 for i in range(y_pred.shape[1])]
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
    return [x/y_pred.shape[0] for x in rmse]
#     return np.sqrt(mean_squared_error(y_true, y_pred))
    
def mae(y_pred,y_true):
    y_pred,y_true=y_pred.cpu().detach().numpy(),y_true.cpu().detach().numpy()
    y_pred=rearrange(y_pred,'b c h w ->(b c) h w')
#     print(y_pred.shape,y_true.shape)
    mae=[0.0 for i in range(y_pred.shape[1])]
    for i in range(y_pred.shape[0]):
        for j in range(y_pred.shape[1]):
            # is_na(y_true[i,j])
            mae[j]+=mean_absolute_error(y_pred[i,j],y_true[i,j])

    return [x/y_pred.shape[0] for x in mae]
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

def judge(out,batch_size=8):
    line=[]
    for b in range(batch_size):  #(2120, 1, 235, 30)
        tu=[]
        for w in range(30):
            i=0
            idx=[]
            for h in range(234):
                if out[b,h,w]==i and out[b,h+1,w]==i+1 :
                    idx.append(h+1)
                    i+=1
            tu.append(idx)
        line.append(tu)
#     print(np.array(idx).shape,np.array(tu).shape,np.array(line).shape)
    return np.array(line)

def rmseandmae(line,lab,batch_size=8):
    rmse_value=0
    mae_value=0
    for b in range(batch_size):  #(2120, 1, 235, 30)
        for h in range(8):
            rmse_value+=rmse(line[b,h],lab[b,h])
            mae_value+=mae(line[b,h],lab[b,h])
    return rmse_value/(b*h),mae_value/(b*h)

# model = net
# model.optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)
# model.loss_func = criteon(logit,lab,weight=False,n_classes=10)
# model.metric_accuracy = accuracy
# model.metric.rmse=rmse
# model.metric.mae=mae
# model.dice=per_class_dice
#model.metric_name = "accuracy"
# warm_up_epochs = 5
# warm_up_with_cosine_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs \
#     else 0.5 * ( math.cos((epoch - warm_up_epochs) /(epochs - warm_up_epochs) * math.pi) + 1)
# exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR( optimizer, lr_lambda=warm_up_with_cosine_lr)


def train_step(model,features,labels,epoch):
    
    # 训练模式，dropout层发生作用
    model.train()
    
    # 梯度清零
    model.optimizer.zero_grad()
    
    # 正向传播求损失
    predictions = model(features)
#     print(predictions.shape)
    loss = model.loss_func(predictions,labels,epoch)
    # logits=torch.argmax(predictions,dim=1)
#     print(logits.shape,labels.shape)
#     accuracy = model.metric_accuracy(logits,labels)
#     line1=judge(logits)
#     line2=judge(labels)
# #     print(logits.shape,labels.shape,line1.shape,line2.shape)
#     rmse,mae=rmseandmae(line1,line2)
    rmse=model.metric_rmse(predictions,labels)
    mae=model.metric_mae(predictions,labels)
#     dice=model.dice(logit,labels,num_class)

    # 反向传播求梯度
    loss.backward()
#     with torch.autograd.set_detect_anomaly(True):
#         loss.backward()
    model.optimizer.step()

    # return loss.item(),accuracy.item()
    return loss.item(),mae,rmse#,loss.item(),accuracy.item(),rmse.item(),mae.item()#,dice().item()

@torch.no_grad()
def valid_step(model,features,labels,epoch):
    
    # 预测模式，dropout层不发生作用
    model.eval()
    
    predictions= model(features)
    # predictions=
    loss = model.loss_func(predictions,labels,epoch)
    logits=torch.argmax(predictions,dim=1)
    # accuracy = model.metric_accuracy(logits,labels)
#     line1=judge(logits)
#     line2=judge(labels)
#     rmse,mae=rmseandmae(line1,line2)
    rmse=model.metric_rmse(predictions,labels)
    mae=model.metric_mae(predictions,labels)
    # print(predictions.shape,out.shape)
    # mae_p=model.metric_mae(out,labels)
    return loss.item(),mae,rmse#,mae_p#,loss.item(),accuracy.item(),rmse.item(),mae.item()#,dice().item()


# 测试train_step效果
# features,labels = next(iter(dl_train))
# train_step(model,features,labels)


def train_model_step(model,epochs,dl_train,dl_valid,scheduler,PATH=False,n_class=8,log_step_freq=100):

#    metric_name = model.metric_name
#     dfhistory = pd.DataFrame(columns = ["epoch","loss","accuracy","rmse","mae","dice","val_loss","val_accuracy","val_rmse","val_mae","val_dice"]) 
#     dfhistory = pd.DataFrame(columns = ["epoch","loss","accuracy","val_loss","val_accuracy"])
    dfhistory = pd.DataFrame(columns = ["epoch","loss","mae","rmse","val_loss","val_mae","val_rmse"])
    print("Start Training...")
 #   nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
 #   print("=========="*8 + "%s"%nowtime)
    since = time.time()
    best_mae=60.0
    for epoch in range(1,epochs+1):  

        # 1，训练循环-------------------------------------------------
        loss_sum = 0.0
        metric_sum = 0.0
        metric_accuracy=0.0
        metric_mae=np.array([0.0 for i in range(n_class)])
        metric_rmse=np.array([0.0 for i in range(n_class)])

        step = 1
        scheduler.step()
        for step, (features,labels) in enumerate(dl_train, 1):
            features,labels=features.cuda(),labels.cuda()
            # loss,accuracy= train_step(model,features,labels,epoch)#,rmse,mae,dice
            loss,mae,rmse= train_step(model,features,labels,epoch)

            # 打印batch级别日志
            loss_sum += loss
            # metric_accuracy += accuracy
#             metric_rmse += rmse
            metric_mae += np.array(mae)
            metric_rmse+=np.array(rmse)
#             metric_dice += dice
 #           if step%log_step_freq == 0:   
#                print(("[step = %d] loss: %.3f, "+metric_name+": %.3f") %
#                      (step, loss_sum/step, metric_sum/step))

        # 2，验证循环-------------------------------------------------
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_metric_accuracy=0.0
        val_metric_mae=np.array([0.0 for i in range(n_class)])
        # val_metric_mae_p=np.array([0.0 for i in range(n_class)])
        val_metric_rmse=np.array([0.0 for i in range(n_class)])##########################################9 8

        #val_step = 1

        for val_step, (features,labels) in enumerate(dl_valid, 1):
            features,labels=features.cuda(),labels.cuda()
            # val_loss,val_accuracy= valid_step(model,features,labels,epoch)
            val_loss,val_mae,val_rmse= valid_step(model,features,labels,epoch)

            val_loss_sum += val_loss
            # val_metric_accuracy += val_accuracy
#             val_metric_rmse += val_rmse
            val_metric_mae += np.array(val_mae)
            # val_metric_mae_p += np.array(val_mae_p)
            val_metric_rmse += np.array(val_rmse)
            
#             val_metric_dice += val_dice

        # 3，记录日志-------------------------------------------------
#         info = (epoch, loss_sum/step, metric_accuracy/step, metric_rmse/step, metric_mae/step, metric_dice/step, 
#                 val_loss_sum/val_step, val_metric_accuracy/val_step, val_metric_rmse/step, val_metric_mae/step, val_metric_dice/step)
        metric_mae=[round(x/step,3) for x in metric_mae]
        metric_mae=np.concatenate((metric_mae,[np.mean(metric_mae)]),axis=0)

        metric_rmse=[round(x/step,3) for x in metric_rmse]
        metric_rmse=np.concatenate((metric_rmse,[np.mean(metric_rmse)]),axis=0)

        val_metric_mae=[round(x/val_step,3) for x in val_metric_mae]
        val_metric_mae=np.concatenate((val_metric_mae,[np.mean(val_metric_mae)]),axis=0)

        # val_metric_mae_p=[round(x/val_step,3) for x in val_metric_mae_p]
        # val_metric_mae_p=np.concatenate((val_metric_mae_p,[np.mean(val_metric_mae_p)]),axis=0)

        val_metric_rmse=[round(x/val_step,3) for x in val_metric_rmse]
        val_metric_rmse=np.concatenate((val_metric_rmse,[np.mean(val_metric_rmse)]),axis=0)

        info = (epoch, loss_sum/step,  metric_mae[-1],metric_rmse[-1],
                val_loss_sum/val_step,  val_metric_mae[-1],val_metric_rmse[-1])
#         info = (epoch, loss_sum/step,  accuracy/n_train,
#                 val_loss_sum/val_step,  val_accuracy/n_valid)
        dfhistory.loc[epoch-1] = info

        # 打印epoch级别日志
#         print(("\nEPOCH = %d, loss = %.3f,"+ "accuracy =%0.3f"+"rmse=%0.3f","mae=%0.3f","dice=%0.3f",
#         "val_loss =%0.3f","val_accuracy =%0.3f","val_rmse=%0.3f","val_mae=%0.3f","val_dice=%0.3f") %info)
        print(("\nEPOCH = %d,loss = %.3f,mae =%s,\nrmse =%s,\nval_loss =%0.3f,val_mae =%s,\nval_rmse =%s") %info)
#         print(("\nEPOCH = %d,loss = %.3f,acc =%s,\nval_loss =%0.3f,val_acc =%s") % info)
 #       nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
 #       print("\n"+"=========="*8 + "%s"%nowtime)
        time_elapsed = time.time() - since;   
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))   

        if(PATH):
            if val_metric_mae[-1]<best_mae:
                    best_mae = val_metric_mae[-1]
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, PATH)
                    if abs(best_mae-0.99)<0.1:
                        best_model_wts = copy.deepcopy(model.state_dict())
                        hcms_path = 'B:/eee/demo/pycharm/oct/weights/transformer/end/HCMS.pkl'
                        torch.save(best_model_wts, hcms_path)
                        print('ok-----------------------------')
                    print('Success save')



    print('Finished Training...')
    return dfhistory
    
def train_model(model,epochs,dl_train,dl_valid,dl_test=None,PATH=False,n_class=8,l=0.001,log_step_freq=100):
#     model.optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)
#     model.optimizer = torch.optim.Adam(model.parameters(),lr = 0.001,weight_decay=0.0001)
    model.optimizer = torch.optim.AdamW(model.parameters(),lr = l,weight_decay=0.0001)
    model.loss_func = criteon
    # model.metric_accuracy = acc
    model.metric_rmse=rmse
    model.metric_mae=mae
#     model.dice=per_class_dice
    warm_up_epochs = 5
    warm_up_with_cosine_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs \
        else 0.5 * ( math.cos((epoch - warm_up_epochs) /(epochs - warm_up_epochs) * math.pi) + 1)
    exp_scheduler = torch.optim.lr_scheduler.LambdaLR( model.optimizer, lr_lambda=warm_up_with_cosine_lr)

    
    train_model_step(model,epochs,dl_train,dl_valid,exp_scheduler,PATH=PATH,n_class=n_class,log_step_freq=100)
    if dl_test:
        evaluation(model,dl_test)
def sm(line):
    tmp_smooth = savgol_filter(line, 53, 3)
    return tmp_smooth

# def evaluation(model,test1,test2):
#     since = time.time()
#     num = len(test1)
#     print(num)
#     # test1,test2=test1.cuda(),test2().cuda()
#     test_rmse = rmse(test1, test2)
#     test_mae = mae(test1, test2)
#     v_mae = [round(x / num, 3) for x in test_mae]
#     v_mae = np.concatenate((v_mae, [np.mean(v_mae)]), axis=0)
#     v_rmse = [round(x / num, 3) for x in test_rmse]
#     v_rmse = np.concatenate((v_rmse, [np.mean(v_rmse)]), axis=0)
#
#     print('mae', v_mae)
#     print('rmse', v_rmse)
#     time_elapsed = time.time() - since;
#     print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#     # since = time.time()
#     # model.metric_rmse = rmse
#     # model.metric_mae = mae
#     # test_metric_mae = np.array([0.0 for i in range(8)])
#     # test_metric_rmse = np.array([0.0 for i in range(8)])
#     #
#     #
#     # for test_step, (features, labels) in enumerate(dl_test, 1):
#     #     features, labels = features.cuda(), labels.cuda()
#     #     # val_loss,val_accuracy= valid_step(model,features,labels,epoch)
#     #     # test_loss, test_mae, test_rmse,test_mae_p = valid_step(model, features, labels,1)
#     #     test_loss, test_mae, test_rmse= valid_step(model, features, labels,1)
#     #     test_metric_mae += np.array(test_mae)
#     #     # test_metric_mae_p += np.array(test_mae_p)
#     #     test_metric_rmse += np.array(test_rmse)
#     #
#     # val_metric_mae = [round(x / test_step, 3) for x in test_metric_mae]
#     # val_metric_mae = np.concatenate((val_metric_mae, [np.mean(val_metric_mae)]), axis=0)
#     # # val_metric_mae_p = [round(x / test_step, 3) for x in test_metric_mae_p]
#     # # val_metric_mae_p = np.concatenate((val_metric_mae, [np.mean(val_metric_mae_p)]), axis=0)
#     #
#     # val_metric_rmse = [round(x / test_step, 3) for x in test_metric_rmse]
#     # val_metric_rmse = np.concatenate((val_metric_rmse, [np.mean(val_metric_rmse)]), axis=0)
#     # print("val_metric_mae:",val_metric_mae)
#     # print("val_metric_rmse:",val_metric_rmse)
#     # time_elapsed = time.time() - since;
#     # print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

def mediasmooth(line,size):
    line=rearrange(line,'b c h w ->(b c) h w')
    out=np.zeros_like(line)
    w=line.shape[-1]
    line=np.pad(line,((0,0),(0,0),((size-1)//2,(size-1)//2)),'edge')
    for i in range(w):
        out[:,:,i]=np.median(line[:,:,i:i+size],axis=2)
        # out[:,:,i]=np.mean(line[:,:,i:i+size],axis=2)
    out = rearrange(out, '(b c) h w->b c h w',c=1)
    return out


# def evaluation(model, dl_test):
#     since = time.time()
#     model.loss_func = criteon
#     model.metric_rmse = rmse
#     model.metric_mae = mae
#     test_metric_mae = np.array([0.0 for i in range(8)])
#     # test_metric_mae_p = np.array([0.0 for i in range(8)])
#     test_metric_rmse = np.array([0.0 for i in range(8)])
#
#     for test_step, (features, labels) in enumerate(dl_test, 1):
#         model.eval()
#         features, labels = features.cuda(), labels.cuda()
#         predictions = model(features)
#         out=predictions.cpu().detach().numpy()
#         # with open('abc.txt','ab') as f:
#         #     np.savetxt(f,out,delimiter=" ")
#
#         predictions = savgol_filter(out, 47, 3) #49-3 75-4
#         # predictions=mediasmooth(out,15)
#
#
#
#         predictions=torch.tensor(predictions).cuda()
#         test_rmse = model.metric_rmse(predictions, labels)
#         test_mae = model.metric_mae(predictions, labels)
#
#         test_metric_mae += np.array(test_mae)
#         # test_metric_mae_p += np.array(test_mae_p)
#         test_metric_rmse += np.array(test_rmse)
#
#     val_metric_mae = [round(x / test_step, 3) for x in test_metric_mae]
#     val_metric_mae = np.concatenate((val_metric_mae, [np.mean(val_metric_mae)]), axis=0)
#     # val_metric_mae_p = [round(x / test_step, 3) for x in test_metric_mae_p]
#     # val_metric_mae_p = np.concatenate((val_metric_mae, [np.mean(val_metric_mae_p)]), axis=0)
#
#     val_metric_rmse = [round(x / test_step, 3) for x in test_metric_rmse]
#     val_metric_rmse = np.concatenate((val_metric_rmse, [np.mean(val_metric_rmse)]), axis=0)
#     print("val_metric_mae:", val_metric_mae)
#     # print("val_metric_mae_p:",val_metric_mae_p)
#     print("val_metric_rmse:", val_metric_rmse)
#     time_elapsed = time.time() - since;
#     print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

def evaluation(model, dl_test):
    since = time.time()
    model.loss_func = criteon
    model.metric_rmse = rmse
    model.metric_mae = mae
    test_metric_mae = np.array([0.0 for i in range(9)])
    # test_metric_mae_p = np.array([0.0 for i in range(8)])
    test_metric_rmse = np.array([0.0 for i in range(9)])

    # pre_sd=[]
    # lab_sd=[]
    # ds=[]

    for test_step, (features, labels) in enumerate(dl_test, 1):
        model.eval()
        features, labels = features.cuda(), labels.cuda()
        predictions = model(features)
        out=predictions.cpu().detach().numpy()
        # with open('abc.txt','ab') as f:
        #     np.savetxt(f,out,delimiter=" ")

        # predictions = savgol_filter(out, 47, 3) #49-3 75-4*********************************************************************************
        predictions=out
        # predictions=mediasmooth(out,15)
        # print(test_step)
        # if test_step==1:
        #     pre_sd=rearrange(predictions, 'b c h w ->(b c) h w')
        #     lab_sd=labels.cpu().detach().numpy()
        # else:
        #     # pre_sd.append(predictions.tolist())
        #     # lab_sd.append(labels.cpu().detach())
        #     pre_sd,lab_sd=hebin(predictions,pre_sd,labels.cpu().detach().numpy(),lab_sd)
        #     # print(len(pre_sd))
        #     # print(predictions)
        pre = rearrange(predictions, 'b c h w ->(b c) h w')
        lab = labels.cpu().detach().numpy()
        if test_step == 1:
            ds=np.abs(pre-lab)
            # print(ds.shape,"11111")
            # rds-mp.abs()
        else:
            ds=np.concatenate((ds,np.abs(pre-lab)),axis=0)
        # pre=rearrange(predictions, 'b c h w ->(b c) h w')
        # lab=labels.cpu().detach().numpy()
        # ds.append(np.abs(pre-lab).tolist())
        # np.concatenate()
        # print(pre.shape,lab.shape,np.array(ds).shape)



        predictions=torch.tensor(predictions).cuda()
        test_rmse = model.metric_rmse(predictions, labels)
        test_mae = model.metric_mae(predictions, labels)

        diff=predictions-labels

        test_metric_mae += np.array(test_mae)
        # test_metric_mae_p += np.array(test_mae_p)
        test_metric_rmse += np.array(test_rmse)

    # print_sd=sd(pre_sd,lab_sd)
    # print(np.array(print_sd).shape)
    # print(np.array(print_sd))
    print(ds.shape)
    m=np.mean(ds,axis=(0,2))
    # ds=np.array(ds)
    # print(ds.shape)
    print_sd=np.std(ds,axis=(0,2))#------------------
    print_rsd = np.std(np.square(ds), axis=(0, 2))
    a1=np.std(ds, axis=2)
    a2 = np.std(a1, axis=0)
    a3 = np.mean(a1, axis=0)
    b1 = np.mean(ds, axis=2)
    b2 = np.mean(b1, axis=0)
    c2=np.std(b1, axis=0)#-----------------
    # print(a1.shape,a2.shape,"r标准差",a2,"标准差",c2)
    # print("MAD",b2,"RMSE",a3)
    # print('标准差',print_sd,'均值',m)
    # print('r标准差',print_rsd,)
    # print
    print('标准差',print_sd)
    print("r标准差",a2)

    val_metric_mae = [round(x / test_step, 3) for x in test_metric_mae]
    val_metric_mae = np.concatenate((val_metric_mae, [np.mean(val_metric_mae)]), axis=0)
    # val_metric_mae_p = [round(x / test_step, 3) for x in test_metric_mae_p]
    # val_metric_mae_p = np.concatenate((val_metric_mae, [np.mean(val_metric_mae_p)]), axis=0)

    val_metric_rmse = [round(x / test_step, 3) for x in test_metric_rmse]
    val_metric_rmse = np.concatenate((val_metric_rmse, [np.mean(val_metric_rmse)]), axis=0)
    print("val_metric_mae:", val_metric_mae)
    # print("val_metric_mae_p:",val_metric_mae_p)
    print("val_metric_rmse:", val_metric_rmse)
    time_elapsed = time.time() - since;
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # if abs(val_metric_mae - 0.99) < 0.2:
    #     best_model_wts = copy.deepcopy(model.state_dict())
    #     hcms_path = 'B:/eee/demo/pycharm/oct/weights/transformer/end/HCMS.pkl'
    #     torch.save(best_model_wts, hcms_path)
    #     print('ok-----------------------------')

# def sd(pre,tar):





# def sd(pre,tar):
#     # pre,tar=pre.cpu(),tar.cpu()
#     # pre=np.array(pre).reshape(-1)
#     # tar=np.array(tar).reshape(-1)
#     # y_pred, y_true = y_pred.cpu().detach().numpy(), y_true.cpu().detach().numpy()
#     print(pre.shape,tar.shape)
#     # print(pre)
#     # print(pre.device)
#     # print(pre[0].device)
#     for i in range(8):
#         pre_xin=pre[:,i,:].reshape(-1)
#         tar_xin=tar[:,i,:].reshape(-1)
#         # m=np.sum(pre_xin-tar_xin)/len(pre_xin)
#         ds=np.abs(pre_xin-tar_xin)
#         # ds=pre_xin-tar_xin
#         m=np.mean(ds)
#         # sd=np.sqrt(np.sum((ds-m)**2)/len(pre_xin))
#         sd=np.std(ds)
#         print(len(pre_xin))
#         # print('第’,i,'层标准差‘,sd)
#
#         print('第',i,'标准差',sd,m)
#     return sd

# def sd(y_pred,y_true):
#     print(y_pred.shape)
#     mae=[0.0 for i in range(y_pred.shape[1])]
#     for i in range(y_pred.shape[0]):
#         for j in range(y_pred.shape[1]):
#             # is_na(y_true[i,j])
#             mae[j]+=mean_absolute_error(y_pred[i,j],y_true[i,j])
#
#         return [x/y_pred.shape[0] for x in mae]
# def sd(y_pred,y_true):
#     print(y_pred.shape)
#     mae=[0.0 for i in range(y_pred.shape[1])]
#     for j in range(y_pred.shape[1]):
#         # is_na(y_true[i,j])
#         mae[j]=mean_absolute_error(y_pred[:,j,:],y_true[:,j,:])
#         # print(mae[j],j)
#
#     return mae


# def mae(y_pred,y_true):
#     y_pred,y_true=y_pred.cpu().detach().numpy(),y_true.cpu().detach().numpy()
#     y_pred=rearrange(y_pred,'b c h w ->(b c) h w')
# #     print(y_pred.shape,y_true.shape)
#     mae=[0.0 for i in range(y_pred.shape[1])]
#     for i in range(y_pred.shape[0]):
#         for j in range(y_pred.shape[1]):
#             # is_na(y_true[i,j])
#             mae[j]+=mean_absolute_error(y_pred[i,j],y_true[i,j])
#
#     return [x/y_pred.shape[0] for x in mae]


# [19.291708  21.409605  17.059944  14.234131  14.002656   1.9887661
#   1.6136651  1.8446077]
def hebin(arr1,arr2,arr3,arr4):
    arr1 = rearrange(arr1, 'b c h w ->(b c) h w')
    arr2=np.concatenate((arr2,arr1),axis=0)
    arr4=np.concatenate((arr3,arr4),axis=0)
    return arr2,arr4
