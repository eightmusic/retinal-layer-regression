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
from utils.loss import DiceLoss,FocalLoss,loss_compu,rmse,mae


def train_step(model ,features ,labels ,epoch):

    # 训练模式，dropout层发生作用
    model.train()

    # 梯度清零
    model.optimizer.zero_grad()

    # 正向传播求损失
    predictions = model(features)
    #     print(predictions.shape)
    loss = model.loss_func(predictions ,labels ,epoch)
    # logits=torch.argmax(predictions,dim=1)
    #     print(logits.shape,labels.shape)
    #     accuracy = model.metric_accuracy(logits,labels)
    #     line1=judge(logits)
    #     line2=judge(labels)
    # #     print(logits.shape,labels.shape,line1.shape,line2.shape)
    #     rmse,mae=rmseandmae(line1,line2)
    rmse =model.metric_rmse(predictions ,labels)
    mae =model.metric_mae(predictions ,labels)
    #     dice=model.dice(logit,labels,num_class)

    # 反向传播求梯度
    loss.backward()
    #     with torch.autograd.set_detect_anomaly(True):
    #         loss.backward()
    model.optimizer.step()

    # return loss.item(),accuracy.item()
    return loss.item() ,mae ,rmse  # ,loss.item(),accuracy.item(),rmse.item(),mae.item()#,dice().item()

@torch.no_grad()
def valid_step(model ,features ,labels ,epoch):

    # 预测模式，dropout层不发生作用
    model.eval()

    predictions= model(features)
    # predictions=
    loss = model.loss_func(predictions ,labels ,epoch)
    logits =torch.argmax(predictions ,dim=1)
    # accuracy = model.metric_accuracy(logits,labels)
    #     line1=judge(logits)
    #     line2=judge(labels)
    #     rmse,mae=rmseandmae(line1,line2)
    rmse =model.metric_rmse(predictions ,labels)
    mae =model.metric_mae(predictions ,labels)
    # print(predictions.shape,out.shape)
    # mae_p=model.metric_mae(out,labels)
    return loss.item() ,mae ,rmse  # ,mae_p#,loss.item(),accuracy.item(),rmse.item(),mae.item()#,dice().item()


# 测试train_step效果
# features,labels = next(iter(dl_train))
# train_step(model,features,labels)


def train_model_step(model ,epochs ,dl_train ,dl_valid ,scheduler ,PATH=False ,n_class=8 ,log_step_freq=100):

    #    metric_name = model.metric_name
    #     dfhistory = pd.DataFrame(columns = ["epoch","loss","accuracy","rmse","mae","dice","val_loss","val_accuracy","val_rmse","val_mae","val_dice"])
    #     dfhistory = pd.DataFrame(columns = ["epoch","loss","accuracy","val_loss","val_accuracy"])
    dfhistory = pd.DataFrame(columns = ["epoch" ,"loss" ,"mae" ,"rmse" ,"val_loss" ,"val_mae" ,"val_rmse"])
    print("Start Training...")
    #   nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    #   print("=========="*8 + "%s"%nowtime)
    since = time.time()
    best_mae =60.0
    for epoch in range(1 ,epochs +1):

        # 1，训练循环-------------------------------------------------
        loss_sum = 0.0
        metric_sum = 0.0
        metric_accuracy =0.0
        metric_mae =np.array([0.0 for i in range(n_class)])
        metric_rmse =np.array([0.0 for i in range(n_class)])

        step = 1
        scheduler.step()
        for step, (features ,labels) in enumerate(dl_train, 1):
            features ,labels =features.cuda() ,labels.cuda()
            # loss,accuracy= train_step(model,features,labels,epoch)#,rmse,mae,dice
            loss ,mae ,rmse= train_step(model ,features ,labels ,epoch)

            # 打印batch级别日志
            loss_sum += loss
            # metric_accuracy += accuracy
            #             metric_rmse += rmse
            metric_mae += np.array(mae)
            metric_rmse +=np.array(rmse)
        #             metric_dice += dice
        #           if step%log_step_freq == 0:
        #                print(("[step = %d] loss: %.3f, "+metric_name+": %.3f") %
        #                      (step, loss_sum/step, metric_sum/step))

        # 2，验证循环-------------------------------------------------
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_metric_accuracy =0.0
        val_metric_mae =np.array([0.0 for i in range(n_class)])
        # val_metric_mae_p=np.array([0.0 for i in range(n_class)])
        val_metric_rmse =np.array([0.0 for i in range(n_class)]  )##########################################9 8

        # val_step = 1

        for val_step, (features ,labels) in enumerate(dl_valid, 1):
            features ,labels =features.cuda() ,labels.cuda()
            # val_loss,val_accuracy= valid_step(model,features,labels,epoch)
            val_loss ,val_mae ,val_rmse= valid_step(model ,features ,labels ,epoch)

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
        metric_mae =[round( x /step ,3) for x in metric_mae]
        metric_mae =np.concatenate((metric_mae ,[np.mean(metric_mae)]) ,axis=0)

        metric_rmse =[round( x /step ,3) for x in metric_rmse]
        metric_rmse =np.concatenate((metric_rmse ,[np.mean(metric_rmse)]) ,axis=0)

        val_metric_mae =[round( x /val_step ,3) for x in val_metric_mae]
        val_metric_mae =np.concatenate((val_metric_mae ,[np.mean(val_metric_mae)]) ,axis=0)

        # val_metric_mae_p=[round(x/val_step,3) for x in val_metric_mae_p]
        # val_metric_mae_p=np.concatenate((val_metric_mae_p,[np.mean(val_metric_mae_p)]),axis=0)

        val_metric_rmse =[round( x /val_step ,3) for x in val_metric_rmse]
        val_metric_rmse =np.concatenate((val_metric_rmse ,[np.mean(val_metric_rmse)]) ,axis=0)

        info = (epoch, loss_sum /step,  metric_mae[-1] ,metric_rmse[-1],
                val_loss_sum /val_step,  val_metric_mae[-1] ,val_metric_rmse[-1])
        #         info = (epoch, loss_sum/step,  accuracy/n_train,
        #                 val_loss_sum/val_step,  val_accuracy/n_valid)
        dfhistory.loc[epoch -1] = info

        # 打印epoch级别日志
        #         print(("\nEPOCH = %d, loss = %.3f,"+ "accuracy =%0.3f"+"rmse=%0.3f","mae=%0.3f","dice=%0.3f",
        #         "val_loss =%0.3f","val_accuracy =%0.3f","val_rmse=%0.3f","val_mae=%0.3f","val_dice=%0.3f") %info)
        print(("\nEPOCH = %d,loss = %.3f,mae =%s,\nrmse =%s,\nval_loss =%0.3f,val_mae =%s,\nval_rmse =%s") %info)
        #         print(("\nEPOCH = %d,loss = %.3f,acc =%s,\nval_loss =%0.3f,val_acc =%s") % info)
        #       nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        #       print("\n"+"=========="*8 + "%s"%nowtime)
        time_elapsed = time.time() - since;
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        if (PATH):
            if val_metric_mae[-1] < best_mae:
                best_mae = val_metric_mae[-1]
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, PATH)
                if abs(best_mae - 0.99) < 0.1:
                    best_model_wts = copy.deepcopy(model.state_dict())
                    hcms_path = 'B:/eee/demo/pycharm/oct/weights/transformer/end/HCMS.pkl'
                    torch.save(best_model_wts, hcms_path)
                    print('ok-----------------------------')
                print('Success save')

    print('Finished Training...')
    return dfhistory


def criteon(logit, lab, epoch, n_classes=10):
    c1 = nn.CrossEntropyLoss(weight=torch.tensor((0.25, 1, 1, 1, 1, 1, 1, 1, 0.25, 1)).cuda())
    c2 = DiceLoss(n_classes)
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
    loss = loss_compu(logit, lab)
    # print(type(loss))
    return loss

def train_model(model, epochs, dl_train, dl_valid, dl_test=None, PATH=False, n_class=8,
                optimizer=None,log_step_freq=100):
    #     model.optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)
    #     model.optimizer = torch.optim.Adam(model.parameters(),lr = 0.001,weight_decay=0.0001)
    # model.optimizer = torch.optim.AdamW(model.parameters(), lr=l, weight_decay=0.0001)
    model.loss_func = criteon
    # model.metric_accuracy = acc
    model.metric_rmse = rmse
    model.metric_mae = mae
    #     model.dice=per_class_dice
    warm_up_epochs = 5
    warm_up_with_cosine_lr = lambda epoch: (epoch + 1) / warm_up_epochs if epoch < warm_up_epochs \
        else 0.5 * (math.cos((epoch - warm_up_epochs) / (epochs - warm_up_epochs) * math.pi) + 1)
    exp_scheduler = torch.optim.lr_scheduler.LambdaLR(model.optimizer, lr_lambda=warm_up_with_cosine_lr)

    train_model_step(model, epochs, dl_train, dl_valid, exp_scheduler, PATH=PATH, n_class=n_class, log_step_freq=100)
    if dl_test:
        evaluation(model, dl_test)

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
