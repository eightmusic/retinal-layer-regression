import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

def swish(x):
    return x * torch.sigmoid(x)

# class Block(nn.Module):
#     def __init__(self,cin):
#         super(Block, self).__init__()
#         self.fc1=nn.Linear(cin,cin)
#         self.ln1=nn.LayerNorm(cin)
#         self.dwconv = nn.Conv1d(cin, cin, 3, 1, 1, bias=True)#groups=cin
#         self.fc2 = nn.Linear(cin,cin)
#         self.ln2 = nn.LayerNorm(cin)
#     def forward(self,x):
#         x0=x
#         x=self.fc1(x)
#         x=swish(self.ln1(x))
#         # x=x.permute(0,2,1)
#         # x = self.dwconv(x)
#         # x = x.permute(0, 2, 1)
#         x=self.fc2(x)
#         x=self.ln2(x)
#         x=swish(x0+x)
#         x=swish(x)
#         return x
class Block(nn.Module):
    def __init__(self,cin):
        super(Block, self).__init__()
        self.fc1=nn.Linear(cin,cin)
        self.ln1=nn.LayerNorm(cin)
        self.pool=nn.MaxPool1d(2,2)
        # self.dwconv = nn.Conv1d(cin, cin, 3, 1, 1, bias=True)#groups=cin
        self.fc2 = nn.Linear(cin//2,cin)
        self.ln2 = nn.LayerNorm(cin)
    def forward(self,x):
        x0=x
        x=self.fc1(x)
        x=swish(self.ln1(x))
        x=self.pool(x)
        # x=x.permute(0,2,1)
        # x = self.dwconv(x)
        # x = x.permute(0, 2, 1)
        x=self.fc2(x)
        x=self.ln2(x)
        x=swish(x0+x)
        x=swish(x)
        return x

class Resmlp(nn.Module):
    def __init__(self,cin,cc,cout,n_class):  #(128, 64, 64, 9)
        super(Resmlp, self).__init__()
        self.a1=nn.Linear(cin,cc) #280
        self.a2 = nn.Linear(cc, cc)
        self.a3 = nn.Linear(cc, cc)
        self.a4 = nn.Linear(cc, cin)
        # self.c1=nn.LayerNorm(280)
        # self.c2 = nn.LayerNorm(280)
        # self.c3 = nn.LayerNorm(280)
        # self.c4 = nn.LayerNorm(cin)

        self.fc=nn.Linear(cin,cin)
        self.ln = nn.LayerNorm(cin)
        self.b1=Block(cin)
        self.b2=Block(cin)
        # self.b2_1 = Block(cout)
        self.mid=nn.Linear(cin,cout)
        self.ln_mid = nn.LayerNorm(cout)
        self.b3=Block(cout)
        self.b4 = Block(cout)
        # self.b4_1 = Block(90)
        self.foot=nn.Linear(cout,90)
        self.ln_foot = nn.LayerNorm(90)
        self.end = nn.Linear(90,n_class)

        self.foot1=nn.MaxPool1d(2,2)
        self.end1 = nn.Linear(32, n_class)
        self.rowindex=torch.range(0,cin-1).cuda()
        self.learn_index=nn.Parameter(torch.tensor(0.01))
        self.ac=nn.ReLU()

        # self.conv=nn.Conv2d(1,1,kernel_size=(3,3),padding=(1,1),padding_mode='replicate')
        # # self.bn=nn.BatchNorm2d(1)
        # self.mlp=nn.Linear(num_patches,8)
    def forward(self,x):
        _,_,_,w0=x.shape
        x = rearrange(x, 'b c h w ->(b w) c h')

        # x=swish(self.c1(self.a1(x)))
        # x = swish(self.c2(self.a2(x)))
        # x = swish(self.c3(self.a3(x)))
        # x = swish(self.c4(self.a4(x)))

        # x=x+self.rowindex*self.learn_index
        # print(x.shape)

        x=swish(self.ln(self.fc(x)))
        x=self.b1(x)
        x = self.b1(x)
        x = self.b2(x)
        # x = self.b2_1(x)
        # print(x.shape)
        x=swish(self.ln_mid(self.mid(x)))
        x = self.b3(x)
        x = self.b4(x)
        # x = self.b4_1(x)
        # x=swish(self.ln_foot(self.foot(x)))
        # x=self.end(x)

        x = self.foot1(x)

        x = self.end1(x)
        # x=self.ac(x)
        # print(x.shape)
        x = rearrange(x, '(b w) c n -> b c n w', w=w0) ###################################64 50
        # print(x.shape)
        # x=self.conv(x)
        # print(x.shape)
        return x