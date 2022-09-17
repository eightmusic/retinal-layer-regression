import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torchsummary import summary
from thop import profile

def swish(x):
    return x * torch.sigmoid(x)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout=0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.MaxPool1d(2,2),
#             nn.Linear(hidden_dim//2, dim),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x):
#         return self.net(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.l1=nn.Linear(dim, hidden_dim)
        self.ac1=nn.GELU()
        self.d1=nn.Dropout(dropout)
        # self.up=nn.MaxPool1d(2,2)
        self.up=nn.Conv1d(hidden_dim,hidden_dim,3,1,1)
        self.l2=nn.Linear(hidden_dim, dim)
        self.d2=nn.Dropout(dropout)
    def forward(self,x):
        x=self.d1(self.ac1(self.l1(x)))
        x=x.permute(0,2,1)
        x=self.up(x)
        x = x.permute(0, 2, 1)
        x=self.d2(self.l2(x))
        return x


# class Attention(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
#         super().__init__()
#         inner_dim = dim_head * heads
#         self.heads = heads
#         self.scale = dim ** -0.5
#
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x, mask=None):
#         # b, 65, 1024, heads = 8
#         b, n, _, h = *x.shape, self.heads
#
#         # self.to_qkv(x): b, 65, 64*8*3
#         # qkv: b, 65, 64*8
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#
#         # b, 65, 64, 8
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
#
#         # dots:b, 65, 64, 64
#         dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
#         mask_value = -torch.finfo(dots.dtype).max
#
#         if mask is not None:
#             mask = F.pad(mask.flatten(1), (1, 0), value=True)
#             assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
#             mask = mask[:, None, :] * mask[:, :, None]
#             dots.masked_fill_(~mask, mask_value)
#             del mask
#
#         # attn:b, 65, 64, 64
#         attn = dots.softmax(dim=-1)
#
#         # 使用einsum表示矩阵乘法：
#         # out:b, 65, 64, 8
#         out = torch.einsum('bhij,bhjd->bhid', attn, v)
#
#         # out:b, 64, 65*8
#         out = rearrange(out, 'b h n d -> b n (h d)')
#
#         # out:b, 64, 1024
#         out = self.to_out(out)
#         return out
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.,sr_ratio=2):
        super().__init__()
        inner_dim = dim_head * heads
        self.ind=inner_dim
        # inner_dim = dim * heads
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_q = nn.Linear(dim, inner_dim * 1, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.sr=sr_ratio
        if sr_ratio > 1:
            # self.pool=nn.MaxPool1d(self.sr,self.sr)

            self.pool = nn.Conv1d(dim, dim, kernel_size=self.sr, stride=self.sr)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        # b, 65, 1024, heads = 8
        b, n, d, h = *x.shape, self.heads
        q = self.to_q(x).reshape(b,n,h,-1).permute(0, 2, 1, 3)
        if self.sr>1:
            # x=x.permute(0,2,1)
            # x=self.pool(x)
            # x = x.permute(0, 2, 1)

            x_ = x.permute(0, 2, 1)
            x_ = self.pool(x_).permute(0, 2, 1)
            x = self.norm(x_)

        kv = self.to_kv(x).reshape(b, -1, 2, h, self.ind//h).permute(2, 0, 3, 1, 4)
        # kv = self.to_kv(x).reshape(b, n, 2, h, -1 ).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, self.ind)
        x = self.to_out(x)
        return x

class Transformer(nn.Module):  # dim:,depth：num_transformer_layer,dim_head:多头，mlp_dim:
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                    Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
                ]))

            # self.layers = nn.ModuleList([
            #     nn.ModuleList([
            #         Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
            #         Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            #     ]),
            #     nn.ModuleList([
            #         Residual(
            #             PreNorm(2 * dim, Attention(2 * dim, heads=2 * heads, dim_head=2 * dim_head, dropout=dropout))),
            #         Residual(PreNorm(2 * dim, FeedForward(2 * dim, 2 * mlp_dim, dropout=dropout)))
            #     ]),
            #     nn.ModuleList([
            #         Residual(
            #             PreNorm(4 * dim, Attention(4 * dim, heads=4 * heads, dim_head=4 * dim_head, dropout=dropout))),
            #         Residual(PreNorm(4 * dim, FeedForward(4 * dim, 4 * mlp_dim, dropout=dropout)))
            #     ]),
            #     #             nn.ModuleList([
            #     #                 Residual(PreNorm(8*dim, Attention(8*dim, heads = 8*heads, dim_head = 8*dim_head, dropout = dropout))),
            #     #                 Residual(PreNorm(8*dim, FeedForward(8*dim, 8*mlp_dim, dropout = dropout)))
            #     #             ])
            # ])
    def forward(self, x, mask=None):
        for idx, (attn, ff) in enumerate(self.layers):

            x = attn(x, mask=mask)
            x = ff(x)

            # if idx != 2:
            #     x0 = x[:, 0::2, :]
            #     x1 = x[:, 1::2, :]
            #     x = torch.cat((x0, x1), -1)
            #     x = self.norm[idx](x)
            #     x = self.reduction[idx](x)

        return x


class double_conv(nn.Module):
    ''' Conv => Batch_Norm => ReLU => Conv2d => Batch_Norm => ReLU
    '''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.re1 = swish  # nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.re2 = swish  # nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x0 = self.conv1x1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.re1(x)
        x = self.conv2(x)
        x = x + x0
        x = self.bn2(x)
        x = self.re2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self,dim, trans_depth, heads, dim_head, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.t1 = Transformer(dim, 2, heads//2, dim_head, mlp_dim, dropout)
        self.up1=nn.Conv1d(dim,2*dim,kernel_size=3,stride=2,padding=1)
        self.t2 = Transformer(2*dim, trans_depth, heads, 2*dim_head, 2*mlp_dim, dropout)
        self.up2=nn.Conv1d(2*dim,4*dim,kernel_size=3,stride=2,padding=1)
        self.t3 = Transformer(4*dim, trans_depth, heads, 4*dim_head, 4*mlp_dim, dropout)
    def forward(self,x, mask=None):
        x=self.t1(x)
        x=x.permute(0,2,1)
        # print(x.shape)
        x=self.up1(x)
        # print(x.shape)
        x=x.permute(0,2,1)
        # print(x.shape)
        x=self.t2(x)
        x = x.permute(0, 2, 1)
        x=self.up2(x)
        x = x.permute(0, 2, 1)
        x=self.t3(x)
        return x

class TransFoot(nn.Module):  # 参数统一？
    def __init__(self, image_size, patch_size, dim, trans_depth, heads, mlp_dim,
                 channels=3, num_classes=8, dim_head=360, dropout=0.1, emb_dropout=0., head_channels=512):  #这里dim_head很费显存
        super(TransFoot, self).__init__()
        self.patch_size = patch_size
        num_patches = image_size[0] // patch_size
        # num_patches = image_size[1]
        patch_dim = channels * patch_size
        # patch_dim = channels * image_size[0]

        # self.head=double_conv(1,3)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))  # 位置编码，这里num_patchs不加1
        self.patch_to_embedding = nn.Linear(patch_dim, dim)  # P^2C->D
        self.dropout = nn.Dropout(emb_dropout)
        # self.transformer = Transformer(dim, trans_depth, heads, dim_head, mlp_dim, dropout)
        self.transformer = TransformerBlock(dim, 3, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)  #1
        )
        self.mlp_head_P = nn.Sequential(
            nn.LayerNorm(4*dim),
            nn.Linear(4*dim, 1)  #1
        )
        self.mlp_head1 = nn.Sequential(
            nn.LayerNorm(num_patches),
            nn.Linear(num_patches, num_classes)
        )
        # self.mlp_head1_P0 = nn.Sequential(
        #     nn.LayerNorm(num_patches//4),
        #     nn.Linear(num_patches//4, num_patches//4)
        # )
        self.mlp_head1_P = nn.Sequential(
            nn.LayerNorm(num_patches//4),
            nn.Linear(num_patches//4, num_classes)
        )
        self.ac0=nn.ReLU()
        self.ac=nn.ReLU()
        # self.conv=nn.Conv2d(dim,1,kernel_size=(1,3),padding=(0,1),padding_mode='replicate')
        # self.conv=nn.Conv2d(8,1,kernel_size=1,stride=1,padding=0,padding_mode='replicate')
        # # self.bn=nn.BatchNorm2d(1)
        # self.mlp=nn.Linear(num_patches,8)

        self.pad = nn.ReplicationPad2d((3,3,0,0))
        # self.apply(self._init_weights)

    def forward(self, img, mask=None):
        p = self.patch_size
        # img=self.head(img)
        b0,c0,h0,w0=img.shape
        # 图片分块
        x = rearrange(img, 'b c (h p) w ->(b w) h (c p)', p=p)
        # x = rearrange(img,'b c h w->b w (c h)')
        # 降维(b,N,d)
        # print(x.shape)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        # Positional Encoding：(b,N+1,d)
        x += self.pos_embedding[:, :(n)]  # 这里n应该不加1
        x = self.dropout(x)

        # transformer
        # print('tran', x.shape)
        x = self.transformer(x, mask)

        # x = x.mean(dim=1)   #1.这里不用mean，2.这里dim不为1    尝试dim=2下面的mlp_head和rearrange就要改了
        # x = self.to_latent(x)
        #下面使用卷积和mlp模块
        # x=rearrange(x,'(b w) n d ->b d n w',w=30)  #b n d ->b d n w
        # x=self.conv(x)        #b d n w->b 1 n w
        # # x=self.bn(x)
        # x=swish((x))
        # x=x.permute(0,1,3,2)  #b 1 n w->b 1 w n
        # x=self.mlp(x)         #b 1,w,n->b 1 w 8
        # x=x.permute(0,1,3,2)  #b 1 w 8->b 1 8 w


        # F.conv2d

        # print(x.shape)
        ##############
        # print(x.shape)
        # x = self.mlp_head(x)   #dim->8,不太合理
        x = self.mlp_head_P(x)   #dim->8,不太合理
        x=x.permute(0,2,1)
        # x = self.mlp_head1(x)
        # x = self.mlp_head1_P0(x)
        x = self.mlp_head1_P(x)
        # print(self.mlp_head1_P[0],'*****')
        # print(self.mlp_head1_P[1].weight)
        ################
        # j=torch.ones_like(x)
        # j[:, :, 0]=self.ac0(x[:,:,0])
        # for i in range(1,8):
        #     a=self.ac(x[:,:,i]-x[:,:,i-1])
        #     j[:,:,i]=x[:,:,i-1]+a
        # ###############
        x = rearrange(x, '(b w) c n -> b c n w', w=w0)

        #######
        # x1=self.pad(x)
        # # k = torch.Tensor([[[[1 / 3, 1 / 3, 1 / 3]]]])  # .cuda()
        # # k=torch.Tensor([[[[1/5,1/5,1/5,1/5,1/5]]]])#.cuda()
        # k=torch.Tensor([[[[1/7,1/7,1/7,1/7,1/7,1/7,1/7]]]])#.cuda()
        # # k = torch.repeat_interleave(k, 1, 0)
        # # print(x1.shape,k.shape)
        # x=F.conv2d(x1,k,stride=1)
        ########
        ###########
        # x=self.conv(x)
        ###########
        # x=x.permute(0,2,1)
        # x = self.mlp_head1(x)
        # x = x.permute(0, 2, 1)
        # x = self.mlp_head(x)
        # x = rearrange(x, '(b w) n c -> b c n w', w=50)
        ############
        # x=torch.reshape(x,)
        # x = rearrange(x, '(b c) n w->b c n w', c=1)
        # MLP classification layer

        # x = self.mlp_head(x)
        # # print(x.shape)
        # x = rearrange(x, 'b w (n c) -> b c n w', c=1)

        # print(x.shape)
        return x#,out

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


model = TransFoot(image_size=(224, 100), patch_size=4, dim=4, trans_depth=8, heads=8, mlp_dim=16,dim_head=16,num_classes=8,
                               channels=1).cuda()
# summary(model,(1,224,500))
# dummy_input = torch.randn(2, 1, 224, 500).cuda()
#
# out=model(dummy_input)
# print(out.shape)
# flops, params = profile(model, (dummy_input,))
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
