import math
import torch
import time
from torch import nn
from model.nets.efficientdet import efficient
from model.convnext import convnext_tiny, convnext_tiny_up,convnext_tiny_small
from model.bifpn import BiFPN, BiFPN_noup
from model.ASFF import ASFF
from model.head import Head, Sizex2

class Convnext_Centernet(nn.Module):
    def __init__(self, bifpn_repeat,gt_enlarge, pretrained=False, phi=0):
        super(Convnext_Centernet, self).__init__()
        # 控制sizex2层的输出通道.分别是做1次到做4次上采样的输出通道
        self.sizex2_num_filters=[128,64,64,32]
        self.pretrain = pretrained
        self.phi = phi
        # 输入:(b,1,128,128)
        # 输出:[(b,96,64,64),
        #      (b,192,32,32),
        #      (b,384,16,16),
        #      (b,768,8,8)]

        self.backbone = convnext_tiny(pretrained=self.pretrain,in_22k=False,phi=phi)
        self.bifn = nn.Sequential(
            *[BiFPN(num_channels=128,
                    conv_channels=[96,192,384,768],
                    first_time=True if _ == 0 else False,)
              for _ in range(bifpn_repeat)]
        )
        self.asff = ASFF(128,128,True)
        self.sizex2 = Sizex2(128,gt_enlarge+1)
        self.head = Head((self.sizex2_num_filters[gt_enlarge]))

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self,x):
        feat = self.backbone(x)
        #print(f'backbone feat :{feat[0].shape} {feat[1].shape} {feat[2].shape} {feat[3].shape}')
        feat = self.bifn(feat)
        #print(f'bifn feat :{feat[0].shape} {feat[1].shape} {feat[2].shape} {feat[3].shape}')
        # asff的输出的size是64*64.本来想设置成128*128的,但那样会让这个模型占用24G显存
        feat = self.asff(feat)
        #print(f'asff feat :{feat.shape}')
        output = self.sizex2(feat)
        #print(f'sizex2 output :{output.shape}')

        return self.head(output)


class Convnext_Centernet_e(nn.Module):
    def __init__(self, bifpn_repeat,gt_enlarge, bx=0,pretrained=False, phi=0):
        super(Convnext_Centernet_e, self).__init__()
        dims=[
            [24,40,112,320],
            [24,40,112,320],
            [24,48,120,352],
            [32,48,136,384],
            [32,56,160,448],
            [40,64,176,512],
            [40,72,200,576],
            [48,80,224,640]
        ]
        # 控制sizex2层的输出通道.分别是做1次到做4次上采样的输出通道
        self.sizex2_num_filters=[96,64,64,32]
        self.pretrain = pretrained
        self.phi = phi
        self.bx = bx
        # 输入:(b,1,128,128)
        # 输出:[(b,96,64,64),
        #      (b,192,32,32),
        #      (b,384,16,16),
        #      (b,768,8,8)]

        self.backbone = efficient(phi=self.phi,bx=self.bx,pretrained=self.pretrain,in_channels=1)
        self.bifn = nn.Sequential(
            *[BiFPN(num_channels=96,
                    conv_channels=dims[self.bx],
                    first_time=True if _ == 0 else False,)
              for _ in range(bifpn_repeat)]
        )
        self.asff = ASFF(96,96,True)
        self.sizex2 = Sizex2(96,gt_enlarge+1,num_filters=[96])
        self.head = Head((self.sizex2_num_filters[gt_enlarge]))

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self,x):
        feat = self.backbone(x)
        #print(f'backbone feat :{feat[0].shape} {feat[1].shape} {feat[2].shape} {feat[3].shape}')
        feat = self.bifn(feat)
        #print(f'bifn feat :{feat[0].shape} {feat[1].shape} {feat[2].shape} {feat[3].shape}')
        # asff的输出的size是64*64.本来想设置成128*128的,但那样会让这个模型占用24G显存
        feat = self.asff(feat)
        #print(f'asff feat :{feat.shape}')
        output = self.sizex2(feat)
        #print(f'sizex2 output :{output.shape}')

        return self.head(output)


class Convnext_Centernet_small(nn.Module):
    def __init__(self, bifpn_repeat,gt_enlarge, pretrained=False, phi=0):
        super(Convnext_Centernet_small, self).__init__()
        # 控制sizex2层的输出通道.分别是做1次到做4次上采样的输出通道
        self.sizex2_num_filters=[96,64,64,32]
        self.pretrain = pretrained
        self.phi = phi
        # 输入:(b,1,128,128)
        # 输出:[(b,96,64,64),
        #      (b,192,32,32),
        #      (b,384,16,16),
        #      (b,768,8,8)]

        self.backbone = convnext_tiny_small(pretrained=self.pretrain,in_22k=False,phi=phi)
        self.bifn = nn.Sequential(
            *[BiFPN(num_channels=96,
                    conv_channels=[64,128,240,480],
                    first_time=True if _ == 0 else False,)
              for _ in range(bifpn_repeat)]
        )
        self.asff = ASFF(96,96,True)
        self.sizex2 = Sizex2(96,gt_enlarge+1,num_filters=[96])
        self.head = Head((self.sizex2_num_filters[gt_enlarge]))

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self,x):
        feat = self.backbone(x)
        #print(f'backbone feat :{feat[0].shape} {feat[1].shape} {feat[2].shape} {feat[3].shape}')
        feat = self.bifn(feat)
        #print(f'bifn feat :{feat[0].shape} {feat[1].shape} {feat[2].shape} {feat[3].shape}')
        # asff的输出的size是64*64.本来想设置成128*128的,但那样会让这个模型占用24G显存
        feat = self.asff(feat)
        #print(f'asff feat :{feat.shape}')
        output = self.sizex2(feat)
        #print(f'sizex2 output :{output.shape}')

        return self.head(output)





model = Convnext_Centernet_small(bifpn_repeat=1, gt_enlarge=0, pretrained=False,phi=8)
input_tensor = torch.rand(1, 1, 128, 128)
output = model(input_tensor)
print(output[0].shape)
print(output[1].shape)
