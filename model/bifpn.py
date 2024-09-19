import torch
import torch.nn as nn
# from utils.anchors import Anchors
# from nets.efficientnet import EfficientNet as EffNet
from utils.layers import (Conv2dStaticSamePadding, MaxPool2dStaticSamePadding,
                          MemoryEfficientSwish, Swish)


# ----------------------------------#
#   Xception中深度可分离卷积
#   先3x3的深度可分离卷积
#   再1x1的普通卷积
# ----------------------------------#
class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels, kernel_size=3, stride=1,
                                                      groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-5)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x

def create_convtranspose_layer(conv_channels, num_channels, deconv_with_bias):
    return nn.Sequential(
        nn.Conv2d(conv_channels, num_channels, 1, stride=1,
                  bias=True, groups=1),
        nn.ConvTranspose2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0,
            bias=deconv_with_bias
        ),
        nn.BatchNorm2d(num_channels,momentum=0.01,eps=1e-5),
        nn.ReLU(inplace=True)
    )

# 官方代码中的num_channels里提供了[64, 88, 112, 160, 224, 288, 384, 384]这些选项
class BiFPN(nn.Module):
    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True):
        super(BiFPN, self).__init__()
        self.deconv_with_bias = False
        self.epsilon = epsilon
        self.conv6_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.first_time = first_time
        if self.first_time:
            # # 获取到了efficientnet的最后三层，对其进行通道的下压缩
            # self.p5_down_channel = nn.Sequential(
            #     Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
            #     nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            # )
            # self.p4_down_channel = nn.Sequential(
            #     Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
            #     nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            # )
            # self.p3_down_channel = nn.Sequential(
            #     Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
            #     nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            # )

            # 将convnext中的每一层进行一次上采样,并调整通道数
            self.p6_downchannel_sizex2 = create_convtranspose_layer(conv_channels[3],num_channels,
                                                                   self.deconv_with_bias)
            self.p5_downchannel_sizex2 = create_convtranspose_layer(conv_channels[2], num_channels,
                                                                   self.deconv_with_bias)
            self.p5_downchannel_sizex2_2 = create_convtranspose_layer(conv_channels[2], num_channels,
                                                                   self.deconv_with_bias)
            self.p4_downchannel_sizex2 = create_convtranspose_layer(conv_channels[1], num_channels,
                                                                   self.deconv_with_bias)
            self.p4_downchannel_sizex2_2 = create_convtranspose_layer(conv_channels[1], num_channels,
                                                                   self.deconv_with_bias)
            self.p3_downchannel_sizex2 = create_convtranspose_layer(conv_channels[0], num_channels,
                                                                   self.deconv_with_bias)

            # # 对输入进来的p5进行宽高的下采样
            # self.p5_to_p6 = nn.Sequential(
            #     Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
            #     nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            #     MaxPool2dStaticSamePadding(3, 2)
            # )
            # self.p6_to_p7 = nn.Sequential(
            #     MaxPool2dStaticSamePadding(3, 2)
            # )
            #
            # # BIFPN第一轮的时候，跳线那里并不是同一个in
            # self.p4_down_channel_2 = nn.Sequential(
            #     Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
            #     nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            # )
            # self.p5_down_channel_2 = nn.Sequential(
            #     Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
            #     nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            # )

        # 简易注意力机制的weights
        # self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        # self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        # self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        # self.p7_w2_relu = nn.ReLU()

        self.attention = attention

    def forward(self, inputs):
        """ bifpn模块结构示意图
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """
        if self.attention:
            p3_out, p4_out, p5_out, p6_out= self._forward_fast_attention(inputs)
        else:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward(inputs)

        return p3_out, p4_out, p5_out, p6_out

    def _forward_fast_attention(self, inputs):
        # ------------------------------------------------#
        #   当phi=1、2、3、4、5的时候使用fast_attention
        #   获得四个shape的有效特征层
        #   分别是 C3  64, 64, 96
        #         C4  32, 32, 192
        #         C5  16, 16, 384
        #         C6  8,  8,  768
        # ------------------------------------------------#
        if self.first_time:
            # ------------------------------------------------------------------------#
            #   第一次BIFPN需要 下采样 与 调整通道 获得 p3_in p4_in p5_in p6_in p7_in
            # ------------------------------------------------------------------------#
            p3, p4, p5, p6 = inputs
            # -------------------------------------------#
            #   首先对通道数进行调整
            #   C3 64, 64, 96 -> 128, 128, 64
            # -------------------------------------------#
            p3_in = self.p3_downchannel_sizex2(p3)

            # -------------------------------------------#
            #   首先对通道数进行调整
            #   C4 32, 32, 192 -> 64, 64, 64
            #                  -> 64, 64, 64
            # -------------------------------------------#
            p4_in_1 = self.p4_downchannel_sizex2(p4)
            p4_in_2 = self.p4_downchannel_sizex2_2(p4)

            # -------------------------------------------#
            #   首先对通道数进行调整
            #   C5 16, 16, 384 -> 32, 32, 64
            #                  -> 32, 32, 64
            # -------------------------------------------#
            p5_in_1 = self.p5_downchannel_sizex2(p5)
            p5_in_2 = self.p5_downchannel_sizex2_2(p5)

            # -------------------------------------------#
            #   首先对通道数进行调整
            #   C6 8, 8, 768 -> 16, 16, 64
            #                -> 16, 16, 64
            # -------------------------------------------#
            p6_in = self.p6_downchannel_sizex2(p6)
            # # -------------------------------------------#
            # #   对P6_in进行下采样，调整宽高
            # #   P6_in 8, 8, 64 -> 4, 4, 64
            # # -------------------------------------------#
            # p7_in = self.p6_to_p7(p6_in)

            # # 简单的注意力机制，用于确定更关注p7_in还是p6_in
            # p6_w1 = self.p6_w1_relu(self.p6_w1)
            # weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
            # p6_td = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

            # 简单的注意力机制，用于确定更关注p6_up还是p5_in
            p5_w1 = self.p5_w1_relu(self.p5_w1)
            weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
            p5_td = self.conv5_up(self.swish(weight[0] * p5_in_1 + weight[1] * self.p5_upsample(p6_in)))

            # 简单的注意力机制，用于确定更关注p5_up还是p4_in
            p4_w1 = self.p4_w1_relu(self.p4_w1)
            weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
            p4_td = self.conv4_up(self.swish(weight[0] * p4_in_1 + weight[1] * self.p4_upsample(p5_td)))

            # 简单的注意力机制，用于确定更关注p4_up还是p3_in
            p3_w1 = self.p3_w1_relu(self.p3_w1)
            weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
            p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_td)))

            # 简单的注意力机制，用于确定更关注p4_in_2还是p4_up还是p3_out
            p4_w2 = self.p4_w2_relu(self.p4_w2)
            weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
            p4_out = self.conv4_down(
                self.swish(weight[0] * p4_in_2 + weight[1] * p4_td + weight[2] * self.p4_downsample(p3_out)))

            # 简单的注意力机制，用于确定更关注p5_in_2还是p5_up还是p4_out
            p5_w2 = self.p5_w2_relu(self.p5_w2)
            weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
            p5_out = self.conv5_down(
                self.swish(weight[0] * p5_in_2 + weight[1] * p5_td + weight[2] * self.p5_downsample(p4_out)))

            # 简单的注意力机制，用于确定更关注p6_in还是p6_up还是p5_out
            p6_w2 = self.p6_w2_relu(self.p6_w2)
            weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
            p6_out = self.conv6_down(
                self.swish(weight[0] * p6_in + weight[1] * self.p6_downsample(p5_out)))

            # # 简单的注意力机制，用于确定更关注p7_in还是p7_up还是p6_out
            # p7_w2 = self.p7_w2_relu(self.p7_w2)
            # weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
            # p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))
        else:
            p3_in, p4_in, p5_in, p6_in = inputs

            # # 简单的注意力机制，用于确定更关注p7_in还是p6_in
            # p6_w1 = self.p6_w1_relu(self.p6_w1)
            # weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
            # p6_td = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

            # 简单的注意力机制，用于确定更关注p6_up还是p5_in
            p5_w1 = self.p5_w1_relu(self.p5_w1)
            weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
            p5_td = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_in)))

            # 简单的注意力机制，用于确定更关注p5_up还是p4_in
            p4_w1 = self.p4_w1_relu(self.p4_w1)
            weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
            p4_td = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_td)))

            # 简单的注意力机制，用于确定更关注p4_up还是p3_in
            p3_w1 = self.p3_w1_relu(self.p3_w1)
            weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
            p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_td)))

            # 简单的注意力机制，用于确定更关注p4_in还是p4_up还是p3_out
            p4_w2 = self.p4_w2_relu(self.p4_w2)
            weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
            p4_out = self.conv4_down(
                self.swish(weight[0] * p4_in + weight[1] * p4_td + weight[2] * self.p4_downsample(p3_out)))

            # 简单的注意力机制，用于确定更关注p5_in还是p5_up还是p4_out
            p5_w2 = self.p5_w2_relu(self.p5_w2)
            weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
            p5_out = self.conv5_down(
                self.swish(weight[0] * p5_in + weight[1] * p5_td + weight[2] * self.p5_downsample(p4_out)))

            # 简单的注意力机制，用于确定更关注p6_in还是p6_up还是p5_out
            p6_w2 = self.p6_w2_relu(self.p6_w2)
            weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
            p6_out = self.conv6_down(
                self.swish(weight[0] * p6_in + weight[1] * self.p6_downsample(p5_out)))

            # # 简单的注意力机制，用于确定更关注p7_in还是p7_up还是p6_out
            # p7_w2 = self.p7_w2_relu(self.p7_w2)
            # weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
            # p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out

class BiFPN_noup(nn.Module):
    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False,
                 attention=True):
        super(BiFPN_noup, self).__init__()
        self.deconv_with_bias = False
        self.epsilon = epsilon
        self.conv6_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.first_time = first_time
        if self.first_time:
            # # 获取到了efficientnet的最后三层，对其进行通道的下压缩
            # self.p5_down_channel = nn.Sequential(
            #     Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
            #     nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            # )
            # self.p4_down_channel = nn.Sequential(
            #     Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
            #     nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            # )
            # self.p3_down_channel = nn.Sequential(
            #     Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
            #     nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            # )

            # 将convnext中的每一层进行一次上采样,并调整通道数
            self.p6_downchannel_sizex2 = create_convtranspose_layer(conv_channels[3], num_channels,
                                                                    self.deconv_with_bias)
            self.p6_downchannel_sizex2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[3], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_downchannel_sizex2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_downchannel_sizex2_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_downchannel_sizex2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_downchannel_sizex2_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_downchannel_sizex2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )


        # 简易注意力机制的weights
        # self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        # self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        # self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        # self.p7_w2_relu = nn.ReLU()

        self.attention = attention

    def forward(self, inputs):
        """ bifpn模块结构示意图
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """
        if self.attention:
            p3_out, p4_out, p5_out, p6_out = self._forward_fast_attention(inputs)
        else:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward(inputs)

        return p3_out, p4_out, p5_out, p6_out

    def _forward_fast_attention(self, inputs):
        # ------------------------------------------------#
        #   当phi=1、2、3、4、5的时候使用fast_attention
        #   获得四个shape的有效特征层
        #   分别是 C3  64, 64, 96
        #         C4  32, 32, 192
        #         C5  16, 16, 384
        #         C6  8,  8,  768
        # ------------------------------------------------#
        if self.first_time:
            # ------------------------------------------------------------------------#
            #   第一次BIFPN需要 下采样 与 调整通道 获得 p3_in p4_in p5_in p6_in p7_in
            # ------------------------------------------------------------------------#
            p3, p4, p5, p6 = inputs
            # -------------------------------------------#
            #   首先对通道数进行调整
            #   C3 64, 64, 96 -> 128, 128, 64
            # -------------------------------------------#
            p3_in = self.p3_downchannel_sizex2(p3)

            # -------------------------------------------#
            #   首先对通道数进行调整
            #   C4 32, 32, 192 -> 64, 64, 64
            #                  -> 64, 64, 64
            # -------------------------------------------#
            p4_in_1 = self.p4_downchannel_sizex2(p4)
            p4_in_2 = self.p4_downchannel_sizex2_2(p4)

            # -------------------------------------------#
            #   首先对通道数进行调整
            #   C5 16, 16, 384 -> 32, 32, 64
            #                  -> 32, 32, 64
            # -------------------------------------------#
            p5_in_1 = self.p5_downchannel_sizex2(p5)
            p5_in_2 = self.p5_downchannel_sizex2_2(p5)

            # -------------------------------------------#
            #   首先对通道数进行调整
            #   C6 8, 8, 768 -> 16, 16, 64
            #                -> 16, 16, 64
            # -------------------------------------------#
            p6_in = self.p6_downchannel_sizex2(p6)
            # # -------------------------------------------#
            # #   对P6_in进行下采样，调整宽高
            # #   P6_in 8, 8, 64 -> 4, 4, 64
            # # -------------------------------------------#
            # p7_in = self.p6_to_p7(p6_in)

            # # 简单的注意力机制，用于确定更关注p7_in还是p6_in
            # p6_w1 = self.p6_w1_relu(self.p6_w1)
            # weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
            # p6_td = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

            # 简单的注意力机制，用于确定更关注p6_up还是p5_in
            p5_w1 = self.p5_w1_relu(self.p5_w1)
            weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
            p5_td = self.conv5_up(self.swish(weight[0] * p5_in_1 + weight[1] * self.p5_upsample(p6_in)))

            # 简单的注意力机制，用于确定更关注p5_up还是p4_in
            p4_w1 = self.p4_w1_relu(self.p4_w1)
            weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
            p4_td = self.conv4_up(self.swish(weight[0] * p4_in_1 + weight[1] * self.p4_upsample(p5_td)))

            # 简单的注意力机制，用于确定更关注p4_up还是p3_in
            p3_w1 = self.p3_w1_relu(self.p3_w1)
            weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
            p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_td)))

            # 简单的注意力机制，用于确定更关注p4_in_2还是p4_up还是p3_out
            p4_w2 = self.p4_w2_relu(self.p4_w2)
            weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
            p4_out = self.conv4_down(
                self.swish(weight[0] * p4_in_2 + weight[1] * p4_td + weight[2] * self.p4_downsample(p3_out)))

            # 简单的注意力机制，用于确定更关注p5_in_2还是p5_up还是p4_out
            p5_w2 = self.p5_w2_relu(self.p5_w2)
            weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
            p5_out = self.conv5_down(
                self.swish(weight[0] * p5_in_2 + weight[1] * p5_td + weight[2] * self.p5_downsample(p4_out)))

            # 简单的注意力机制，用于确定更关注p6_in还是p6_up还是p5_out
            p6_w2 = self.p6_w2_relu(self.p6_w2)
            weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
            p6_out = self.conv6_down(
                self.swish(weight[0] * p6_in + weight[1] * self.p6_downsample(p5_out)))

            # # 简单的注意力机制，用于确定更关注p7_in还是p7_up还是p6_out
            # p7_w2 = self.p7_w2_relu(self.p7_w2)
            # weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
            # p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))
        else:
            p3_in, p4_in, p5_in, p6_in = inputs

            # # 简单的注意力机制，用于确定更关注p7_in还是p6_in
            # p6_w1 = self.p6_w1_relu(self.p6_w1)
            # weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
            # p6_td = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

            # 简单的注意力机制，用于确定更关注p6_up还是p5_in
            p5_w1 = self.p5_w1_relu(self.p5_w1)
            weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
            p5_td = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_in)))

            # 简单的注意力机制，用于确定更关注p5_up还是p4_in
            p4_w1 = self.p4_w1_relu(self.p4_w1)
            weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
            p4_td = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_td)))

            # 简单的注意力机制，用于确定更关注p4_up还是p3_in
            p3_w1 = self.p3_w1_relu(self.p3_w1)
            weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
            p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_td)))

            # 简单的注意力机制，用于确定更关注p4_in还是p4_up还是p3_out
            p4_w2 = self.p4_w2_relu(self.p4_w2)
            weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
            p4_out = self.conv4_down(
                self.swish(weight[0] * p4_in + weight[1] * p4_td + weight[2] * self.p4_downsample(p3_out)))

            # 简单的注意力机制，用于确定更关注p5_in还是p5_up还是p4_out
            p5_w2 = self.p5_w2_relu(self.p5_w2)
            weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
            p5_out = self.conv5_down(
                self.swish(weight[0] * p5_in + weight[1] * p5_td + weight[2] * self.p5_downsample(p4_out)))

            # 简单的注意力机制，用于确定更关注p6_in还是p6_up还是p5_out
            p6_w2 = self.p6_w2_relu(self.p6_w2)
            weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
            p6_out = self.conv6_down(
                self.swish(weight[0] * p6_in + weight[1] * self.p6_downsample(p5_out)))

            # # 简单的注意力机制，用于确定更关注p7_in还是p7_up还是p6_out
            # p7_w2 = self.p7_w2_relu(self.p7_w2)
            # weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
            # p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out
