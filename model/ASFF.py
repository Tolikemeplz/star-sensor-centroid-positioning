import torch
import torch.nn as nn
import torch.nn.functional as F

def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage


def add_convtranspose(in_ch, out_ch, deconv_with_bias, leaky=True):

    stage = nn.Sequential()

    stage.add_module('convtranspose', nn.ConvTranspose2d(in_channels=in_ch,
                                                            out_channels=out_ch,
                                                            kernel_size=4,
                                                            stride=2,
                                                            padding=1,
                                                            output_padding=0,
                                                            bias=deconv_with_bias))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage




class ASFF(nn.Module):
    def __init__(self, in_channels, out_channels, rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.deconv_with_bias = False
        self.in_channels=in_channels
        self.out_channels=out_channels
        #self.level = level
        #self.dim = [512, 256, 256]
        #self.inter_dim = self.dim[self.level]
        self.inter_dim=in_channels
        # if level==0:
        #     self.stride_level_1 = add_conv(256, self.inter_dim, 3, 2)
        #     self.stride_level_2 = add_conv(256, self.inter_dim, 3, 2)
        #     self.expand = add_conv(self.inter_dim, 1024, 3, 1)
        # elif level==1:
        #     self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
        #     self.stride_level_2 = add_conv(256, self.inter_dim, 3, 2)
        #     self.expand = add_conv(self.inter_dim, 512, 3, 1)
        # elif level==2:
        #     self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
        #     self.expand = add_conv(self.inter_dim, 256, 3, 1)

        self.sizex2_level_0 = add_convtranspose(self.in_channels, self.in_channels, self.deconv_with_bias)
        # self.compress_level_0 = add_conv(self.in_channels, self.inter_dim, 1, 1)
        # self.compress_level_1 = add_conv(self.in_channels, self.inter_dim, 1, 1)
        # self.compress_level_2 = add_conv(self.in_channels, self.inter_dim, 1, 1)

        self.expand = add_conv(self.inter_dim, self.out_channels, 3, 1)


        compress_c = 8 if rfb else 16  #when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c*4, 4, kernel_size=1, stride=1, padding=0)
        self.vis= vis


    def forward(self, x):
        # if self.level==0:
        #     level_0_resized = x_level_0
        #     level_1_resized = self.stride_level_1(x_level_1)
        #
        #     level_2_downsampled_inter =F.max_pool2d(x_level_2, 3, stride=2, padding=1)
        #     level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        #
        # elif self.level==1:
        #     level_0_compressed = self.compress_level_0(x_level_0)
        #     level_0_resized =F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
        #     level_1_resized =x_level_1
        #     level_2_resized =self.stride_level_2(x_level_2)
        # elif self.level==2:
        #     level_0_compressed = self.compress_level_0(x_level_0)
        #     level_0_resized =F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
        #     level_1_resized =F.interpolate(x_level_1, scale_factor=2, mode='nearest')
        #     level_2_resized =x_level_2
        x_level_3, x_level_2, x_level_1, x_level_0 = x

        level_0_sizex2 = self.sizex2_level_0(x_level_0)
        #level_0_compressed = self.compress_level_0(level_0_sizex2)
        level_0_resized = F.interpolate(level_0_sizex2, scale_factor=4, mode='nearest')
        #level_1_compressed = self.compress_level_1(x_level_1)
        level_1_resized = F.interpolate(x_level_1, scale_factor=4, mode='nearest')
        level_2_resized = F.interpolate(x_level_2, scale_factor=2, mode='nearest')
        level_3_resized = x_level_3




        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        level_3_weight_v = self.weight_level_3(level_3_resized)
        # print(level_0_resized.shape)
        # print(level_1_resized.shape)
        # print(level_2_resized.shape)
        # print(level_3_resized.shape)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+\
                            level_1_resized * levels_weight[:,1:2,:,:]+\
                            level_2_resized * levels_weight[:,2:3,:,:]+ \
                            level_3_resized * levels_weight[:,3:4,:,:]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out

# model = ASFF(128,128,True)
# x0 = torch.rand(1, 128, 64, 64)
# x1 = torch.rand(1, 128, 32, 32)
# x2 = torch.rand(1, 128, 16, 16)
# x3 = torch.rand(1, 128, 8, 8)
# x=(x0,x1,x2,x3)
# output = model(x)
# print(output)