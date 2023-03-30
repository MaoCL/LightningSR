import torch
import torch.nn as nn
import archs.block as B
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import Upsample
from basicsr.utils import tensor2img
from basicsr.metrics import calculate_psnr
from archs.ACNetv2.dbb_transforms import *
import math
import numpy as np

def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                   padding_mode='zeros'):
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding, dilation=dilation, groups=groups,
                           bias=True, padding_mode=padding_mode)
    se = nn.Sequential()
    se.add_module('conv', conv_layer)
    return se



class IdentityBasedConv1x1(nn.Conv2d):

    def __init__(self, channels, groups=1):
        super(IdentityBasedConv1x1, self).__init__(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=True)

        assert channels % groups == 0
        input_dim = channels // groups
        id_value = np.zeros((channels, input_dim, 1, 1))
        for i in range(channels):
            id_value[i, i % input_dim, 0, 0] = 1
        self.id_tensor = torch.from_numpy(id_value).type_as(self.weight)
        nn.init.zeros_(self.weight)

    def forward(self, input):
        kernel = self.weight + self.id_tensor.to(self.weight.device)
        result = F.conv2d(input, kernel, None, stride=1, padding=0, dilation=self.dilation, groups=self.groups)
        return result

    def get_actual_kernel(self):
        return self.weight + self.id_tensor.to(self.weight.device)


class BNAndPadLayer(nn.Module):
    def __init__(self,
                 pad_pixels,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = self.bn.bias.detach() - self.bn.running_mean * self.bn.weight.detach() / torch.sqrt(self.bn.running_var + self.bn.eps)
            else:
                pad_values = - self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0:self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0:self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


class TBM(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, nonlinear=None, single_init=False):
        super(TBM, self).__init__()
        self.deploy = deploy

        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2

        if deploy:
            self.dbb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True)

        else:

            self.dbb_origin = conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)

            if groups < out_channels:
                self.dbb_1x1 = conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                       padding=0, groups=groups)

            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels   # For mobilenet, it is better to have 2X internal channels

            self.dbb_1x1_kxk = nn.Sequential()
            if internal_channels_1x1_3x3 == in_channels:
                self.dbb_1x1_kxk.add_module('idconv1', IdentityBasedConv1x1(channels=in_channels, groups=groups))
            else:
                self.dbb_1x1_kxk.add_module('conv1', nn.Conv2d(in_channels=in_channels, out_channels=internal_channels_1x1_3x3,
                                                            kernel_size=1, stride=1, padding=0, groups=groups, bias=True))
            self.dbb_1x1_kxk.add_module('conv2', nn.Conv2d(in_channels=internal_channels_1x1_3x3, out_channels=out_channels,
                                                            kernel_size=kernel_size, stride=stride, padding=1, groups=groups, bias=True))


        #   The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing the initialization may be useful in some cases.
        if single_init:
            #   Initialize the bn.weight of dbb_origin as 1 and others as 0. This is not the default setting.
            self.single_init()

    def get_equivalent_kernel_bias(self):
        k_origin, b_origin = self.dbb_origin.conv.weight, self.dbb_origin.conv.bias

        if hasattr(self, 'dbb_1x1'):
            k_1x1, b_1x1 = self.dbb_1x1.conv.weight, self.dbb_1x1.conv.bias
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0

        if hasattr(self.dbb_1x1_kxk, 'idconv1'):
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight, self.dbb_1x1_kxk.conv1.bias
        k_1x1_kxk_second, b_1x1_kxk_second = self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.conv2.bias
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second, b_1x1_kxk_second, groups=self.groups)

        # k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)
        # k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg.to(self.dbb_avg.avgbn.weight.device), self.dbb_avg.avgbn)
        # if hasattr(self.dbb_avg, 'conv'):
        #     k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(self.dbb_avg.conv.weight, self.dbb_avg.bn)
        #     k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second, b_1x1_avg_second, groups=self.groups)
        # else:
        #     k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second

        return transII_addbranch((k_origin, k_1x1, k_1x1_kxk_merged), (b_origin, b_1x1, b_1x1_kxk_merged))

    def switch_to_deploy(self):
        if hasattr(self, 'dbb_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2d(in_channels=self.dbb_origin.conv.in_channels, out_channels=self.dbb_origin.conv.out_channels,
                                     kernel_size=self.dbb_origin.conv.kernel_size, stride=self.dbb_origin.conv.stride,
                                     padding=self.dbb_origin.conv.padding, dilation=self.dbb_origin.conv.dilation, groups=self.dbb_origin.conv.groups, bias=True)
        self.dbb_reparam.weight.data = kernel
        self.dbb_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('dbb_origin')
        # self.__delattr__('dbb_avg')
        if hasattr(self, 'dbb_1x1'):
            self.__delattr__('dbb_1x1')
        self.__delattr__('dbb_1x1_kxk')

    def forward(self, inputs):

        if hasattr(self, 'dbb_reparam'):
            return self.nonlinear(self.dbb_reparam(inputs))

        out = self.dbb_origin(inputs)
        # print(out.size())
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
            # print(out.size())
        # out += self.dbb_avg(inputs)
        x = self.dbb_1x1_kxk(inputs)
        # print(x.size())
        out += self.dbb_1x1_kxk(inputs)
        # print(out.size())
        return self.nonlinear(out)

    def init_gamma(self, gamma_value):
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, "dbb_avg"):
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)



class ESA(nn.Module):
    def __init__(self, n_feats, conv, deploy = False):
        super(ESA, self).__init__()
        self.deploy = deploy
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.deploy = deploy
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = TBM(in_channels = f ,out_channels = f, kernel_size = 3 ,deploy=self.deploy)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.relu(self.conv3(v_max))
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m

    def switch_to_deploy(self):
        self.conv3.switch_to_deploy()

class TBB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, nonlinear=None, single_init=False):
        super(TBB, self).__init__()

        self.deploy = deploy
        self.G1 = TBM(in_channels = in_channels ,out_channels = out_channels, kernel_size = kernel_size ,deploy=deploy)
        self.act1 = nn.GELU()
        self.G2 = TBM(in_channels = in_channels ,out_channels = out_channels, kernel_size = kernel_size ,deploy=deploy)
        self.act2 = nn.GELU()
        self.G3 = TBM(in_channels = in_channels ,out_channels = out_channels, kernel_size = kernel_size ,deploy=deploy)
        self.act3 = nn.GELU()
        self.conv = B.conv_layer(in_channels, in_channels, kernel_size=3)
        self.esa = ESA(in_channels, nn.Conv2d,deploy=deploy)

        
    def forward(self,input):
        output = self.act1(self.G1(input))
        output = self.act2(self.G2(output))
        output = self.act3(self.G3(output))
        output = output + input
        output = self.conv(output)
        output = self.esa(output)

        return output

    def switch_to_deploy(self):
         for group in [self.G1, self.G2, self.G3, self.esa]:
            group.switch_to_deploy()


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

class _build_block(nn.Module):
    """Residual Group of RCAB.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_block (int): Block number in the body network.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self,  in_nc, nf, step, n_layer_curr, kernel_size = 3,deploy=False):
        super(_build_block, self).__init__()
        n_layer_curr += step
        self.deploy = deploy
        self.B1 = TBB(in_channels = nf, out_channels = nf, kernel_size = 3 ,deploy=self.deploy)
        self.B2 = TBB(in_channels = nf, out_channels = nf, kernel_size = 3 ,deploy=self.deploy)
        self.B3 = TBB(in_channels = nf, out_channels = nf, kernel_size = 3 ,deploy=self.deploy)
        self.B4 = TBB(in_channels = nf, out_channels = nf, kernel_size = 3 ,deploy=self.deploy)  
        self.c = B.conv_block(nf * step, nf, kernel_size=1, act_type='lrelu')

    def forward(self, x):
        out_B1 = self.B1(x)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B = self.c(torch.cat([out_B1, out_B2, out_B3,out_B4], dim=1))
        return out_B + x

    def switch_to_deploy(self):
         for group in [self.B1, self.B2, self.B3,self.B4]:
            group.switch_to_deploy()


#------------------------------------------------------------------#
@ARCH_REGISTRY.register()
class MSDSRNet_eight(nn.Module):
    def __init__(self,
                 in_nc=3,
                 nf=64,
                 num_modules=4,
                 num_block=8,
                 out_nc=3,
                 upscale=4,
                 img_range=255.,
                 rgb_mean=(0.4480, 0.4370, 0.4040),
                 deploy=False):
        super(MSDSRNet_eight, self).__init__()
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.nBlocks = num_block
        self.deploy = deploy
        self.steps = num_modules

        n_layer_curr =  0

        # print("building network of steps: ")


        nIn = in_nc
        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)
        for i in range(self.nBlocks):
            print(' ********************** Block {} '
                  ' **********************'.format(i + 1))
            m = _build_block(in_nc = nIn, nf = nf, step = self.steps,n_layer_curr = n_layer_curr, deploy=deploy)
            self.blocks.append(m)
            n_layer_curr += self.steps


            self.upsample.append(self._build_upsample(nf, upscale))
        # self.Ups = Upsample(upscale, nf)
        # self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        for m in self.blocks:
            if hasattr(m, '__iter__'):
                for _m in m:
                    self._init_weights(_m)
            else:
                self._init_weights(m)
 # ------------------------------------------------#
        for m in self.upsample:
            if hasattr(m, '__iter__'):
                for _m in m:
                    self._init_weights(_m)
            else:
                self._init_weights(m)
 # ------------------------------------------------#

    def _build_upsample(self, nf, scale):

            layers = []
            if (scale & (scale - 1)) == 0:  # scale = 2^n
                for _ in range(int(math.log(scale, 2))):
                    layers.append(nn.Conv2d(nf, 4 * nf, 3, 1, 1))
                    layers.append(nn.PixelShuffle(2))
            elif scale == 3:
                layers.append(nn.Conv2d(nf, 9 * nf, 3, 1, 1))
                layers.append(nn.PixelShuffle(3))
            else:
                raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')

            layers.append(nn.Conv2d(nf, 3, 3, 1, 1))



            return nn.Sequential(*layers)


    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def forward(self, x, predict):
        res = []
        index = 0
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        x = self.fea_conv(x)
        if self.training:
            for i in range(self.nBlocks):
                x = self.blocks[i](x)
                output = self.upsample[i](x)
                res.append(output / self.img_range + self.mean)
        else:
            for i in range(predict):
                x = self.blocks[i](x)
                index = i
            output = self.upsample[index](x)
            res.append(output / self.img_range + self.mean)
        return res