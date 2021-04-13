import torch.nn as nn
import torch
from models.networks_other import init_weights


class UNetConv2D(nn.Module):
    def __init__(self, in_size, out_size, kernel=(3,3), pad=(1,1), stride=(1,1), bn = True):
        super(UNetConv2D, self).__init__()
        if bn:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel, stride, pad),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, kernel, 1, pad),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel, stride, pad),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, kernel, 1, pad),
                                       nn.ReLU(inplace=True),)

        #initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

class UNetConv3D(nn.Module):
    def __init__(self, in_size, out_size, kernel=(3,3,3), pad=(1,1,1), stride=(1,1,1), bn = True):
        super(UNetConv3D, self).__init__()
        if bn:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel, stride, pad),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel, 1, pad),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel, stride, pad),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel, 1, pad),
                                       nn.ReLU(inplace=True),)
        #initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UnetUp2D(nn.Module):
    def __init__(self, in_size, out_size, bn = True, up_mode='biline'):
        super(UnetUp2D, self).__init__()
        self.up_mode = up_mode

        if self.up_mode == 'biline':
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            self.conv0 = nn.Conv2d(in_size, out_size, kernel_size=(2,2))
            self.conv1 = UNetConv2D(in_size, out_size, bn=bn)
        elif self.up_mode == 'deconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, 2, stride=2)
            self.conv1 = UNetConv2D(in_size, out_size, bn=bn)

        #initialise the blocks
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.ConvTranspose2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        if self.up_mode == 'biline':
            outputs2 = self.conv0(nn.functional.pad(self.up(inputs2), (1,0,1,0)))
            return self.conv1(torch.cat([inputs1, outputs2], 1))
        elif self.up_mode == 'deconv':
            outputs2 = self.up(inputs2)
            return self.conv1(torch.cat([inputs1, outputs2], 1))

class UnetUp3D(nn.Module):
    def __init__(self, in_size, out_size, bn = True):
        super(UnetUp3D, self).__init__()
        if self.up_mode == 'triline':
            self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
            self.conv0 = nn.Conv3d(in_size, out_size, kernel_size=(2,2,2))
            self.conv1 = UNetConv3D(in_size, out_size, bn=bn)
        elif self.up_mode == 'deconv':
            self.up = nn.ConvTranspose3d(in_size, out_size, 2, stride=2)
            self.conv1 = UNetConv3D(in_size, out_size, bn=bn)

        #initialise the blocks
        for m in self.children():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
    def forward(self, inputs1, inputs2):
        # outputs2 = self.conv0(self.up(inputs2))
        if self.up_mode == 'triline':
            outputs2 = self.conv0(nn.functional.pad(self.up(inputs2), (1,0,1,0,1,0)))
            return self.conv1(torch.cat([inputs1, outputs2], 1))
        elif self.up_mode == 'deconv':
            outputs2 = self.up(inputs2)
            return self.conv1(torch.cat([inputs1, outputs2], 1))


class UNETRSkip(nn.Module):
    def __init__(self,d_model, filters, mode = "deconv", bn = True):
        super(UNETRSkip, self).__init__()

        self.d_model = d_model
        self.filters = [d_model] + filters
        self.n_module = len(filters)
        self.bn = True

        self.module_list = []
        for i in range(self.n_module):
            l = [nn.Conv3d(self.filters[i], self.filters[i+1], kernel_size=3)]
            if bn: l.append(nn.BatchNorm3d(self.filters[i+1]))
            l.append(nn.ReLU(inplace=True))
            l.append(nn.ConvTranspose3d(self.filters[i+1], self.filters[i+1], 2, stride = 2, padding=1))

            self.module_list.append(nn.Sequential(*l))
        self.module_list = nn.Sequential(*self.module_list)

        for m in self.children():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.ConvTranspose3d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        return self.module_list(x)