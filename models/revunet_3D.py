import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import revtorch.revtorch as rv
import random

id = random.getrandbits(64)

class softmax(nn.Module):
    def __init__(self, dim):
        super(softmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        return F.softmax(x, dim = self.dim)


class OldResidualInner(nn.Module):
    def __init__(self, channels, groups):
        super(OldResidualInner, self).__init__()
        # self.gn = nn.BatchNorm3d(channels)
        self.gn = nn.GroupNorm(groups, channels)
        self.conv = nn.Conv3d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.gn(self.conv(x)), inplace=True)
        return x

class ResidualInner(nn.Module):
    def __init__(self, channels, groups):
        super(ResidualInner, self).__init__()
        self.gn = nn.GroupNorm(groups, channels)
        self.conv = nn.Conv3d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        x = F.relu(self.gn(self.conv(x)), inplace=True)
        return x

class NoGNResidualInner(nn.Module):
    def __init__(self, channels, groups):
        super(NoGNResidualInner, self).__init__()
        self.conv = nn.Conv3d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        x = F.relu(self.conv(x), inplace=True)
        return x

def makeReversibleSequence(channels):
    innerchannels = channels // 2
    groups = 2 if innerchannels > 1 else 1 #channels[0] // 2
    # print("chan, groups" ,channels, groups)
    fBlock = NoGNResidualInner(innerchannels, groups)
    gBlock = NoGNResidualInner(innerchannels, groups)
    #gBlock = nn.Sequential()
    return rv.ReversibleBlock(fBlock, gBlock)

def makeReversibleComponent(channels, blockCount):
    modules = []
    for i in range(blockCount):
        modules.append(makeReversibleSequence(channels))
    return rv.ReversibleSequence(nn.ModuleList(modules))

def getchannelsAtIndex(index, channels):
    if index < 0: index = 0
    if index >= len(channels): index = len(channels) - 1
    return channels[index]

class EncoderModule(nn.Module):
    def __init__(self, inchannels, outchannels, depth, downsample=True):
        super(EncoderModule, self).__init__()
        self.downsample = downsample
        if downsample:
            self.conv = nn.Conv3d(inchannels, outchannels, 1)
        self.reversibleBlocks = makeReversibleComponent(outchannels, depth)

    def forward(self, x):
        if self.downsample:
            x = F.max_pool3d(x, 2)
            x = self.conv(x) #increase number of channels
        x = self.reversibleBlocks(x)
        return x

class DecoderModule(nn.Module):
    def __init__(self, inchannels, outchannels, depth, upsample=True):
        super(DecoderModule, self).__init__()
        self.reversibleBlocks = makeReversibleComponent(inchannels, depth)
        self.upsample = upsample
        self.inc = inchannels
        self.out = outchannels
        if self.upsample:
            self.conv = nn.Conv3d(inchannels, outchannels, 1)

    def forward(self, x, shape):
        x = self.reversibleBlocks(x)
        if self.upsample:
            x = self.conv(x)
            # x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
            print('x s ', x.shape)
            print('i s ', shape)
            x = F.interpolate(x, size=shape[-3:])
            print('x s ', x.shape)
            print(self.inc, self.out)
        # for i in range(1,4):
        #     # print("#" ,x.shape,shape)
        #     if x.shape[-i] != shape[-i]:

        #         tup = [0,0,0,0,0,0]
        #         n_tmp = abs(x.shape[-i] - shape[-i])
        #         tup[i*2 -1] = n_tmp

        #         x = F.pad(x, tuple(tup), 'constant')
        #         # print("##", n_tmp, x.shape)
        return x

class RevUnet3D(nn.Module):
    def __init__(self, inchannels ,channels, out_size, depth = 1, interpolation = None):
        super(RevUnet3D, self).__init__()
        depth = depth
        self.levels = len(channels)

        self.firstConv = nn.Conv3d(inchannels, channels[0], 3, padding=1, bias=False)
        #self.dropout = nn.Dropout3d(0.2, True)
        self.lastConv = nn.Conv3d(channels[0], out_size, 1, bias=True)

        #create encoder levels
        encoderModules = []
        for i in range(self.levels):
            encoderModules.append(EncoderModule(getchannelsAtIndex(i - 1, channels), getchannelsAtIndex(i, channels), depth, i != 0))
        self.encoders = nn.ModuleList(encoderModules)

        #create decoder levels
        decoderModules = []
        for i in range(self.levels):
            decoderModules.append(DecoderModule(getchannelsAtIndex(self.levels - i - 1, channels), getchannelsAtIndex(self.levels - i - 2, channels), depth, i != (self.levels -1)))
        self.decoders = nn.ModuleList(decoderModules)

        self.softmax = softmax(1)

        # self.interpolation = interpolation
        # if self.interpolation != None:
        #     self.interpolation = nn.Upsample(size = interpolation, mode = "trilinear")

    def forward(self, x):
        x = self.firstConv(x)
        inputStack = []
        shapes = [x.shape]
        for i in range(self.levels):
            
            x = self.encoders[i](x)
            shapes.append(x.shape)
            if i < self.levels - 1:
                inputStack.append(x)

        for i in range(self.levels):
            x = self.decoders[i](x, shapes[-(i+1)])
            if i < self.levels - 1:
                x = x + inputStack.pop()

        x = self.lastConv(x)

        # x = self.softmax(x)
        x = F.softmax(x, dim=1)
        # if self.interpolation != None:
        #     x = self.interpolation(x)
        #     return x
        #x = torch.sigmoid(x)
        return x

    @staticmethod
    def apply_argmax_softmax(pred):
        pred = F.softmax(pred, dim=1)
        return pred