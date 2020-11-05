import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import revtorch.revtorch as rv
import random

id = random.getrandbits(64)


class ResidualInner(nn.Module):
    def __init__(self, channels, groups):
        super(ResidualInner, self).__init__()
        # self.gn = nn.BatchNorm3d(channels)
        self.gn = nn.GroupNorm(groups, channels)
        self.conv = nn.Conv3d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.gn(self.conv(x)), inplace=INPLACE)
        return x

def makeReversibleSequence(channels):
    innerchannels = channels // 2
    groups = 2#channels[0] // 2
    fBlock = ResidualInner(innerchannels, groups)
    gBlock = ResidualInner(innerchannels, groups)
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
        if self.upsample:
            self.conv = nn.Conv3d(inchannels, outchannels, 1)

    def forward(self, x):
        x = self.reversibleBlocks(x)
        if self.upsample:
            x = self.conv(x)
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        return x

class RevUnet3D(nn.Module):
    def __init__(self, channels):
        super(RevUnet3D, self).__init__()
        depth = 1
        self.levels = 5

        self.firstConv = nn.Conv3d(1, channels[0], 3, padding=1, bias=False)
        #self.dropout = nn.Dropout3d(0.2, True)
        self.lastConv = nn.Conv3d(channels[0], 2, 1, bias=True)

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


        self.interpolation = nn.Upsample(size = (512,512,256), mode = "trilinear")

    def forward(self, x):
        # tibo_in_shape = x.shape[-3:]
        x = self.firstConv(x)
        #x = self.dropout(x)

        inputStack = []
        for i in range(self.levels):
            x = self.encoders[i](x)
            if i < self.levels - 1:
                inputStack.append(x)

        for i in range(self.levels):
            x = self.decoders[i](x)
            if i < self.levels - 1:
                x = x + inputStack.pop()

        x = self.lastConv(x)

        # if tibo_in_shape != [512,512,256]:
            # x = self.interpolation(x)
        #x = torch.sigmoid(x)
        return x

    @staticmethod
    def apply_argmax_softmax(pred):
        pred = F.softmax(pred, dim=1)
        return pred