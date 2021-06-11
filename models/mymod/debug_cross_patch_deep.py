import torch.nn as nn
import torch.nn.functional as F
import torch
from models.mymod.utils import UNetConv2D, UNetConv3D, UnetUp2D, UnetUp3D, PositionalEncodingPermute3D
from models.mymod.transTools import PositionalEncoding, CrossAttention
from models.networks_other import init_weights
import numpy as np
from einops import rearrange



class DebugCrossPatch3DTr(nn.Module):

    def __init__(self, base_model,filters = 512, n_classes=14):
        super(DebugCrossPatch3DTr, self).__init__()

        self.base_model = base_model
        # print(self.base_model)
        self.final_conv = nn.Conv3d(filters, n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')

    def forward(self, X, pos, val=False):
        out = self.base_model(X, pos, val)
        # print(out.shape)
        out = nn.functional.interpolate(out[0], scale_factor=16)
        return [out]



def convert_bytes(size):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return "%3.2f %s" % (size, x)
        size /= 1024.0

    return size

