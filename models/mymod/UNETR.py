import torch.nn as nn
import torch.nn.functional as F
import torch
from models.networks_other import init_weights
from models.mymod.utils import UNetConv3D, UNETRSkip
class UNETR(nn.Module):

    def __init__(self, filters = [64, 128, 256, 512], d_model=768, input_shape= (512,512,512), patch_size=(16,16,16), skip_idx = [3,6,9,12], n_classes=2, in_channels=1, n_heads=8, bn = True, up_mode='deconv', n_layers=12):
        super(UNETR, self).__init__()

        self.in_channels = in_channels
        self.d_model = d_model
        self.n_heads = n_heads
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.filters = filters
        self.filters.reverse()
        self.skip_idx = skip_idx

        self.emb_size_reshape = [int(i/j) for i,j in zip(self.input_shape, self.patch_size)] + [np.prod(self.patch_size)]
        self.emb_size_flat = [np.prod(self.emb_size_reshape[:3]), self.emb_size_reshape[3]]
        
        # Encoders
        self.lin = nn.Linear(self.emb_size_reshape[3], seld.d_model)
        self.ListTrans = []
        for i in range(self.n_layers):
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads)
            self.ListTrans.append(nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers))

        # Skips
        self.skip0 = UNetConv3D(self.in_channels, self.filters[3], bn=bn)
        self.skip1 = UNETRSkip(self.d_model, self.filters[:3], bn=bn)
        self.skip2 = UNETRSkip(self.d_model, self.filters[:2], bn=bn)
        self.skip3 = UNETRSkip(self.d_model, self.filters[:1], bn=bn)


        # Upsamplers
        self.up_concat4 = nn.ConvTranspose3d(self.d_model, self.filters[0])
        self.up_concat3 = nn.ModuleList([UNetConv3D(self.filters[0]*2,self.filters[1], bn=bn), nn.ConvTranspose3d(self.filters[1], self.filters[1])])
        self.up_concat2 = nn.ModuleList([UNetConv3D(self.filters[1]*2,self.filters[2], bn=bn), nn.ConvTranspose3d(self.filters[2], self.filters[2])])
        self.up_concat1 = nn.ModuleList([UNetConv3D(self.filters[2]*2,self.filters[3], bn=bn), nn.ConvTranspose3d(self.filters[3], self.filters[3])])

        # final conv (without any concat)
        self.final = nn.ModuleList([UNetConv3D(self.filters[3]*2,n_classes, bn=bn), nn.Conv3d(n_classes, n_classes, kernel_size=1)])

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')


    def forward(self, X, mode=None):


        sk0 = self.skip0(X)
        sk123 = []

        # Get patches, flat and project
        X = torch.reshape(X, self.emb_size_reshape)
        X = torch.reshape(X, self.emb_size_flat)
        X = self.lin(X)

        # Go through transformers and save reshaped skip
        for i in range(self.n_layers):
            X = self.ListTrans[i](X)
            if i+1 in self.skip_idx:
                sk123.append(torch.reshape(X, self.emb_size_reshape))

        # Decode
        X = self.up_concat4(X)
        X = self.up_concat3(torch.cat([self.skip3(sk123[3]), X],1))
        X = self.up_concat2(torch.cat([self.skip2(sk123[2]), X],1))
        X = self.up_concat1(torch.cat([self.skip1(sk123[1]), X],1))

        # Final
        X = self.final(torch.cat([sk0, X], 1))
        
        return X

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p



def convert_bytes(size):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return "%3.2f %s" % (size, x)
        size /= 1024.0

    return size

