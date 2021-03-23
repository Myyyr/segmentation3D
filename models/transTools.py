import torch
import torch.nn as nn
import math


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

class trans(nn.Module):
    def __init__(self, d, w, h, projection = "conv"):
        super(trans, self).__init__()

        
        self.n = w*h
        self.w = w
        self.h = h
        self.d = d
        
        self.pe = positionalencoding2d(self.d, self.h, self.w)
        self.projection = projection
        if self.projection == "linear":
            self.wq = nn.Linear(self.n*self.d, self.n*self.d)
            self.wk = nn.Linear(self.n*self.d, self.n*self.d)
            self.wv = nn.Linear(self.n*self.d, self.n*self.d)
        else:
            self.wq = nn.Conv2d(1, 1, (1,1), bias = False)
            self.wk = nn.Conv2d(1, 1, (1,1), bias = False)
            self.wv = nn.Conv2d(1, 1, (1,1), bias = False)
        
    def forward(self, x):
        bs = x.shape[0]
        x = x + self.pe.repeat(bs,1,1,1)
        if self.projection == 'linear':
            x = torch.reshape(x, (bs, self.n * self.d))
        else:
            x = torch.reshape(x, (bs, 1, self.n, self.d))
        
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)
        
        Q = torch.reshape(Q, (bs,self.n,self.d))
        K = torch.reshape(K, (bs,self.n,self.d))
        V = torch.reshape(V, (bs,self.n,self.d))
        
        A = nn.functional.softmax(torch.bmm(Q, torch.reshape(K, (bs,self.d,self.n)))/(self.d**0.5), dim = 2)
        
        Y = torch.bmm(A, V)
        Y = torch.reshape(Y, (bs, self.d, self.h, self.w))
        
        return Y, Q, K, V








