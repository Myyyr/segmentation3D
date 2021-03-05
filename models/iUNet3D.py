# from models.iunets import iUNet
# from models.iunets.layers import create_standard_module
import torch.nn as nn
from iunets import iUNet
class iUNet_3D(nn.Module):
	def __init__(self, in_chan, n_classes, architecture):
		super(iUNet_3D,self).__init__()
		self.first_conv = nn.Conv3d(1,in_chan,1)
		self.iun = iUNet(in_channels=in_chan,
					architecture=architecture, 
					dim=3)
		self.last_conv = nn.Conv3d(in_chan, n_classes, 1)


	def forward(self, x):
		x = self.first_conv(x)
		x = self.iun(x)
		return self.last_conv(x)

