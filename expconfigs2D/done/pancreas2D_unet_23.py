# More Parameters (depth) to match with classical UNet number of parameters.
# n_parameters = 114557582
import os
from models.mymod.UNet import UNet
from models.utils import get_scheduler
import torch.optim as optim
from pancreasCT2DDataset import *
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as tf
# import torchio as tio

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ExpConfig():
    def __init__(self):
        # ID and Name
        self.id = 23
        self.experiment_name = "pancreas_2D_unet_{}".format(self.id)
        self.debug = False

        # System
        self.checkpointsBasePath = "./checkpoints/"
        self.checkpointsBasePathMod = self.checkpointsBasePath + self.experiment_name+'/'
        self.datapath = "/local/DEEPLEARNING/TCIA/"
        self.split = 0
        
        # GPU
        self.gpu = '0'
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu

        # Model
        self.channels = [64, 128, 256, 512, 1024]
        self.channels = [int(x) for x in self.channels]
        self.net = UNet(filters = self.channels, n_classes=2, in_channels=1, dim='2d')
        self.n_parameters = count_parameters(self.net)
        print("N PARAMS : {}".format(self.n_parameters))

        self.model_path = './checkpoints/models/pancreas2D_unet.pth'
        self.load_model()

        self.n_classes = 2
        
        self.transform = tf.Compose([
                            tf.RandomAffine(degrees = 15,
                                            scale = (0.9,1.1),
                                            translate = (0.1, 0.1)),
                            tf.ToTensor(),
                            ])


        # Training
        self.start_epoch = 0
        self.epoch = 300
        self.loss = torch.nn.CrossEntropyLoss()
        self.batchsize = 8
        self.lr_rate = 1e-3
        self.optimizer = optim.Adam(self.net.parameters(), lr = self.lr_rate, weight_decay=5e-6)
        self.optimizer.zero_grad()
        self.validate_every_k_epochs = 1
        # self.lr_scheduler = get_scheduler(self.optimizer, "constant", self.lr_rate)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.96)

        # Other
        self.classes_name = ['background','pancreas']
        
    def set_data(self, split = 0):
        # Data
        trainDataset = SplitTCIA2DDataset(self.datapath, self.split, self.generate_splits('train') , transform = self.transform)
        validDataset = SplitTCIA2DDataset(self.datapath, self.split, self.generate_splits('test'))
        self.trainDataLoader = DataLoader(dataset=trainDataset, num_workers=1, batch_size=self.batchsize, shuffle=True)
        self.valDataLoader = DataLoader(dataset=validDataset, num_workers=1, batch_size=self.batchsize, shuffle=False)

    def generate_splits(self, sets, i = 0):
        splt_1 = [19, 23, 55, 30, 60, 26, 12, 6, 64, 38, 42, 62, 69, 5, 14, 63, 31]
        splt_2 = [46, 59, 81, 61, 41, 15, 49, 43, 48, 32, 11, 37, 53, 17, 47, 21, 2]
        splt_3 = [67, 56, 51, 35, 66, 10, 39, 76, 58, 7, 44, 40, 70, 65, 1, 29]
        splt_4 = [27, 52, 33, 45, 13, 72, 80, 8, 82, 73, 34, 16, 18, 71, 9, 28]
        splt_5 = [57, 22, 68, 78, 74, 54, 20, 24, 4, 79, 36, 75, 25, 50, 3, 77]

        data_splits = {'train':[], 'test':[]}
        all_splits = [splt_1, splt_2, splt_3, splt_4, splt_5]
        
        data_splits['test'] = all_splits[self.split]
        data_splits['train'] = []
        for k in [i for i in all_splits[:self.split]] + all_splits[self.split+1:]:
            data_splits['train'] += k


        return data_splits[sets]

    def load_model(self):
        print('LOAD MODEL ...')
        if not os.path.exists(self.model_path):
            torch.save(self.net.state_dict(), self.model_path)
        else:
            self.net.load_state_dict(torch.load(self.model_path))

    def net_stats(self):
        s = 0
        for p in self.net.parameters():
            if p.requires_grad:
                s += p.sum()

        print('Mean :', s.item()/self.n_parameters)