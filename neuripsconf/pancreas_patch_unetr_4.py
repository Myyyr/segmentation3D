# More Parameters (depth) to match with classical UNet number of parameters.
# n_parameters = 114557582
import os
from models.utils import get_scheduler
import torch.optim as optim
import alltrain.atlasUtils as atlasUtils
from Patched3DPancreasCTDataset import *
from torch.utils.data import DataLoader
import torch
import torchio as tio

from models.mymod.UNETR import UNETR

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ExpConfig():
    def __init__(self):
        # ID and Name
        self.id = 304
        self.experiment_name = "pancreas_unetr_v{}".format(self.id)
        self.debug = False

        # System
        self.checkpointsBasePath = "./checkpoints/"
        self.checkpointsBasePathMod = self.checkpointsBasePath + 'models/'
        self.labelpath = "/local/DEEPLEARNING/PANCREAS_MULTI_RES/512_512_256/"
        self.datapath = self.labelpath


        self.input_shape = [512,512,256]
        filters = [64, 128, 256, 512]
        skip_idx = [3,6,9,12]
        patch_size=(16,16,16)
        self.data_patch = [128,128,128]
        n_layers=12
        self.patched = True
        
        # GPU
        self.gpu = '0'
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu

        # Model
        self.n_classes = 2
        self.net = UNETR(input_shape=self.data_patch,filters=filters,patch_size=patch_size, n_layers=n_layers, skip_idx=skip_idx)
        self.n_parameters = count_parameters(self.net)
        print("N PARAMS : {}".format(self.n_parameters))

        
         
        
        max_displacement = 5,5,5
        deg = (0,5,10)
        scales = 0
        self.transform = tio.Compose([
            tio.RandomElasticDeformation(max_displacement=max_displacement),
            tio.RandomAffine(scales=scales, degrees=deg)
        ])


        # Training
        self.start_epoch = 0
        self.train_original_classes = False
        self.epoch = 100

        # self.model_path = './checkpoints/models/unetr.pth'
        self.model_path = './checkpoints/models/unetr.pth'
        self.load_model()
        self.split = 1

        self.loss = torch.nn.CrossEntropyLoss()

        self.hot = 0
        self.batchsize = 1
        self.lr_rate = 1e-3
        self.final_lr_rate = 1e-4
        self.optimizer = optim.Adam(self.net.parameters(), lr = self.lr_rate)

        self.optimizer.zero_grad()
        self.validate_every_k_epochs = 1
        self.decay = (self.lr_rate/self.final_lr_rate - 1)/self.epoch
        self.lr_scheduler = get_scheduler(self.optimizer, "constant", self.lr_rate, self.decay)

        # Other
        self.classes_name = ['background','pancreas']
        self.look_small = False
        
    def set_data(self, split = 0):
        # Data
        self.trainDataset = Patched3DSplitTCIA3DDataset(self.datapath, self.split, self.generate_splits('train'), self.data_patch , transform = self.transform)
        self.validDataset = Patched3DSplitTCIA3DDataset(self.datapath, self.split, self.generate_splits('test'), self.data_patch )
        self.testDataset  = Patched3DSplitTCIA3DDataset(self.datapath, self.split, self.generate_splits('test'), self.data_patch , mode = 'test')
        self.trainDataLoader = DataLoader(dataset=self.trainDataset, num_workers=1, batch_size=self.batchsize, shuffle=True)
        self.valDataLoader = DataLoader(dataset=self.validDataset, num_workers=1, batch_size=self.batchsize, shuffle=False)
        self.testDataLoader = DataLoader(dataset=self.testDataset, num_workers=1, batch_size=1, shuffle=False)


    def generate_splits(self, sets, i = 0):
        data_splits = {'train':[], 'test':[]}
        all_splits = ['split_'+str(i+1) for i in range(6)]
        
        data_splits['test'] = [all_splits[self.split]]
        data_splits['train'] = all_splits[:self.split] + all_splits[self.split+1:]
        print(sets, data_splits[sets])
        return data_splits[sets]

    def load_model(self):
        print('LOAD MODEL ...')
        if not os.path.exists(self.model_path):
            torch.save(self.net.state_dict(), self.model_path)
        elif self.start_epoch == 0:
            self.net.load_state_dict(torch.load(self.model_path))
        else:
            a = torch.load(self.model_path)
            self.net.load_state_dict(a['net_state_dict'])
            # self.optimizer = optim.Adam(self.net.parameters(), lr = self.lr_rate, weight_decay=0)
            self.optimizer.load_state_dict(a['optimizer_state_dict'])

    def net_stats(self):
        s = 0
        for p in self.net.parameters():
            if p.requires_grad:
                s += p.sum()

        print('Mean :', s.item()/self.n_parameters)