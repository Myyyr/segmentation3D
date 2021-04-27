# More Parameters (depth) to match with classical UNet number of parameters.
# n_parameters = 114557582
import os
from models.utils import get_scheduler
import torch.optim as optim
import alltrain.atlasUtils as atlasUtils
from pancreasCTDataset import *
from torch.utils.data import DataLoader
import torch
import torchio as tio

from models.cotr.CoTr import ResTranUnet

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ExpConfig():
    def __init__(self):
        # ID and Name
        self.id = 400
        self.experiment_name = "ma_cotr_v{}".format(self.id)
        self.debug = False

        # System
        self.checkpointsBasePath = "./checkpoints/"
        self.checkpointsBasePathMod = self.checkpointsBasePath + 'models/'
        self.labelpath = "/local/DEEPLEARNING/MULTI_ATLAS/multi_atlas/data_3D_size_512_512_256_res_1.hdf5"
        self.datapath = self.labelpath


        self.input_shape = [512,512,256]
        # filters = [4, 8, 16, 32]
        # skip_idx = [1,3,5,6]
        self.patch_size=(192,192,48)
        # n_layers=6
        
        # GPU
        self.gpu = '1'
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu

        # Model
        self.n_classes = 14
        self.net = ResTranUnet(norm_cfg='BN', activation_cfg='ReLU', img_size=self.patch_size, num_classes=self.n_classes, weight_std=False, deep_supervision=False)
        self.n_parameters = count_parameters(self.net)
        print("N PARAMS : {}".format(self.n_parameters))

        self.model_path = './checkpoints/models/cotr.pth'
        # self.model_path = './checkpoints/models/300/mod.pth'
        self.load_model()
         
        
        max_displacement = 5,5,5
        deg = (0,5,10)
        scales = 0
        self.transform = tio.Compose([
            tio.RandomElasticDeformation(max_displacement=max_displacement),
            tio.RandomAffine(scales=scales, degrees=deg)
        ])


        # Training
        self.start_epoch = 0
        self.epoch = 1000

        self.loss = torch.nn.CrossEntropyLoss()

        self.batchsize = 2
        self.lr_rate = 2e-2
        # self.final_lr_rate = 1e-5
        # self.optimizer = optim.Adam(self.net.parameters(), lr = self.lr_rate)
        self.optimizer = optim.SGD(self.net.parameters(), lr = self.lr_rate, weight_decay=3e-5, momentum=0.99)

        self.optimizer.zero_grad()
        self.validate_every_k_epochs = 10
        # self.decay = (self.lr_rate/self.final_lr_rate - 1)/self.epoch
        self.lr_scheduler = get_scheduler(self.optimizer, "poly", self.lr_rate, self.epoch)

        # Other
        self.classes_name = ['background','spleen','right kidney','left kidney','gallbladder','esophagus','liver','stomach','aorta','inferior vena cava','portal vein and splenic vein','pancreas','right adrenal gland','left adrenal gland']
        
    def set_data(self, split = 0):
        # Data
        self.trainDataset = PatchedMultiAtlasDataset(self, mode="train", n_iter=250, patch_size=self.patch_size, return_full_image=False)
        self.testDataset  = PatchedMultiAtlasDataset(self, mode="test", n_iter=1, patch_size=self.patch_size, return_full_image=False)
        self.trainDataLoader = DataLoader(dataset=self.trainDataset, num_workers=1, batch_size=self.batchsize, shuffle=True)
        self.testDataLoader = DataLoader(dataset=self.testDataset, num_workers=1, batch_size=1, shuffle=False)

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