# More Parameters (depth) to match with classical UNet number of parameters.
# n_parameters = 114557582
import os
from models.revunet_3D import RevUnet3D
from models.utils import get_scheduler
import torch.optim as optim
import alltrain.atlasUtils as atlasUtils
from pancreasCTDataset import *
from torch.utils.data import DataLoader
import torch
import torchio as tio

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ExpConfig():
    def __init__(self):
        # ID and Name
        self.id = 200
        self.experiment_name = "tcia_revunet_03_d3_e1000_CE_adam_wd6_da_id{}".format(self.id)
        self.debug = False

        # System
        self.checkpointsBasePath = "./checkpoints/"
        self.checkpointsBasePathMod = self.checkpointsBasePath + 'models/'
        self.labelpath = "/local/SSD_DEEPLEARNING/MULTI_ATLAS/PANCREAS_MULTI_RES/160_160_64/"
        self.datapath = self.labelpath
        
        # GPU
        self.gpu = '0'
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu

        # Model
        self.channels = [64, 128, 256, 512, 1024]
        self.channels = [int(x) for x in self.channels]
        self.net = RevUnet3D(1, self.channels, 14, depth = 3 ,interpolation = None)#(512,512,198))
        # self.net = RevUnet3D(1, self.channels, 12, interpolation = (256,256,99))
        self.n_parameters = count_parameters(self.net)
        print("N PARAMS : {}".format(self.n_parameters))

        self.model_path = './checkpoints/models/revunet_tcia_160_160_64_d3.pth'
        self.load_model()

        self.n_classes = 14 
        
        self.n_classes = 14
        max_displacement = 5,5,5
        deg = (0,5,10)
        scales = 0
        self.transform = tio.Compose([
            tio.RandomElasticDeformation(max_displacement=max_displacement),
            tio.RandomAffine(scales=scales, degrees=deg)
        ])


        # Training
        self.train_original_classes = False
        self.epoch = 1000
        # def loss(outputs, labels):
        #     return atlasUtils.atlasDiceLoss(outputs, labels, n_classe = self.n_classes)
        # self.loss = loss
        self.loss = torch.nn.CrossEntropyLoss()

        self.hot = 0
        # self.loss =  SoftDiceLoss(self.n_classes)

        self.batchsize = 1
        # self.optimizer = optim.Ada(self.net.parameters(),
        #                       lr= 0.01, #to do
        #                       momentum=0.9,
        #                       nesterov=True,
        #                       weight_decay=1e-5) #todo
        # self.optimizer = optim.Adam(self.net.parameters(), lr = 5e-4, weight_decay=1e-5)
        self.lr_rate = 5e-5
        # self.optimizer = optim.SGD(self.net.parameters(),
        #                             lr=self.lr_rate)
        self.optimizer = optim.Adam(self.net.parameters(), lr = self.lr_rate, weight_decay=1e-6)


        # self.optimizer = optim.SGD(self.net.parameters(),
        #                           lr=self.lr_rate,
        #                           momentum=0.9,
        #                           nesterov=True,
        #                           weight_decay=5e-4)
        self.optimizer.zero_grad()
        self.validate_every_k_epochs = 1
        # Scheduler list : [lambdarule_1]
        # self.lr_scheduler = get_scheduler(self.optimizer, "multistep")
        self.lr_scheduler = get_scheduler(self.optimizer, "multistep", self.lr_rate)
        # self.lr_scheduler = get_scheduler(self.optimizer, "lambdarule_1", self.lr_rate)

        # Other
        self.classes_name = ['background','pancreas']
        self.look_small = False
        
    def set_data(self, split = 0):
        # Data
        trainDataset = SplitTCIA3DDataset(self.datapath, self.split, self.generate_splits('train'), im_dim=self.im_dim , transform = self.transform)
        validDataset = SplitTCIA3DDataset(self.datapath, self.split, self.generate_splits('test'), im_dim=self.im_dim )
        self.trainDataLoader = DataLoader(dataset=trainDataset, num_workers=1, batch_size=self.batchsize, shuffle=True)
        self.valDataLoader = DataLoader(dataset=validDataset, num_workers=1, batch_size=self.batchsize, shuffle=False)


    def generate_splits(self, sets, i = 0):
        data_splits = {'train':[], 'test':[]}
        all_splits = ['split_'+str(i+1) for i in range(6)]
        
        data_splits['test'] = [all_splits[self.split]]
        data_splits['train'] = all_splits[:self.split] + all_splits[self.split+1:]


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