# More Parameters (depth) to match with classical UNet number of parameters.
# n_parameters = 114557582
import os
from models.revunet_3D import RevUnet3D
from models.utils import get_scheduler
import torch.optim as optim
import alltrain.atlasUtils as atlasUtils
from pancreasCTDataset import *
from torch.utils.data import DataLoader
from utils.metrics import SoftDiceLoss

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



class ExpConfig():
    def __init__(self):
        # ID and Name
        self.experiment_name = "tcia_revunet_small_3D_016_split1"
        self.id = 32
        self.debug = False

        # System
        self.checkpointsBasePath = "./checkpoints/"
        self.checkpointsBasePathMod = self.checkpointsBasePath + 'models/'
        self.datapath ="/local/SSD_DEEPLEARNING/PANCREAS_MULTI_RES/80_80_32/"
        self.im_dim = (80,80,32)
        
        # GPU
        self.gpu = '1'
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu

        # Model
        self.n_classes = 2 
        self.channels = [64, 128, 256, 512, 1024]
        self.channels = [int(x//16) for x in self.channels]
        self.net = RevUnet3D(1, self.channels, self.n_classes, depth = 1 ,interpolation = None)#(512,512,198))
        # self.net = RevUnet3D(1, self.channels, 12, interpolation = (256,256,99))
        self.n_parameters = count_parameters(self.net)

        
        
        self.nn_augmentation = False
        self.soft_augmentation = False
        self.do_rotate = False
        self.rot_degrees = 20
        self.do_scale = False
        self.scale_factor = False
        self.do_flip = False
        self.do_elastic_aug = False
        self.sigma = 10
        self.do_intensity_shift = False
        self.max_intensity_shift = 0.1

        self.split = 3

        # Training
        self.train_original_classes = False
        self.epoch = 100
        # def loss(outputs, labels):
        #     return atlasUtils.atlasDiceLoss(outputs, labels, nonSquared=True, n_classe = self.n_classes)
        # self.loss = loss

        self.loss =  SoftDiceLoss(self.n_classes)

        self.batchsize = 1
        # self.optimizer = optim.Ada(self.net.parameters(),
        #                       lr= 0.01, #to do
        #                       momentum=0.9,
        #                       nesterov=True,
        #                       weight_decay=1e-5) #todo
        # self.optimizer = optim.Adam(self.net.parameters(), lr = 5e-4, weight_decay=1e-5)
        self.lr_rate = 5e-3
        self.optimizer = optim.SGD(self.net.parameters(),
                                    lr=self.lr_rate)
        self.optimizer.zero_grad()
        self.validate_every_k_epochs = 1
        # Scheduler list : [lambdarule_1]
        # self.lr_scheduler = get_scheduler(self.optimizer, "multistep")
        # self.lr_scheduler = get_scheduler(self.optimizer, "multistep", self.lr_rate)
        self.lr_scheduler = get_scheduler(self.optimizer, "lambdarule_1", self.lr_rate)

        # Other
        self.classes_name = ['background','pancreas']#,'right kidney','left kidney','gallbladder','esophagus','liver','stomach','aorta','inferior vena cava','portal vein and splenic vein','pancreas','right adrenal gland','left adrenal gland']
        self.look_small = False
        
    def set_data(self, split = 0):
        # Data
        trainDataset = SplitTCIA3DDataset(self.datapath, self.split, self.generate_splits('train'), im_dim=self.im_dim )
        validDataset = SplitTCIA3DDataset(self.datapath, self.split, self.generate_splits('test'), im_dim=self.im_dim )
        self.trainDataLoader = DataLoader(dataset=trainDataset, num_workers=1, batch_size=self.batchsize, shuffle=True)
        self.valDataLoader = DataLoader(dataset=validDataset, num_workers=1, batch_size=self.batchsize, shuffle=False)

    def generate_splits(self, sets, i = 0):
        data_splits = {'train':[], 'test':[]}
        all_splits = ['split_'+str(i+1) for i in range(6)]
        
        data_splits['test'] = [all_splits[self.split]]
        data_splits['train'] = all_splits[:self.split] + all_splits[self.split+1:]

        return data_splits[sets]
    def net_stats(self):
        s = 0
        for p in self.net.parameters():
            if p.requires_grad:
                s += p.sum()

        print('Mean :', s.item()/self.n_parameters)