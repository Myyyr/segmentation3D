# More Parameters (depth) to match with classical UNet number of parameters.
# n_parameters = 114557582
import os
from models.revunet_3D import RevUnet3D
from models.utils import get_scheduler
import torch.optim as optim
import alltrain.atlasUtils as atlasUtils
from multiatlasDataset import *
from torch.utils.data import DataLoader
from utils.metrics import SoftDiceLoss

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ExpConfig():
    def __init__(self):
        # ID and Name
        self.id = 701
        self.experiment_name = "multi_atlas_revunet_1_d3_e2000_CE_adam_wd0_bs1_da_lr5_gr0_id{}".format(self.id)
        self.debug = False

        # System
        self.checkpointsBasePath = "./checkpoints/"
        self.checkpointsBasePathMod = self.checkpointsBasePath + 'models/'
#        self.labelpath = "/local/SSD_DEEPLEARNING/MULTI_ATLAS/multi_atlas/data_3D_size_512_512_198_res_1.0_1.0_1.0.hdf5"
        # self.labelpath = "/local/SSD_DEEPLEARNING/MULTI_ATLAS/multi_atlas/data_3D_size_256_256_99_res_0.5_0.5.hdf5"
        self.labelpath = "/local/DEEPLEARNING/MULTI_ATLAS/multi_atlas/data_3D_size_512_512_208_res_1.hdf5"
        self.datapath = self.labelpath
        
        # GPU
        self.gpu = '1'
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu

        # Model
        # self.channels = [64, 128, 256, 512, 1024]
        self.channels = [2, 2, 2, 2]
        self.channels = [int(x//1) for x in self.channels]
        self.net = RevUnet3D(1, self.channels, 14, depth = 1 ,interpolation = None, groups=None)#(512,512,198))
        # self.net = RevUnet3D(1, self.channels, 12, interpolation = (256,256,99))
        self.n_parameters = count_parameters(self.net)
        print("N PARAMS : {}".format(self.n_parameters))

        self.model_path = './checkpoints/models/revunet_atlas_512_512_208_d3_fp2_gr0.pth'
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
        self.start_epoch = 0
        self.epoch = 2000
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
        self.optimizer = optim.Adam(self.net.parameters(), lr = self.lr_rate, weight_decay=0)


        # self.optimizer = optim.SGD(self.net.parameters(),
        #                           lr=self.lr_rate,
        #                           momentum=0.9,
        #                           nesterov=True,
        #                           weight_decay=5e-4)
        self.optimizer.zero_grad()
        self.validate_every_k_epochs = 1
        # Scheduler list : [lambdarule_1]
        # self.lr_scheduler = get_scheduler(self.optimizer, "multistep")
        self.lr_scheduler = get_scheduler(self.optimizer, "constant", self.lr_rate)
        # self.lr_scheduler = get_scheduler(self.optimizer, "lambdarule_1", self.lr_rate)

        # Other
        self.classes_name = ['background','spleen','right kidney','left kidney','gallbladder','esophagus','liver','stomach','aorta','inferior vena cava','portal vein and splenic vein','pancreas','right adrenal gland','left adrenal gland']
        
    def set_data(self, split = 0):
        # Data
        self.experiment_name += "_split_{}".format(split)
        trainDataset = MultiAtlasDataset(self, mode="train", randomCrop=None, hasMasks=True, returnOffsets=False, split = split, hot = self.hot)
        validDataset = MultiAtlasDataset(self, mode="validation", randomCrop=None, hasMasks=True, returnOffsets=False, split = split, hot = self.hot)
        self.trainDataLoader = DataLoader(dataset=trainDataset, num_workers=1, batch_size=self.batchsize, shuffle=True)
        self.valDataLoader = DataLoader(dataset=validDataset, num_workers=1, batch_size=self.batchsize, shuffle=False)

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