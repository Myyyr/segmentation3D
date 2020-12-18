import os
from models.unet_3D import unet_3D
from models.utils import get_scheduler
from bratsDataset import BratsDataset
import torch.optim as optim
import alltrain.atlasUtils as atlasUtils

class ExpConfig():
    def __init__(self):
        # ID and Name
        self.experiment_name = "atlas_unet_3D_resolution_01_v2"
        self.id = 7

        # System
        self.checkpointsBasePath = "./checkpoints"
        self.labelpath = "/local/SSD_DEEPLEARNING/MULTI_ATLAS/multi_atlas/data_3D_size_512_512_198_res_1.0_1.0_1.0.hdf5"
        # self.labelpath = "/local/SSD_DEEPLEARNING/MULTI_ATLAS/multi_atlas/data_3D_size_51_51_20_res_0.1.hdf5"
        self.datapath = "/local/SSD_DEEPLEARNING/MULTI_ATLAS/multi_atlas/data_3D_size_51_51_20_res_0.1.hdf5"
        
        # GPU
        self.gpu = '1'
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu

        # Model
        self.channels = [64, 128, 256, 512, 1024]
        self.channels = [int(x/2) for x in self.channels]
        # self.net = unet_3D(self.channels, n_classes=14, is_batchnorm=False, in_channels=1, interpolation = None)#(512, 512, 198))#1, self.channels, 12, interpolation = (512,512,198))
        self.net = unet_3D(self.channels, n_classes=14, is_batchnorm=False, in_channels=1, interpolation = (512, 512, 198))#1, self.channels, 12, interpolation = (512,512,198))

        # Data
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

        # Training
        self.train_original_classes = False
        self.epoch = 1000
        def loss(outputs, labels):
            # return atlasUtils.MyAtlasDiceLoss(outputs, labels)
            return atlasUtils.atlasDiceLoss(outputs, labels)
        self.loss = loss
        self.batchsize = 1
        # self.optimizer = optim.Ada(self.net.parameters(),
        #                       lr= 0.01, #to do
        #                       momentum=0.9,
        #                       nesterov=True,
        #                       weight_decay=1e-5) #todo
        # self.optimizer = optim.Adam(self.net.parameters(), lr = 5e-4, weight_decay=1e-5)

        self.lr_rate = 5e-3
        # self.lr_rate = [0.1, 0.01, 0.0001]
        self.l2_reg_weight = 1e-5

        self.optimizer = optim.SGD(self.net.parameters(),
                                    lr=self.lr_rate)
        self.validate_every_k_epochs = 1
        # Scheduler list : [lambdarule_1]
        # self.lr_scheduler = get_scheduler(self.optimizer, "lambdarule_e1000", self.lr_rate)
        self.lr_scheduler = get_scheduler(self.optimizer, "multistep", self.lr_rate)

        # Other
        self.classes_name = ['background','spleen','right kidney','left kidney','gallbladder','esophagus','liver','stomach','aorta','inferior vena cava','portal vein and splenic vein','pancreas','right adrenal gland','left adrenal gland']


        self.look_small = False


        