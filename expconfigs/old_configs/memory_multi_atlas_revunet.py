import os
from models.memory_unet_3D import memory_unet_3D
from models.utils import get_scheduler
from bratsDataset import BratsDataset
import torch.optim as optim
import alltrain.atlasUtils as atlasUtils

class ExpConfig():
    def __init__(self):
        # ID and Name
        self.experiment_name = "memory_unet_3D"
        self.id = 3

        # System
        self.checkpointsBasePath = "./checkpoints/"
        self.labelpath = "/local/SSD_DEEPLEARNING/MULTI_ATLAS/multi_atlas/data_3D_size_512_512_198_res_1.0_1.0_1.0.hdf5"
        self.datapath = "/local/SSD_DEEPLEARNING/MULTI_ATLAS/multi_atlas/data_3D_size_51_51_19_res_0.1.hdf5"
        # self.labelpath = "/local/SSD_DEEPLEARNING/MULTI_ATLAS/multi_atlas/data_3D_size_256_256_99_res_0.5_0.5.hdf5"
        # GPU
        self.gpu = '1'
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu

        # Model
        self.channels = [64, 128, 256, 512, 1024]
        self.channels = [int(x/1) for x in self.channels]
        # self.net = RevUnet3D(1, self.channels, 12, interpolation = (512,512,198))
        self.net = memory_unet_3D(self.channels, is_batchnorm=False, n_classes=13, in_channels=1, interpolation = (512, 512, 198))#1, self.channels, 12, interpolation = (512,512,198))
        # self.net = RevUnet3D(1, self.channels, 12, interpolation = (256,256,99))

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
        self.epoch = 2
        def loss(outputs, labels):
            return atlasUtils.atlasDiceLoss(outputs, labels, nonSquared=True)
        self.loss = loss
        self.batchsize = 1
        self.lr_rate = 5e-4
        self.l2_reg_weight = 1e-5
        # self.optimizer = optim.Ada(self.net.parameters(),
        #                       lr= 0.01, #to do
        #                       momentum=0.9,
        #                       nesterov=True,
        #                       weight_decay=1e-5) #todo
        # self.optimizer = optim.Adam(self.net.parameters(), lr = 0.0001, weight_decay=1e-5)
        self.optimizer = optim.SGD(self.net.parameters(),
                                    lr=self.lr_rate,
                                    momentum=0.9,
                                    nesterov=True,
                                    weight_decay=self.l2_reg_weight)
        self.optimizer.zero_grad()
        self.validate_every_k_epochs = 1
        # Scheduler list : [lambdarule_1]
        self.lr_scheduler = get_scheduler(self.optimizer, "multistep")

        # Other
        self.classes_name = ['spleen','right kidney','left kidney','gallbladder','esophagus','liver','stomach','aorta','inferior vena cava','portal vein and splenic vein','pancreas','right adrenal gland','left adrenal gland']
        self.look_small = False
        


        