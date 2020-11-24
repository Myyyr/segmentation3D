import os
from models.revunet_3D import RevUnet3D
from models.utils import get_scheduler
from bratsDataset import BratsDataset
import torch.optim as optim
import alltrain.atlasUtils as atlasUtils

class ExpConfig():
    def __init__(self):
        # ID and Name
        self.experiment_name = "revunet_3D"
        self.id = 2

        # System
        self.checkpointsBasePath = "./models/checkpoints"
        self.labelpath = "/local/SSD_DEEPLEARNING/MULTI_ATLAS/multi_atlas/data_3D_size_256_256_99_res_0.5_0.5.hdf5"
        self.datapath = "/local/SSD_DEEPLEARNING/MULTI_ATLAS/multi_atlas/data_3D_size_256_256_99_res_0.5_0.5.hdf5"
        
        # GPU
        self.gpu = '0'
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu

        # Model
        self.channels = [64, 128, 256, 512, 1024]
        self.channels = [int(x/16) for x in self.channels]
        self.net = RevUnet3D(1, self.channels, 12, interpolation = (256,256,99))

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
            return atlasUtils.atlasDiceLoss(outputs, labels, nonSquared=True)
        self.loss = loss
        self.batchsize = 2
        # self.optimizer = optim.Ada(self.net.parameters(),
        #                       lr= 0.01, #to do
        #                       momentum=0.9,
        #                       nesterov=True,
        #                       weight_decay=1e-5) #todo
        self.optimizer = optim.Adam(self.net.parameters(), lr = 0.0001, weight_decay=1e-5)
        self.validate_every_k_epochs = 1
        # Scheduler list : [lambdarule_1]
        self.lr_scheduler = get_scheduler(self.optimizer, "multistep")


        


        