import os
from models.revunet_3D import RevUnet3D
from models.utils import get_scheduler
from bratsDataset import BratsDataset
import torch.optim as optim
import alltrain.bratsUtils as bratsUtils

class ExpConfig():
    def __init__(self):
        # ID and Name
        self.name = "Example"
        self.id = 0

        # System
        self.checkpointsBasePathSave = "path"
        self.datapath = ""

        # GPU
        self.gpu = '0'
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu

        # Model
        self.channels = [64, 128, 256, 512, 1024]
        self.channels = [int(x/16) for x in self.channels]
        self.net = RevUnet3D(channels)

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
        self.epoch = 1
        def loss(outputs, labels):
            return bratsUtils.bratsDiceLoss(outputs, labels, nonSquared=True)
        self.loss = loss
        self.batchsize = None
        self.optimizer = optim.SGD(params,
                              lr= 0.01, #to do
                              momentum=0.9,
                              nesterov=True,
                              weight_decay=1e-5) #todo
        self.validate_every_k_epochs = 1
        # Scheduler list : [lambdarule_1]
        self.lr_scheduler = get_scheduler(self.optimizer, "lambdarule_1")


        


        