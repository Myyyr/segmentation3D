# More Parameters (depth) to match with classical UNet number of parameters.
# n_parameters = 114557582
import os
from models.utils import get_scheduler
import torch.optim as optim
import alltrain.atlasUtils as atlasUtils
from PatchedMultiatlasDataset_v3 import *
from torch.utils.data import DataLoader
import torch
import torchio as tio

from models.mymod.cross_patch_deep import CrossPatch3DTr
from utils.metrics import DC_and_CE_loss, MultipleOutputLoss2   
from nnunet.utilities.nd_softmax import softmax_helper

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# TRAINING NO CROSS
# BIGGER MODEL

class ExpConfig():
    def __init__(self):
        # ID and Name
        self.id = "506b5"
        self.experiment_name = "ma_crosstr_v{}".format(self.id)
        self.debug = False

        # System
        self.checkpointsBasePath = "./checkpoints/"
        self.checkpointsBasePathMod = self.checkpointsBasePath + 'models/'
        self.labelpath = '/local/DEEPLEARNING/MULTI_ATLAS/MULTI_ATLAS/nnUNet_preprocessed/Task017_BCV/nnUNetData_plans_v2.1_stage1/'
        self.datapath = self.labelpath


        self.input_shape = [512,512,256]
        # self.filters = [16, 32, 64, 128]
        # self.filters = [64, 192, 448, 704]
        # self.filters = [16, 32, 64, 128, 256]
        self.filters = [32, 64, 128, 256, 512]
        d_model = self.filters[-1]

        # skip_idx = [1,3,5,6]
        # self.patch_size=(128,128,128)
        self.patch_size=(192,192,48)
        # n_layers=6
        self.clip = False
        self.patched = True
        # GPU
        self.gpu = '1'
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu
        # torch.backends.cudnn.benchmark = False

        # Model
        number_of_cross_heads = 1
        number_of_self_heads = 8
        number_of_self_layer = 6

        self.n_classes = 14
        self.net = CrossPatch3DTr(filters=self.filters,patch_size=[1,1,1],
                                d_model=d_model,n_classes=self.n_classes,
                                n_cheads=number_of_cross_heads,n_sheads=number_of_self_heads,
                                bn=True,up_mode='deconv',
                                n_strans=number_of_self_layer, do_cross=True)
        self.net.inference_apply_nonlin = softmax_helper
        self.n_parameters = count_parameters(self.net)
        print("N PARAMS : {}".format(self.n_parameters))

        # self.model_path = './checkpoints/models/deep_crosstr.pth'
        self.model_path = './checkpoints/models/506/modlast.pt'
        
         
        
        max_displacement = 5,5,5
        deg = (0,5,10)
        scales = 0
        self.transform = tio.Compose([
            tio.RandomElasticDeformation(max_displacement=max_displacement),
            tio.RandomAffine(scales=scales, degrees=deg)
        ])


        # Training
        self.start_epoch = 1000
        self.epoch = 2000

        # self.loss = torch.nn.CrossEntropyLoss()

        self.loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
        self.ds_scales = ((1, 1, 1), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25), (0.125,0.125,0.125))
        ################# Here we wrap the loss for deep supervision ############
        # we need to know the number of outputs of the network
        net_numpool = 4

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
        weights[~mask] = 0
        weights = weights / weights.sum()
        self.ds_loss_weights = weights
        # now wrap the loss
        self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
        ################# END ###################

        self.batchsize = 2
        self.lr_rate = 1e-2
        # self.final_lr_rate = 1e-5
        # self.optimizer = optim.Adam(self.net.parameters(), lr = self.lr_rate)
        self.optimizer = optim.SGD(self.net.parameters(), lr = self.lr_rate, weight_decay=3e-5, momentum=0.99, nesterov=True)

        self.optimizer.zero_grad()
        self.validate_every_k_epochs = 1
        # self.decay = (self.lr_rate/self.final_lr_rate - 1)/self.epoch
        self.lr_scheduler = get_scheduler(self.optimizer, "poly", self.lr_rate, max_epochs=self.epoch)

        self.load_lr = False
        self.load_model()
        self.net.reinit_decoder()

        # Other
        self.classes_name = ['background','spleen','right kidney','left kidney','gallbladder','esophagus','liver','stomach','aorta','inferior vena cava','portal vein and splenic vein','pancreas','right adrenal gland','left adrenal gland']
        
    def set_data(self, split = 0):
        # Data
        # print(self.ds_scales)s
        self.trainDataset = PatchedMultiAtlasDataset(self, mode="train", n_iter=250, patch_size=self.patch_size, return_full_image=True, ds_scales=self.ds_scales, do_tr=True, return_pos=True)
        self.testDataset  = PatchedMultiAtlasDataset(self, mode="test", n_iter=1, patch_size=self.patch_size, return_full_image=True, ds_scales=None, do_tr=False, return_pos=True)
        self.trainDataLoader = DataLoader(dataset=self.trainDataset, num_workers=2, batch_size=self.batchsize, shuffle=True)
        self.testDataLoader = DataLoader(dataset=self.testDataset, num_workers=2, batch_size=1, shuffle=False)

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
            # self.optimizer.load_state_dict(a['optimizer_state_dict'])
            # self.lr_scheduler.load_state_dict(a['scheduler'])
            if self.load_lr:
                self.lr_scheduler.load_state_dict(a['scheduler'])

    def net_stats(self):
        s = 0
        for p in self.net.parameters():
            if p.requires_grad:
                s += p.sum()

        print('Mean :', s.item()/self.n_parameters)