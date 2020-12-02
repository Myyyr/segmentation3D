import numpy as numpy
import torch
from alltrain.Train import *
import  torch
from torch.utils.tensorboard import SummaryWriter
import alltrain.atlasUtils as atlasUtils
from multiatlasDataset import *
from torch.utils.data import DataLoader
import os
import torchvision

class LookMAT(Train):

    def __init__(self, expconfig, split = 0):
        super(LookMAT, self).__init__(expconfig)
        self.expconfig = expconfig
        self.tb = SummaryWriter(comment=expconfig.experiment_name+str('_lookimages'))

        trainDataset = MultiAtlasDataset(expconfig, mode="train", randomCrop=None, hasMasks=True, returnOffsets=False, split = split)
        validDataset = MultiAtlasDataset(expconfig, mode="validation", randomCrop=None, hasMasks=True, returnOffsets=False, split = split)
        self.trainDataLoader = DataLoader(dataset=trainDataset, num_workers=1, batch_size=expconfig.batchsize, shuffle=True)
        self.valDataLoader = DataLoader(dataset=validDataset, num_workers=1, batch_size=expconfig.batchsize, shuffle=False)

        self.split = split


    def train(self, start, end, ps = 0):
        expcf = self.expconfig
        expcf.optimizer.zero_grad()
        print("#### EXPERIMENT : {} | ID : {} ####".format(expcf.experiment_name, expcf.id))
        print("#### TRAIN SET :", len(self.trainDataLoader))
        print("#### VALID SET :", len(self.valDataLoader))
        print("#### LOOK THE DATA")

        data = iter(trainDataset)

        for i in range(end):
            if expcf.look_small:
                if i < start:
                    inputs, labels, smalllabels = data.next()
                    _,_,x,y,z = inputs.shape
                    _,c,lx,ly,lz = labels.shape

                    img_grid = torchvision.utils.make_grid(inputs[0,0,int(x//2)+ps,int(y//2)+ps,int(z//2)+ps])
                    self.tb.add_image('image_input'+str(i)+str((int(x//2)+ps,int(y//2)+ps,int(z//2)+p)), img_grid)

                    for k in range(c):
                        img_grid = torchvision.utils.make_grid(labels[0,0,int(x//2)+ps,int(y//2)+ps,int(z//2)+ps])
                        self.tb.add_image('image_labels_'+str(k)+str(i)+str((int(lx//2)+ps,int(ly//2)+ps,int(lz//2)+p)), img_grid)

                        img_grid = torchvision.utils.make_grid(inputs[0,0,int(x//2)+ps,int(y//2)+ps,int(z//2)+ps])
                        self.tb.add_image('image_smalllabel'+str(k)+str(i)+str((int(x//2)+ps,int(y//2)+ps,int(z//2)+p)), img_grid)
                else:
                    _ = data.next()
            else:
                if i < start:
                    inputs, labels = data.next()
                else:
                    _ = data.next()
                


        self.tb.close()



