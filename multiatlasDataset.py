import torch
import torch.utils.data
import h5py
import numpy as np
import time
import random
import torchio as tio
# import dataProcessing.augmentation as aug

class MultiAtlasDataset(torch.utils.data.Dataset):
    #mode must be trian, test or val
    def __init__(self, expConfig, mode="train", randomCrop=None, hasMasks=True, returnOffsets=False, split = 0, hot = 0):
        super(MultiAtlasDataset, self).__init__()
        self.filePath = expConfig.datapath
        self.labelPath = expConfig.labelpath
        self.mode = mode
        self.file = None
        self.labelFile = None
        self.trainOriginalClasses = expConfig.train_original_classes
        self.randomCrop = randomCrop
        self.hasMasks = hasMasks
        self.returnOffsets = returnOffsets

        #augmentation settings
        

        self.split = split
        self.hot = hot

        self.transform = expConfig.transform


        self.n_classes = 14

    def __getitem__(self, item_index):

        #lazily open file
        self.openFileIfNotOpen()
        

        index = self.used_split[item_index]

        #load from hdf5 file
        image = self.file["images_" + 'train'][index, ...]
        if self.hasMasks: 
            labels = self.labelFile["masks_" + 'train'][index, ...]
            

        #Prepare data depeinding on soft/hard augmentation scheme
        n_classes = self.n_classes


        if self.transform != None and self.mode=="train":
            sub = tio.Subject(image = tio.ScalarImage(tensor = image[None, :,:,:]), 
                              labels = tio.LabelMap(tensor = labels[None, :,:,:]))
            sub = self.transform(sub)
            image = np.array(sub['image'][0,...])
            labels = np.array(sub['labels'][0,...])

        if 1 == self.hot:
            labels = self._toEvaluationOneHot(labels)
            if self.hasMasks: 
                labels = np.transpose(labels, (3, 0, 1, 2))  # bring into NCWH format

        image = torch.from_numpy(image)
        image = image.expand(1,-1,-1,-1)
        if 1 == 1:
            if self.hasMasks:
                #labels = labels[:, 0:32, 0:32, 0:32]
                labels = torch.from_numpy(labels).long()
                


        
        if self.hasMasks:
            return image, labels
        else:
            return image

    def __len__(self):
        #lazily open file
        self.openFileIfNotOpen()

        return len(self.used_split)

        # return self.file["images_" + self.mode].shape[0]

    def openFileIfNotOpen(self):
        if self.file == None:
            self.file = h5py.File(self.filePath, "r")
        if self.labelFile == None:
            self.labelFile = h5py.File(self.labelPath, "r")

        self.splits = self.myAtlasKFold()
        n = self.file["images_" + 'train'].shape[0]
        nb_splits = 5

        if self.mode == 'train':
            self.used_split = self.splits[:self.split] + self.splits[self.split+1:]
            self.used_split = [j for i in self.used_split for j in i]
            for i in self.splits[self.split]:
                if i in self.used_split:
                    self.used_split.remove(i)
            self.used_split = list(set(self.used_split))
        else:
            self.used_split = self.splits[self.split]


    def myAtlasKFold(self): 
        ind = [[21, 22, 23, 24, 25, 26, 27, 28, 29],
               [12, 13, 14, 15, 16, 17, 18, 19, 20],
               [ 3,  4,  5,  6,  7,  8,  9, 10, 11],
               [ 0,  1,  2,  3, 12, 21,  4, 13, 22],
               [ 1,  5, 14, 23,  2,  6, 15, 24,  7]]

        return ind
        
            
    def _toEvaluationOneHot(self, labels):
        shape = labels.shape
        out = np.zeros([shape[0], shape[1], shape[2], self.n_classes], dtype=np.float32)
        for i in range(self.n_classes):
            out[:,:,:,i] = (labels == i)
        # out[:, :, :, 0] = (labels != 0)
        # out[:, :, :, 1] = (labels != 0) * (labels != 2)
        # out[:, :, :, 2] = (labels == 4)
        return out

    def _toOrignalCategoryOneHot(self, labels):
        shape = labels.shape
        out = np.zeros([shape[0], shape[1], shape[2], self.n_classes], dtype=np.float32)
        for i in range(5):
            out[:, :, :, i] = (labels == i)
        return out

    def _toOrdinal(self, labels):
        return np.argmax(labels, axis=3)
