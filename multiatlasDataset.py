import torch
import torch.utils.data
import h5py
import numpy as np
import time
import random
# import dataProcessing.augmentation as aug

class MultiAtlasDataset(torch.utils.data.Dataset):
    #mode must be trian, test or val
    def __init__(self, expConfig, mode="train", randomCrop=None, hasMasks=True, returnOffsets=False, split = 0):
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
        self.nnAugmentation = expConfig.nn_augmentation
        self.softAugmentation = expConfig.soft_augmentation
        self.doRotate = expConfig.do_rotate
        self.rotDegrees =  expConfig.rot_degrees
        self.doScale = expConfig.do_scale
        self.scaleFactor = expConfig.scale_factor
        self.doFlip = expConfig.do_flip
        self.doElasticAug = expConfig.do_elastic_aug
        self.sigma = expConfig.sigma
        self.doIntensityShift = expConfig.do_intensity_shift
        self.maxIntensityShift = expConfig.max_intensity_shift

        self.look_small = expConfig.look_small
        self.split = split


        self.n_classes = 12

    def __getitem__(self, item_index):

        #lazily open file
        self.openFileIfNotOpen()
        

        index = self.used_split[item_index]

        #load from hdf5 file
        image = self.file["images_" + 'train'][index, ...]
        if self.hasMasks: 
            labels = self.labelFile["masks_" + 'train'][index, ...]
            print( 'np.sum(labels)' ,np.sum(labels))
            exit(0)
        if self.look_small: 
            smalllabels = self.file["masks_" + 'train'][index, ...]
        else:
            smalllabels = None

        #Prepare data depeinding on soft/hard augmentation scheme
        n_classes = 12
        if not self.nnAugmentation:
            if not self.trainOriginalClasses and (self.mode != "train" or self.softAugmentation):
                if self.hasMasks: 
                    labels = self._toEvaluationOneHot(labels)
                    if self.look_small:
                        smalllabels = self._toEvaluationOneHot(smalllabels)
                defaultLabelValues = np.zeros(n_classes, dtype=np.float32)
            else:
                if self.hasMasks: 
                    labels = self._toOrignalCategoryOneHot(labels)
                    if self.look_small:
                        smalllabels = self._toOrignalCategoryOneHot(smalllabels)
        elif self.hasMasks:
            if labels.ndim < n_classes+1:
                labels = np.expand_dims(labels, n_classes)
                if self.look_small:
                    smalllabels = np.expand_dims(smalllabels, n_classes)



        if self.nnAugmentation:
            if self.hasMasks: 
                labels = self._toEvaluationOneHot(np.squeeze(labels, 3))
                if self.look_small:
                    smalllabels = self._toEvaluationOneHot(np.squeeze(smalllabels, 3))
        else:
            if self.mode == "train" and not self.softAugmentation and not self.trainOriginalClasses and self.hasMasks:
                labels = self._toOrdinal(labels)
                labels = self._toEvaluationOneHot(labels)
                if self.look_small:
                    smalllabels = self._toOrdinal(smalllabels)
                    smalllabels = self._toEvaluationOneHot(smalllabels)

        if self.hasMasks: 
            labels = np.transpose(labels, (3, 0, 1, 2))  # bring into NCWH format
            if self.look_small:
                smalllabels = np.transpose(smalllabels, (3, 0, 1, 2))  # bring into NCWH format

        # to tensor
        #image = image[:, 0:32, 0:32, 0:32]

        image = torch.from_numpy(image)
        image = image.expand(1,-1,-1,-1)
        if self.hasMasks:
            #labels = labels[:, 0:32, 0:32, 0:32]
            labels = torch.from_numpy(labels) 
            if self.look_small:
                smalllabels = torch.from_numpy(smalllabels) 

        #get pid
        # pid = self.file["pids_" + self.mode][index]


        
        if self.hasMasks and self.look_small:
            return image, labels, smalllabels
        elif self.hasMasks:
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
