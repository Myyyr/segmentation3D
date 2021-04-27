import torch
import torch.utils.data
import h5py
import numpy as np
import time
import random
import torchio as tio
import datetime
import os
# import dataProcessing.augmentation as aug

class PatchedMultiAtlasDataset(torch.utils.data.Dataset):
    #mode must be trian, test or val
    def __init__(self, expConfig, mode="train", n_iter=250, patch_size=(192,192,48), return_full_image=False):
        super(PatchedMultiAtlasDataset, self).__init__()
        self.filePath = expConfig.datapath
        # self.labelPath = expConfig.labelpath
        self.mode = mode
        self.file = {}
        # self.labelFile = None
        self.patch_size = patch_size
        #augmentation settings
        self.transform = expConfig.transform

        self.n_iter = n_iter
        self.return_full_image = return_full_image
        self.n_classes = 14

        for i in os.lisdir(self.filePath):
            pid = i.split('/')[-1].replace('.npy', '').replace('000', '').replace('00', '')
            self.file[str(pid)] = i


        if self.mode == 'train':
            self.used_pids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
            self.used_split = [6,17, 3,16,22,27,10,28, 7, 29, 23, 13,  1,  9,  5, 15, 11, 12, 24, 14,  8]

        else:
            self.used_pids = [32, 33, 34, 35, 36, 37, 38, 39, 40]
            self.used_split = [18, 20, 25, 19, 21,  0,  4,  2, 26]
        self.n_files = len(self.used_split)


    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        #lazily open file
        # self.openFileIfNotOpen()

        if self.mode == 'train':
            item_index = int(index%self.n_files)
        else:
            item_index = index
        
        # print(item_index)
        # print(len(self.used_split))
        index = self.used_pids[item_index]

        #load from hdf5 file
        file = np.load(self.file[str(index)])
        image = file[0,...]
        labels = file[1,...]
            

        #Prepare data depeinding on soft/hard augmentation scheme
        n_classes = self.n_classes


        if self.transform != None and self.mode=="train":
            sub = tio.Subject(image = tio.ScalarImage(tensor = image[None, :,:,:]), 
                              labels = tio.LabelMap(tensor = labels[None, :,:,:]))
            sub = self.transform(sub)
            image = np.array(sub['image'])[0,...]
            labels = np.array(sub['labels'])[0,...]

        image = torch.from_numpy(image)
        image = image.expand(1,-1,-1,-1)
        labels = torch.from_numpy(labels).long()
                  

        if self.mode == 'train':
            b, w,h,d = image.shape
            ps_h, ps_w, ps_d = self.patch_size

            x = random.randint(0, h- ps_h)
            y = random.randint(0, w- ps_w)
            z = random.randint(0, d- ps_d)

            

            ptc_input = image[:,x:(x+ps_h),y:(y+ps_w),z:(z+ps_d)]
            labels = labels[x:(x+ps_h),y:(y+ps_w),z:(z+ps_d)]

            ptc_input = torch.reshape(ptc_input, (ps_w, ps_h, ps_d))
            if self.return_full_image:
                return (ptc_input, image), labels
            return ptc_input[None, ...], labels

        if self.mode == 'test':
            pid = torch.from_numpy(np.array([self.used_pids[item_index]]))


            ps_w, ps_h, ps_d = self.patch_size
            b, w,h,d = image.shape
            if w%ps_w != 0:
                print("H, W, D must be multiple of patch size")
                exit(0)
            nw, nh, nd = int(w/ps_w), int(h/ps_h), int(d/ps_d)
            image = torch.reshape(image[0, ...], (nw,nh,nd, self.patch_size[0], self.patch_size[1], self.patch_size[2]))
            return pid, image[None, ...], target


    def __len__(self):
        self.openFileIfNotOpen()
        if self.mode == 'train':
            return self.n_iter
        return len(self.n_files)
        # return self.file["images_" + self.mode].shape[0]

    # def openFileIfNotOpen(self):
    #     if self.file == None:
    #         self.file = h5py.File(self.filePath, "r")
    #     if self.labelFile == None:
    #         self.labelFile = h5py.File(self.labelPath, "r")

    #     if self.mode == 'train':
    #         self.used_pids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    #         self.used_split = [6,17, 3,16,22,27,10,28, 7, 29, 23, 13,  1,  9,  5, 15, 11, 12, 24, 14,  8]

    #     else:
    #         self.used_pids = [32, 33, 34, 35, 36, 37, 38, 39, 40]
            # self.used_split = [18, 20, 25, 19, 21,  0,  4,  2, 26]
        # self.n_files = len(self.used_split)
            
    def _toEvaluationOneHot(self, labels):
        shape = labels.shape
        out = np.zeros([shape[0], shape[1], shape[2], self.n_classes], dtype=np.float32)
        for i in range(self.n_classes):
            out[:,:,:,i] = (labels == i)
        return out

    def _toOrignalCategoryOneHot(self, labels):
        shape = labels.shape
        out = np.zeros([shape[0], shape[1], shape[2], self.n_classes], dtype=np.float32)
        for i in range(5):
            out[:, :, :, i] = (labels == i)
        return out

    def _toOrdinal(self, labels):
        return np.argmax(labels, axis=3)
