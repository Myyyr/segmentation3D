import torch
import torch.utils.data
import h5py
import numpy as np
import time
import random
import torchio as tio
# import dataProcessing.augmentation as aug

class PatchedMultiAtlasDataset(torch.utils.data.Dataset):
    #mode must be trian, test or val
    def __init__(self, expConfig, mode="train", n_iter=250, return_full_image=False):
        super(PatchedMultiAtlasDataset, self).__init__()
        self.filePath = expConfig.datapath
        self.labelPath = expConfig.labelpath
        self.mode = mode
        self.file = None
        self.labelFile = None

        #augmentation settings
        self.transform = expConfig.transform

        self.n_iter = n_iter
        self.n_files = None
        self.return_full_image = return_full_image


        self.n_classes = 14

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        #lazily open file
        self.openFileIfNotOpen()

        item_index = int(index%self.n_files)
        
        index = self.used_split[item_index]

        #load from hdf5 file
        image = self.file["images_" + 'train'][index, ...]
        labels = self.labelFile["masks_" + 'train'][index, ...]
            

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
            b, w,h,d = input.shape
            ps_h, ps_w, ps_d = self.patch_size

            x = random.randint(0, h- ps_h)
            y = random.randint(0, w- ps_w)
            z = random.randint(0, d- ps_d)

            

            input = input[:,x:(x+ps_h),y:(y+ps_w),z:(z+ps_d)]
            target = target[x:(x+ps_h),y:(y+ps_w),z:(z+ps_d)]

            input = torch.reshape(input, (ps_w, ps_h, ps_d))
            return input, target

        if self.mode == 'test':
            pid = torch.from_numpy(np.array([self.data_splits[index]]))


            ps_w, ps_h, ps_d = self.patch_size
            b, w,h,d = input.shape
            if w%ps_w != 0:
                print("H, W, D must be multiple of patch size")
                exit(0)
            nw, nh, nd = int(w/ps_w), int(h/ps_h), int(d/ps_d)
            input = torch.reshape(input[0, ...], (nw,nh,nd, self.patch_size[0], self.patch_size[1], self.patch_size[2]))
            # input = input.permute((0,1,2,,4,5,6))

            #target = torch.reshape(target, (n**3, int(w/ps_w), int(d/ps_d), int(h/ps_h)))

            return pid, input, target


    def __len__(self):
        return self.n_iter

        # return self.file["images_" + self.mode].shape[0]

    def openFileIfNotOpen(self):
        if self.file == None:
            self.file = h5py.File(self.filePath, "r")
        if self.labelFile == None:
            self.labelFile = h5py.File(self.labelPath, "r")

        if self.mode == 'train':
            # self.used_split = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
            self.used_split = [6,17, 3,16,22,27,10,28, 7, 29, 23, 13,  1,  9,  5, 15, 11, 12, 24, 14,  8]

        else:
            # self.used_split = [32, 33, 34, 35, 36, 37, 38, 39, 40]
            self.used_split = [18, 20, 25, 19, 21,  0,  4,  2, 26]
        self.n_files = len(self.used_split)
            
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
