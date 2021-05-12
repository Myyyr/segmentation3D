import torch
import torch.utils.data
import h5py
import numpy as np
import time
import random
import torchio as tio
import datetime
import os
from utils.util import DownsampleSegForDSTransform2
from utils.transform import TransformData
from utils.util import get_all_crops

SPLITS = [[129,61,128,64,83,88,80,95,99,43,136,81,85,117,125,109,108],
        [107,69,45,100,56,94,49,77,57,53,113,89,116,130,59,110,75],
        [126,86,132,41,76,97,115,51,48,114,123,84,90,134,92,122,91],
        [105,93,87,60,79,118,52,50,124,47,102,68,70,133,71,120,98],
        [106,65,112,74,121,67,127,55,44,73,131,54,58,104,78,82,135]]


class PatchedVPDataset(torch.utils.data.Dataset):
    #mode must be trian, test or val
    def __init__(self, expConfig, mode="train", n_iter=250, patch_size=(192,192,48),n_reg = (4, 3, 3), return_full_image=False, ds_scales=(1, 0.5, 0.25), do_tr=True, return_pos=False, split=0):
        super(PatchedVPDataset, self).__init__()
        self.filePath = os.path.join(expConfig.datapath, "3d_images/")
        self.labelfilePath = os.path.join(expConfig.datapath, "3d_annotations/")
        # self.labelPath = expConfig.labelpath
        self.mode = mode
        self.file = {}
        self.labelfile = {}
        # self.labelFile = None
        self.patch_size = patch_size
        #augmentation settings
        self.transform = expConfig.transform

        self.n_iter = n_iter
        self.return_full_image = return_full_image
        self.n_classes = 8
        self.n_reg = n_reg

        self.return_pos = return_pos

        self.ds = None
        if ds_scales != None:
            self.ds = DownsampleSegForDSTransform2(ds_scales=ds_scales)

        for i in os.listdir(self.filePath):
            if ".npy" in i:
                pid = i.replace('.npy', '').replace('bcv_', '')
                self.file[str(pid)] = os.path.join(self.filePath, i)
        for i in os.listdir(self.labelfilePath):
            if ".npy" in i:
                pid = i.replace('.npy', '').replace('bcv_', '')
                self.labelfile[str(pid)] = os.path.join(self.labelfilePath, i)


        if self.mode == 'train':
            self.used_pids = []
            for l in range(len(SPLITS)):
                if l!=split:
                    self.used_pids += [i for i in l]
        else:
            self.used_pids = SPLITS[split]

        self.n_files = len(self.used_split)

        self.do_tr = do_tr
        if self.do_tr:
            self.tr = TransformData()


    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)


        if self.mode == 'train':
            item_index = int(index%self.n_files)
        else:
            item_index = index
        

        index = self.used_pids[item_index]
        # file = np.load(self.file[str(index)])
        # image = file[0,...]
        # labels = file[1,...]
        image = np.load(self.file[str(index)])
        labels = np.load(self.labelfile[str(index)])
            

        #Prepare data depeinding on soft/hard augmentation scheme
        n_classes = self.n_classes
                  

        if self.mode == 'train':
            w,h,d = image.shape
            ps_h, ps_w, ps_d = self.patch_size

            x = random.randint(0, h- ps_h)
            y = random.randint(0, w- ps_w)
            z = random.randint(0, d- ps_d)

            idx = (x,y,z)

            

            ptc_input = image[x:(x+ps_h),y:(y+ps_w),z:(z+ps_d)]
            labels = labels[x:(x+ps_h),y:(y+ps_w),z:(z+ps_d)]

            #Transform if we have to
            if self.do_tr:
                data = {'data':ptc_input[None, None, ...], 'seg':labels[None, None, ...]}
                data = self.tr(data)
                ptc_input = data['data'][0,0,...]
                labels = data['seg'][0,0,...]
            ptc_input = torch.from_numpy(ptc_input)

            if self.ds != None:
                labels = self.ds(labels)

          # if self.return_full_image:
          #       crop, _, idx_h, idx_w, idx_d = get_all_crops(image, self.patch_size)
          #       #### !!! Check here for pos if needed (maybe not need do to that at all)
          #       # nh, nw, nd = int(h/ps_h), int(w/ps_w), int(d/ps_d)
          #       # crop = []
          #       # pos = [torch.from_numpy(np.array(idx))[None,...]]
          #       # for x in range(nh):
          #       #     for y in range(nw):
          #       #         for z in range(nd):
          #       #             crop.append(image[:,None,x*ps_h:(x+1)*ps_h,y*ps_w:(y+1)*ps_w,z*ps_d:(z+1)*ps_d])
          #       #             pos.append( torch.from_numpy(np.array((x,y,z)))[None,...] )
          #       # crop = torch.cat(crop, dim=1)
          #       # image = torch.reshape(image, (b,nh,nw,nd,ps_h,ps_w,ps_d))
          #       # image = torch.reshape(image, (b,nh*nw*nd,ps_h,ps_w,ps_d))
          #       # print(ptc_input.shape, crop.shape,ptc_input[None,None,...].shape)
          #       # pos = torch.cat(pos, dim=0)
                
          #       # return pos, torch.cat([ptc_input[None,None,...], crop], 1), labels
          #   if self.return_pos:
          #       # nh, nw, nd = int(h/ps_h), int(w/ps_w), int(d/ps_d)
          #       # pos = [torch.from_numpy(np.array(idx))[None,...]]
          #       # for x in range(nh):
          #       #     for y in range(nw):
          #       #         for z in range(nd):
          #       #             pos.append( torch.from_numpy(np.array((x,y,z)))[None,...] )
          #       # pos = torch.cat(pos, dim=0)
          #       pos = torch.from_numpy(np.array(idx))[None,...]
          #       return pos, ptc_input[None, ...], labels
            return ptc_input[None, ...], labels

        if self.mode == 'test':
            pid = torch.from_numpy(np.array([self.used_pids[item_index]]))

            crop, all_counts, _,_,_ = get_all_crops(image, self.patch_size)
            crop = torch.from_numpy(crop)
            all_counts = torch.from_numpy(all_counts)
            # ps_w, ps_h, ps_d = self.patch_size
            # image = torch.from_numpy(image)
            # w,h,d = image.shape
            # if w%ps_w != 0:
            #     print("H, W, D must be multiple of patch size")
            #     exit(0)
            # nh, nw, nd = int(w/ps_w), int(h/ps_h), int(d/ps_d)
            # crop = torch.zeros(*(nh,nw,nd, self.patch_size[0], self.patch_size[1], self.patch_size[2]))
            # pos = []
            # for x in range(nh):
            #     for y in range(nw):
            #         for z in range(nd):
            #             crop[x,y,z,...] = image[x*ps_h:(x+1)*ps_h,y*ps_w:(y+1)*ps_w,z*ps_d:(z+1)*ps_d]
            #             pos.append( torch.from_numpy(np.array((x,y,z)))[None,...] )
            # pos = torch.cat(pos, dim=0)
            # image = torch.cat(crop, dim=1)            
            # image = torch.reshape(image[0, ...], (nh,nw,nd, self.patch_size[0], self.patch_size[1], self.patch_size[2]))
            # if self.return_pos: 
            #     return pid, pos, crop[None, ...], labels
            return pid, crop[None, ...], labels, all_counts[...]


    def __len__(self):
        # self.openFileIfNotOpen()
        if self.mode == 'train':
            return self.n_iter
        return self.n_files
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
