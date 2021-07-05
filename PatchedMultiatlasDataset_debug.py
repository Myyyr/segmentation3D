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
# import dataProcessing.augmentation as aug
# from batchgenerators.dataloading.data_loader import DataLoaderBase
from utils.transform import TransformData


class PatchedMultiAtlasDataset(torch.utils.data.Dataset):
    #mode must be trian, test or val
    def __init__(self, expConfig, mode="train", n_iter=250, patch_size=(192,192,48),n_reg = (4, 3, 3), return_full_image=False, ds_scales=(1, 0.5, 0.25), do_tr=True, return_pos=True):
        super(PatchedMultiAtlasDataset, self).__init__()
        self.l_filePath = expConfig.datapath
        self.d_filePath = expConfig.labelpath

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
        self.n_reg = n_reg

        self.return_pos = return_pos

        self.ds = None
        if ds_scales != None:
            self.ds = DownsampleSegForDSTransform2(ds_scales=ds_scales)

        for i in os.listdir(self.l_filePath):
            if ".npy" in i:
                pid = i.replace('.npy', '').replace('bcv_', '')
                self.file[str(pid)] = [os.path.join(self.l_filePath, i)]
        for i in os.listdir(self.d_filePath):
            if ".npy" in i:
                pid = i.replace('.npy', '')
                self.file[str(pid)] += [os.path.join(self.d_filePath, i)]



        if self.mode == 'train':
            self.used_pids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
            self.used_split = [6,17, 3,16,22,27,10,28, 7, 29, 23, 13,  1,  9,  5, 15, 11, 12, 24, 14,  8]

        else:
            self.used_pids = [32, 33, 34, 35, 36, 37, 38, 39, 40]
            self.used_split = [18, 20, 25, 19, 21,  0,  4,  2, 26]
        self.n_files = len(self.used_split)

        self.do_tr = do_tr
        if self.do_tr:
            self.tr = TransformData()

    def pad_or_crop_image(self, x):
        # Get shapes
        ps = [self.patch_size[-1], self.patch_size[0], self.patch_size[1]] 
        n_reg = self.n_reg
        fs = (ps[0]*n_reg[0], ps[1]*n_reg[1], ps[2]*n_reg[2])
        _,d,u,v = x.shape 

        # Pad
        pad_d = (0,0)
        pad_u = (0,0)
        pad_v = (0,0)
        if d<fs[0] : pad_d = ((fs[0]-d)//2, (fs[0]-d)//2 + (fs[0]-d)%2)
        if u<fs[1] : pad_u = ((fs[1]-u)//2, (fs[1]-u)//2 + (fs[1]-u)%2)
        if v<fs[2] : pad_v = ((fs[2]-v)//2, (fs[2]-v)//2 + (fs[2]-v)%2)
        # pad_x = np.pad(x, ((0,0),pad_d,pad_u,pad_v), 'minimum')
        pad_x = []
        pad_x += [np.pad(x[0,...], (pad_d,pad_u,pad_v), 'minimum')[None,...]]
        pad_x += [np.pad(x[1,...], (pad_d,pad_u,pad_v), constant_values=0)[None,...]]
        pad_x = np.concatenate(pad_x, axis=0)
        pad_x[pad_x == -1] = 0 
        
        # Crop
        cd,cu,cv = pad_x.shape[1]//2,pad_x.shape[2]//2,pad_x.shape[3]//2
        crx = pad_x[:,(cd-int(ps[0]*n_reg[0]/2)):(cd+int(ps[0]*n_reg[0]/2)),
                            (cu-int(ps[1]*n_reg[1]/2)):(cu+int(ps[1]*n_reg[1]/2)),
                            (cv-int(ps[2]*n_reg[2]/2)):(cv+int(ps[2]*n_reg[2]/2))]
        
        return np.transpose(crx, axes=(0,2,3,1))
    def __getitem__(self, index):
        # print(index)
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        #lazily open file
        # self.openFileIfNotOpen()

        if self.mode == 'train':
            item_index = int(index%self.n_files)
            # print(index, item_index)
        else:
            item_index = index
        
        # print(item_index)
        # print(len(self.used_split))
        index = self.used_pids[item_index]

        #load from hdf5 file
        file = np.load(self.file[str(index)][0])
        # print(file.shape)
        file = self.pad_or_crop_image(file)

        

        # else:
        #     # print(file.shape)
        image = np.load(self.file[str(index)][1])[0,...]
        labels = file[1,...]
            

        #Prepare data depeinding on soft/hard augmentation scheme
        n_classes = self.n_classes

        if self.mode == 'train':
            w,h,d = image.shape
            ps_h, ps_w, ps_d = self.patch_size

            x = random.randint(0, h- ps_h)
            y = random.randint(0, w- ps_w)
            z = random.randint(0, d- ps_d)

            xi, yi, zi = x//16, y//16, z//16
            ps_hi = 12
            ps_wi = 12
            ps_di = 3

            idx = (x,y,z)

            

            ptc_input = image[:,xi:(xi+ps_hi),yi:(yi+ps_wi),zi:(zi+ps_di)]
            # ptc_input = ptc_input[0,...]
            labels = labels[x:(x+ps_h),y:(y+ps_w),z:(z+ps_d)]

            #Transform if we have to
            if self.do_tr:
                # print("-->before transform")
                data = {'data':ptc_input[None, None, ...], 'seg':labels[None, None, ...]}
                # print(data['data'].shape, data['seg'].shape)
                data = self.tr(data)
                # print(data['data'].shape, data['seg'].shape)
                # print("middl transform")

                ptc_input = data['data'][0,0,...]
                labels = data['seg'][0,0,...]
                # print("-->after transform")
                # exit(0)
            ptc_input = torch.from_numpy(ptc_input)

            if self.ds != None:
                labels = self.ds(labels)
                # labels = [torch.from_numpy(l).long() for l in labels]


            # ptc_input = torch.reshape(ptc_input, (ps_h, ps_w, ps_d))
            if self.return_full_image:
                nh, nw, nd = 3,3,4
                crop = []
                pos = [torch.from_numpy(np.array(idx))[None,...]]
                for x in range(nh):
                    for y in range(nw):
                        for z in range(nd):
                            crop.append(torch.from_numpy(image[None,:,xi*ps_hi:(xi+1)*ps_hi,yi*ps_wi:(yi+1)*ps_wi,zi*ps_di:(zi+1)*ps_di]))
                            pos.append( torch.from_numpy(np.array((x,y,z)))[None,...] )
                crop = torch.cat(crop, dim=0)
                pos = torch.cat(pos, dim=0)

                # print(ptc_input.shape)
                
                return pos, torch.cat([ptc_input[None,...], crop], 0)[None,...], labels
            if self.return_pos:
                pos = torch.from_numpy(np.array(idx))[None,...]
                return pos, ptc_input[None, ...], labels
            return ptc_input[None, ...], labels

        if self.mode == 'test':
            pid = torch.from_numpy(np.array([self.used_pids[item_index]]))


            ps_h, ps_w, ps_d = self.patch_size
            ps_hi, ps_wi, ps_di = 12,12,3

            image = torch.from_numpy(image)
            chans,h,w,d = image.shape
            
            nh, nw, nd = int(h/ps_h), int(w/ps_w), int(d/ps_d)
            crop = torch.zeros(*(nh,nw,nd, ps_hi, ps_wi, ps_di))
            pos = []
            for x in range(nh):
                for y in range(nw):
                    for z in range(nd):
                        crop[x,y,z,...] = image[:,xi*ps_hi:(xi+1)*ps_hi,yi*ps_wi:(yi+1)*ps_wi,zi*ps_di:(zi+1)*ps_di]
                        pos.append( torch.from_numpy(np.array((x,y,z)))[None,...] )
            pos = torch.cat(pos, dim=0)
            # image = torch.cat(crop, dim=1)            
            # image = torch.reshape(image[0, ...], (nh,nw,nd, self.patch_size[0], self.patch_size[1], self.patch_size[2]))
            if self.return_pos: 
                return pid, pos, crop[None, ...], labels
            return pid, crop[None, ...], labels


    def __len__(self):
        # self.openFileIfNotOpen()
        if self.mode == 'train':
            return self.n_iter
        return self.n_files
        
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
