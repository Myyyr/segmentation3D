import torch.utils.data as data
import numpy as np
import datetime

from os import listdir
from os.path import join
from tciautils import load_nifti_img, check_exceptions, is_image_file

import torchvision
import skimage.transform
import torchsample.transforms as ts

import torch
import torchio as tio

import random


def shape2str(s):
    return str(s[0])+'_'+str(s[1])+'_'+str(s[2])


class Patched3DSplitTCIA3DDataset(data.Dataset):
    def __init__(self, root_dir, split, dt_splits, patch_size, transform=None, hot = 0, mode = 'train'):
        super(Patched3DSplitTCIA3DDataset, self).__init__()
        
        self.patch_size = patch_size
        self.hot = hot
        self.mode = mode
        self.data_splits = []

        # list_dir = []

        self.image_filenames = []
        self.target_filenames = []

        for i in dt_splits:
            # list_dir.append(join(root_dir, i))

            image_dir = join(root_dir, i, 'image')
            # print("\n\n\n", image_dir,"\n\n\n")
            target_dir = join(root_dir, i, 'label')


            self.image_filenames  += [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
            self.target_filenames += [join(target_dir, x) for x in listdir(target_dir) if is_image_file(x)]

            self.data_splits += [self.get_pid(i) for i in listdir(image_dir) if is_image_file(i)]

        self.image_filenames = sorted(self.image_filenames)
        self.target_filenames = sorted(self.target_filenames)

        assert len(self.image_filenames) == len(self.target_filenames)

        # report the number of images in the dataset
        print('Number of {0} images: {1} NIFTIs'.format(split, self.__len__()))

        # data augmentation
        self.transform = transform


        
        


    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images
        input, _ = load_nifti_img(self.image_filenames[index], dtype=np.int16)
        target, _ = load_nifti_img(self.target_filenames[index], dtype=np.uint8)
        

        #check_exceptions(input, target)
        if self.transform != None:
            sub = tio.Subject(input = tio.ScalarImage(tensor = input[None, :,:,:]), 
                              target = tio.LabelMap(tensor = target[None, :,:,:]))
            sub = self.transform(sub)
            input = np.array(sub['input'])[0,...]
            target = np.array(sub['target'])[0,...]

        if self.hot == 1:
            target = self._toEvaluationOneHot(target)
        input = torch.from_numpy(input[None,:,:,:]).float()
        target = torch.from_numpy(target).long()
        

        if self.mode == 'train':

            x = random.randint(0, h-self.ps_h)
            y = random.randint(0, w-self.ps_w)
            z = random.randint(0, d-self.ps_d)

            ps_h, ps_w, ps_d = self.patch_size

            input = input[:,x:(x+ps_h),y:(y+ps_w),z:(z+ps_d)]
            target = target[x:(x+ps_h),y:(y+ps_w),z:(z+ps_d)]

            return input, target

        if self.mode == 'test':
            pid = torch.from_numpy(np.array([self.data_splits[index]]))


            ps_w, ps_h, ps_d = self.patch_size
            b, w,h,d = input.shape
            if w%ps_w != 0:
                print("H, W, D must be multiple of patch size")
                exit(0)
            nw, nh, nd = int(w/ps_w), int(h/ps_h), int(d/ps_d)
            input = torch.reshape(input, (b,nw,nh,nd, self.patch_size[0], self.patch_size[1], self.patch_size[2]))
            input = input.permute((1,2,3,0,4,5,6))

            #target = torch.reshape(target, (n**3, int(w/ps_w), int(d/ps_d), int(h/ps_h)))

            return pid, input, target



    def _toEvaluationOneHot(self, labels):
        shape = labels.shape
        out = np.zeros([2, shape[0], shape[1], shape[2]], dtype=np.float32)
        for i in range(2):
            out[i, ...] = (labels == i)
        return out

    def get_pid(self, index):
        a = index.replace('.nii','').replace('000','').replace('00','')
        return int(a)


    def __len__(self):
        return len(self.image_filenames)