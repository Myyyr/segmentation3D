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
from PIL import Image
import tensorflow as tf

def shape2str(s):
    return str(s[0])+'_'+str(s[1])+'_'+str(s[2])


class SplitTCIA2DDataset(data.Dataset):
    def __init__(self, root_dir, split, data_splits, im_dim = None, transform=None, hot = 0, mode = 'train'):
        super(SplitTCIA2DDataset, self).__init__()
        

        self.image_filenames = []
        self.target_filenames = []
        self.mode = mode
        self.data_splits = data_splits

        for i in data_splits:
            for j in listdir(join(root_dir, 'images')):
                # print(i, j)
                # print(j)
                image_dir = join(root_dir, 'images', str(i), str(j)+'.npy')
                target_dir = join(root_dir, 'annotations', str(i), str(j)+'.npy')
                self.image_filenames .append(image_dir)
                self.target_filenames.append(target_dir)



            # self.image_filenames  += [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
            # self.target_filenames += [join(target_dir, x) for x in listdir(target_dir) if is_image_file(x)]

        # self.image_filenames = sorted(self.image_filenames)
        # self.target_filenames = sorted(self.target_filenames)

        assert len(self.image_filenames) == len(self.target_filenames)

        # report the number of images in the dataset
        print('Number of {0} images: {1} NIFTIs'.format(split, self.__len__()))

        # data augmentation
        self.transform = transform

        
        


    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        input = np.load(self.image_filenames[index])
        target = np.load(self.target_filenames[index])

       
        
        # if self.transform != None and self.mode == 'train':
        #     input = Image.fromarray(input)
        #     target = Image.fromarray(target)
        #     seed = np.random.randint(2147483647)
        #     random.seed(seed)
        #     input = self.transform(input)
        #     random.seed(seed)
        #     target = self.transform(target)

        if self.transform and self.mode == 'train':
            
            theta = np.random.randint(-6, 6)
            tx = np.random.randint(-15, 15)
            ty = np.random.randint(-15, 15)
            zx = 1 + np.random.rand() * 0.2 - 0.1
            zy = 1 + np.random.rand() * 0.2 - 0.1

            input = np.expand_dims(input, axis=-1)
            target = np.expand_dims(target, axis=-1)

            input = tf.keras.preprocessing.image.apply_affine_transform(input,
                                                                        theta=theta,
                                                                        tx=tx, ty=ty,
                                                                        zx=zx,
                                                                        zy=zy,
                                                                        fill_mode='nearest')
            target = tf.keras.preprocessing.image.apply_affine_transform(target,
                                                                         theta=theta,
                                                                         tx=tx, ty=ty,
                                                                         zx=zx, zy=zy,
                                                                         fill_mode='nearest',
                                                                         order=0)
            input = torch.from_numpy(input[None, :, :, 0]).float()
            target = torch.from_numpy(target[:,:,0]).long()

            return input, target


        elif self.mode=='valid':
            input = torch.from_numpy(input[None, :, :]).float()
            target = torch.from_numpy(target).long()
            return input, target
        else:
            pid = self.get_pid
            input = torch.from_numpy(input[None, :, :]).float()
            target = torch.from_numpy(target).long()
            return pid, input, target

    def get_pid(self, index):
        return int(self.image_filenames[index].split('/')[-3])


    def _toEvaluationOneHot(self, labels):
        shape = labels.shape
        out = np.zeros([2, shape[0], shape[1]], dtype=np.float32)
        for i in range(2):
            out[i, ...] = (labels == i)
        return out


    def __len__(self):
        return len(self.image_filenames)