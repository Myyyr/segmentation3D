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


def shape2str(s):
    return str(s[0])+'_'+str(s[1])+'_'+str(s[2])


class SplitTCIA2DDataset(data.Dataset):
    def __init__(self, root_dir, split, data_splits, im_dim = None, transform=None, hot = 0):
        super(SplitTCIA2DDataset, self).__init__()
        

        self.image_filenames = []
        self.target_filenames = []

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

        #check_exceptions(input, target)
        if self.transform != None:
            # sub = tio.Subject(input = tio.ScalarImage(tensor = input[None, :,:,:]), 
            #                   target = tio.LabelMap(tensor = target[None, :,:,:]))
            # sub = self.transform(sub)
            # input = np.array(sub['input'])[0,...]
            # target = np.array(sub['target'])[0,...]
            pass

        # if self.hot == 1:
        # target = self._toEvaluationOneHot(target)
        input = torch.from_numpy(input[None,:,:]).float()
        target = torch.from_numpy(target).long()

        return input, target


    def _toEvaluationOneHot(self, labels):
        shape = labels.shape
        out = np.zeros([2, shape[0], shape[1]], dtype=np.float32)
        for i in range(2):
            out[i, ...] = (labels == i)
        return out


    def __len__(self):
        return len(self.image_filenames)