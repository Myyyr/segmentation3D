import sys
import os
import numpy as np
import nibabel as nib

import torch.nn.functional as F
import torch
import torch.nn as nn


def normalise_image(image):
    '''
    standardize based on nonzero pixels
    '''
    m = np.nanmean(np.where(image == 0, np.nan, image), axis=(0, 1, 2)).astype(np.float32)
    s = np.nanstd(np.where(image == 0, np.nan, image), axis=(0,1,2)).astype(np.float32)
    normalized = np.divide((image - m), s)
    image = np.where(image == 0, 0, normalized)
    return image


def file_list(path):
	return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def save_nii(nii_image, path):
	nii_image.to_filename(path)

def load_npy(path):
	return np.load(path)

def npy2nii(npy_img):
	return nib.Nifti1Image(npy_img, np.eye(4))

def set_up_folders(path):
	os.mkdir(os.path.join(path,'train'))
	os.mkdir(os.path.join(path,'test'))
	os.mkdir(os.path.join(path,'validation'))

	for split in ['train', 'test', 'validation']:
		os.mkdir(os.path.join(path, split, 'label'))
		os.mkdir(os.path.join(path, split, 'image'))

def set_up_splits_folders(path, n_split=6, size = None):
	for i in range(n_split):
		if not os.path.exists(os.path.join(path, "split_"+str(i+1))):
			os.mkdir(os.path.join(path, "split_"+str(i+1)))

	for i in range(n_split):
		if not os.path.exists(os.path.join(path, "split_"+str(i+1), 'label')):
			os.mkdir(os.path.join(path, "split_"+str(i+1), 'label'))

		if not os.path.exists(os.path.join(path, "split_"+str(i+1), 'image')):
			os.mkdir(os.path.join(path, "split_"+str(i+1), 'image'))

def pid2niipid(pid):
	return ''.join(['0']*(4 - len(pid))) + pid


def transform_size(img, size = None):
	if size != None:
		img = torch.from_numpy(img)
		img = torch.unsqueeze(img, 0)
		img = torch.unsqueeze(img, 0)
		img = nn.functional.interpolate(img, size, mode='trilinear')
		img = torch.squeeze(img)
		img = torch.squeeze(img)
	return img.numpy()

def rescale_labels(y, new_shape,  c = 14):
    s = y.shape
    ret = np.zeros( tuple([c] + list(new_shape)))
    for i in range(c):
        a = (y == i)
        a = F.interpolate(torch.from_numpy(a)[None, None, :, :, :].float(), size = new_shape, mode='trilinear', align_corners = True).numpy()
        ret[i,...] = a[0,0,...]
    a = np.argmax(ret, axis=0)
    return a


def main(root_path, out_dir, n_split = 6, size = None):
	train = ['74', '52', '22', '81', '44', '40', '35', '76', '58', '54', '77', '13', '45', '41', '3', '50', '8', '18', '43', '39', '80', '67', '66', '25', '32', '46', '49', '51', '53', '28', '16', '36', '11', '61', '21', '78', '17', '71', '73', '56', '48', '65', '34', '10', '27', '15', '1', '68', '57', '37', '20', '59', '4', '7', '33', '79', '9', '75', '82', '47', '29', '2', '72', '24', '70']
	test  = ['6', '62', '64', '55', '38', '26', '5', '30', '12', '42', '19', '14', '31', '60', '63', '69', '23']
	splits = np.array(train + test + [-1,-1])
	splits = np.reshape(splits, (n_split,int(splits.shape[0]/n_split)) )
	
	set_up_splits_folders(out_dir ,n_split=n_split)

	for split in ['images', 'labels']:
		fl = file_list(os.path.join(root_path, split))
		print(os.path.join(root_path, split))
		print("Nmmber of file :",len(fl))
		for f in fl:
			pid = f.replace('.npy','')
			print(split, '::', pid )
			niipid = pid2niipid(pid)


			npyimg = load_npy(os.path.join(root_path, split, f))
			
			# change size
			if split == 'images':
				npyimg = normalise_image(npyimg)
				npyimg = transform_size(npyimg, size)
			else:
				npyimg = rescale_labels(npyimg, size, c = 2)
			niiim  = npy2nii(npyimg)

			out_path = ''
			for i in range(n_split):
				if pid in list(splits[i,:]):
					out_path = os.path.join(out_dir,'split_'+str(i+1), split.replace('s',''), niipid+'.nii')
			# if pid in train:
			# 	out_path = os.path.join(root_path, 'train', split.replace('s',''), niipid+'.nii')
			# elif pid in test:
			# 	out_path = os.path.join(root_path, 'test', split.replace('s',''), niipid+'.nii')

			save_nii(niiim, out_path)


if __name__ == '__main__':
	main("/local/SSD_DEEPLEARNING/PANCREAS_MULTI_RES/TCIA_torch/", "/local/SSD_DEEPLEARNING/PANCREAS_MULTI_RES/160_160_64", 6, (160,160,64))