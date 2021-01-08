import os
import numpy as np
import logging
import string
import gc
import h5py
from skimage import transform

import utils
import torch.nn.functional as F
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

MAX_WRITE_BUFFER = 5
alpha_dic = {ch: n for n, ch in enumerate(string.ascii_uppercase)}

def normalise_image(image):
    '''
    standardize based on nonzero pixels
    '''
    m = np.nanmean(np.where(image == 0, np.nan, image), axis=(0, 1, 2)).astype(np.float32)
    s = np.nanstd(np.where(image == 0, np.nan, image), axis=(0,1,2)).astype(np.float32)
    normalized = np.divide((image - m), s)
    image = np.where(image == 0, 0, normalized)
    return image





def pad_slice_to_size(image, target_size):
    
    x_t, y_t, z_t = target_size[0:3]
    x_s, y_s, z_s = image.shape[0:3]

    xd = x_t - x_s
    yd = y_t - y_s
    zd = z_t - z_s

    xd_a, xd_b = xd // 2 + xd % 2, xd // 2
    yd_a, yd_b = yd // 2 + yd % 2, yd // 2
    zd_a, zd_b = zd // 2 + zd % 2, zd // 2

    pad_width = ((xd_a, xd_b),(yd_a, yd_b),(zd_a, zd_b))

    output_volume = np.pad(image, pad_width, mode='constant', constant_values = np.min(image))
    

    return output_volume

def get_im_id(f):
	return f[3:7]

def prepare_data(input_folder, output_file, size, input_channels, target_resolution):

    hdf5_file = h5py.File(output_file, "w")

    file_list = []

    logging.info('Counting files and parsing meta data...')

    pid = 0
    for folder in os.listdir(input_folder+ '/img'):
        print(get_im_id(folder))
        # train_test = test_train_val_split(pid)
        pid = pid + 1
        file_list.append(get_im_id(folder))


    n_train = len(file_list)
    

    print('Debug: Check if sets add up to correct value:')
    print(n_train)

    # Create datasets for images and masks
    data = {}
    for tt, num_points in zip(['train'], [n_train]):

        if num_points > 0:
            print([num_points] + list(size) + [input_channels])
            if input_channels != 1:
                data['images_%s' % tt] = hdf5_file.create_dataset("images_%s" % tt, [num_points] + list(size) + [input_channels],
                                                                  dtype=np.float32)
                data['masks_%s' % tt] = hdf5_file.create_dataset("masks_%s" % tt, [num_points] + list(size), dtype=np.uint8)
                data['pids_%s' % tt] = hdf5_file.create_dataset("pids_%s" % tt, [num_points] , dtype=h5py.special_dtype(vlen=str))
            else:
                data['images_%s' % tt] = hdf5_file.create_dataset("images_%s" % tt, [num_points] + list(size),
                                                                  dtype=np.float32)
                data['masks_%s' % tt] = hdf5_file.create_dataset("masks_%s" % tt, [num_points] + list(size), dtype=np.uint8)
                data['pids_%s' % tt] = hdf5_file.create_dataset("pids_%s" % tt, [num_points] , dtype=h5py.special_dtype(vlen=str))

    mask_list = []
    img_list = []
    pids_list = []

    logging.info('Parsing image files')

    #get max dimension in z-axis
    maxX = 0
    maxY = 0
    maxZ = 0
    i = 0
    for train_test in ['train']:
        for file in file_list:
            print("Doing file {}".format(i))
            i += 1

            baseFilePath = os.path.join(input_folder, 'img','img'+file+'.nii.gz')
            img_dat, _, img_header = utils.load_nii(baseFilePath)

            maxX = max(maxX, img_dat.shape[0])
            maxY = max(maxY, img_dat.shape[1])
            maxZ = max(maxZ, img_dat.shape[2])

    print("Max x: {}, y: {}, z: {}".format(maxX, maxY, maxZ))

    for train_test in ['train']:

        write_buffer = 0
        counter_from = 0

        for file in file_list:

            logging.info('-----------------------------------------------------------')
            logging.info('Doing: %s' % file)

            patient_id = file

            baseFilePath = os.path.join(input_folder, 'img','img'+file+'.nii.gz')
            img, _, img_header = utils.load_nii(baseFilePath )
            mask, _, _ = utils.load_nii(os.path.join(input_folder, 'label','label'+file+'.nii.gz'))

            # print("mask sum ", np.sum(mask))

            img = pad_slice_to_size(img, (512, 512, 200))
            mask = pad_slice_to_size(mask, (512, 512, 200))

            print_info(img, "X")
            print_info(mask, "Y")

            # print("mask sum ", np.sum(mask))
            scale_vector = target_resolution

            if scale_vector != [1.0]:
                # print(img.shape)
                # #img = transform.resize(img, size)
                # img = transform.rescale(img, scale_vector[0], anti_aliasing=False, preserve_range=True)
                # #mask = transform.resize(mask, size)
                # mask = rescale_labels(mask, scale_vector[0])

                img = F.interpolate(torch.from_numpy(img)[None, None, :, :, :].float(), size = size, mode = 'trilinear', align_corners = True).numpy()[0,0,...]
                mask = rescale_labels(mask, scale_vector[0], size)

                np.save('checkpoints/images/img.npy', img)
                np.save('checkpoints/images/mask.npy', mask)

                
                print_info(img, "x")
                print_info(mask, "y", unique = True)

            # print("mask sum ", np.sum(mask))
            img = normalise_image(img)

            img_list.append(img)
            mask_list.append(mask)
            pids_list.append(patient_id)

            write_buffer += 1

            if write_buffer >= MAX_WRITE_BUFFER:

                counter_to = counter_from + write_buffer
                _write_range_to_hdf5(data, 'train', img_list, mask_list, pids_list, counter_from, counter_to)
                _release_tmp_memory(img_list, mask_list, pids_list, 'train')

                # reset stuff for next iteration
                counter_from = counter_to
                write_buffer = 0

        logging.info('Writing remaining data')
        counter_to = counter_from + write_buffer

        if len(file_list) > 0:
            _write_range_to_hdf5(data, 'train', img_list, mask_list, pids_list, counter_from, counter_to)
        _release_tmp_memory(img_list, mask_list, pids_list, 'train')

    # After test train loop:
    hdf5_file.close()


def rescale_labels(y, factor, new_shape,  c = 14):
    s = y.shape
    # ret = np.zeros((c, int(round(s[0]*factor)), int(round(s[1]*factor)), int(round(s[2]*factor))))
    ret = np.zeros( tuple([c] + list(new_shape)))
    for i in range(c):
        a = (y == i)
        # a = transform.rescale(a, factor, preserve_range=True, anti_aliasing=False, order=0)
        a = F.interpolate(torch.from_numpy(a)[None, None, :, :, :].float(), size = new_shape, mode='trilinear', align_corners = True).numpy()
        # print('a.shape :',a.shape)
        # print('ret.shape :',ret.shape)
        # print('a[0,0,...].shape :',a[0,0,...].shape)
        ret[i,...] = a[0,0,...]
    a = np.argmax(ret, axis=0)
    return a


def print_info(x, name, unique = False):
    txt = "## INFO : "+name+"##\n"
    txt += "mean : " + str(np.mean(x)) + "\n"
    txt += "min : " + str(np.min(x)) + "\n"
    txt += "max : " + str(np.max(x)) + "\n"
    txt += "sum : " + str(np.sum(x)) + "\n"
    if unique:
        txt += "unique : " + str(np.unique(x)) + "\n"
    txt += "#############"

    print(txt)

def _write_range_to_hdf5(hdf5_data, train_test, img_list, mask_list, pids_list, counter_from, counter_to):
    '''
    Helper function to write a range of data to the hdf5 datasets
    '''

    logging.info('Writing data from %d to %d' % (counter_from, counter_to))

    img_arr = np.asarray(img_list, dtype=np.float32)
    mask_arr = np.asarray(mask_list, dtype=np.uint8)

    # a = np.sum(mask_arr)
    # b = [np.sum(i) for i in mask_list]
    # c = sum(b)

    # print("all mask sum ", a, '| ',c, '|',b)
    # print("min", [np.min(i) for i in mask_list])
    # print("max", [np.max(i) for i in mask_list])
    # exit(0)

    hdf5_data['images_%s' % train_test][counter_from:counter_to, ...] = img_arr
    hdf5_data['masks_%s' % train_test][counter_from:counter_to, ...] = mask_arr
    hdf5_data['pids_%s' % train_test][counter_from:counter_to, ...] = pids_list


def _release_tmp_memory(img_list, mask_list, pids_list, train_test):
    '''
    Helper function to reset the tmp lists and free the memory
    '''

    img_list.clear()
    mask_list.clear()
    pids_list.clear()
    gc.collect()





# TO DO
def load_and_maybe_process_data(input_folder,
                                preprocessing_folder,
                                size,
                                input_channels,
                                target_resolution,
                                force_overwrite=False):
    '''
    This function is used to load and if necessary preprocesses the ACDC challenge data

    :param input_folder: Folder where the raw ACDC challenge data is located
    :param preprocessing_folder: Folder where the proprocessed data should be written to
    :param size: Size of the output slices/volumes in pixels/voxels
    :param target_resolution: Resolution to which the data should resampled. Should have same shape as size
    :param force_overwrite: Set this to True if you want to overwrite already preprocessed data [default: False]

    :return: Returns an h5py.File handle to the dataset
    '''

    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])

    data_file_name = 'data_3D_size_%s_res_%s.hdf5' % (size_str, res_str)

    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    utils.makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:
        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_folder, data_file_path, size, input_channels, target_resolution)
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')


if __name__ == '__main__':
    input_folder = "/local/SSD_DEEPLEARNING/MULTI_ATLAS/RawData/Training/"
    preprocessing_folder = "/local/SSD_DEEPLEARNING/MULTI_ATLAS/multi_atlas/"
    # target_size = (512x512x~198) # ORIGINAL SIZE
    # target_size = (512, 512, 198)
    rescale = [1.0]
    # target_size = (512//10, 512//10, 198//10)
    # target_size = (int(round(512*rescale[0])), int(round(512*rescale[0])), int(round(198*rescale[0])))
    target_size = (512,512,200)
    # rescale = [0.1]

    d = load_and_maybe_process_data(input_folder, preprocessing_folder, target_size, 1, rescale, force_overwrite=True)



# INPUT VOLUMES SHAPE : (240, 240, 155)



