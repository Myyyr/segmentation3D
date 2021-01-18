import os
import argparse
import numpy as np
import copy
import threading
import tqdm
import time
import yaml
import pprint
import sys
# sys.path.append("~/stage2020/prototype.tar/prototype/datasets")
# import datasets
# from datasets.lits.lits_dataset import LITSDataset
from datasets.tcia.tcia_pancreas_dataset import TCIAPancreasDataset
# from datasets.vpdataset.vp_dataset import VPDataset

# from tcia.tcia_pancreas_dataset import TCIAPancreasDataset

from experimentation import read_config_file

SAVE_MODES = ['slices', 'volumes']
SLICE_VIEWS_AXES = {'axial' : 2,
                    'sagittal' : 1,
                    'coronal' : 0}


def get_dataset(dataset_name):
    if dataset_name == 'tcia':
        return TCIAPancreasDataset
    elif dataset_name == 'lits':
        return LITSDataset
    elif dataset_name == 'vpdataset':
        return VPDataset
    else:
        raise Exception('Unknown dataset name :', dataset_name)


# Data array is saved as follow {target_directory}/pid/slice_num.npy
def serialize_slice(arr, patient_id, slice_num, target_directory, verbose=False):
    filename = os.path.join(target_directory, str(patient_id), '{}.npy'.format(slice_num))
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if verbose:
        print('Saving file in : {}'.format(filename))
    np.save(filename, arr)

    
# Data array is saved as follow {target_directory}/pid.npy
def serialize_volume(arr, patient_id, target_directory, verbose=False):
    filename = os.path.join(target_directory, '{}.npy'.format(patient_id))
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if verbose:
        print('Saving file in : {}'.format(filename))
    np.save(filename, arr)


def get_slice(image, i, slice_view):
    if slice_view == 'axial':
        return image[:,:,i]
    elif slice_view == 'sagittal':
        return image[:,i,:]
    elif slice_view == 'coronal':
        return image[i,:,:]
    else:
        return None
    

processed_patient = 0
def browse_pids(pids, dataset, data_parameters, image_output_dir, annotation_output_dir, save_mode, slice_view):
    global processed_patient
    
    for i, pid in enumerate(pids):
        s_time = time.time()
            
        image, annotation = dataset.get_patient_data_by_id(pid,
                                                           size=data_parameters['size'],
                                                           voxel_spacing=data_parameters['voxel_spacing'],
                                                           windowing=data_parameters['windowing'])
        
        if save_mode == 'slices':
            for s in range(image.shape[SLICE_VIEWS_AXES[slice_view]]):
                slice_image = get_slice(image, s, slice_view)
                slice_annotation = get_slice(annotation, s, slice_view)
                serialize_slice(slice_image, pid, s, image_output_dir)
                serialize_slice(slice_annotation, pid, s, annotation_output_dir)
        elif save_mode == 'volumes':
            
            serialize_volume(image, pid, image_output_dir)
            serialize_volume(annotation, pid, annotation_output_dir)

        processed_patient += 1

        print('{} / {} ({:.2f}s)'.format(processed_patient, len(dataset), time.time()-s_time), end='\r')


        

def preprocess_and_serialize(data_parameters, data_dir, image_output_dir, annotation_output_dir, save_mode='slices', slice_view='axial', njobs=3):

    assert save_mode in SAVE_MODES
    assert slice_view in SLICE_VIEWS_AXES.keys()

    dataset_cls = get_dataset(data_parameters['dataset'])
    dataset = dataset_cls(data_dir, data_parameters['class_info'])
    nb_patient = len(dataset)

    print("Start writing {} patients in ({}, {}), save_mode='{}', slice_view='{}'".format(nb_patient, image_output_dir, annotation_output_dir, save_mode, slice_view))
    
    nb_patient_per_job = []
    for i in range(njobs+1):
        nb_patient_per_job.append(i * (nb_patient // njobs))
    nb_patient_per_job[-1] = nb_patient_per_job[-1] + nb_patient % njobs

    all_jobs = []
    for j in range(njobs):
        pids_to_browse = dataset.all_patients_ids[nb_patient_per_job[j]:nb_patient_per_job[j+1]]
        job = threading.Thread(target=browse_pids, args=(pids_to_browse, dataset, data_parameters, image_output_dir, annotation_output_dir, save_mode, slice_view))
        job.start()
        all_jobs.append(job)

    for job in all_jobs:
        job.join()



if __name__ == "__main__":

    def argument_parser_def():
        parser = argparse.ArgumentParser()
        parser.add_argument('--config_file', type=str, required=True,
                            help='The configuration file with data parameters.')
        parser.add_argument('--data_dir', type=str, required=True,
                            help='Path where the raw data is stored')
        parser.add_argument('--image_output_dir', type=str, required=True,
                            help='Path where the npy of the features will be written')
        parser.add_argument('--annotation_output_dir', type=str, required=True,
                            help='Path where the npy of the annotations will be written')
        parser.add_argument('--save_mode', type=str, default='slices',
                            help='Save mode could be "slices" or "volumes"')
        parser.add_argument('--slice_view', type=str, default='axial',
                            help='View when the volume is slices. Used when save_mode is slices. Possible values are ["axial", "sagittal", "coronal"]')

                
        return parser

    parser = argument_parser_def()
    FLAGS = parser.parse_args()

    data_parameters = read_config_file(FLAGS.config_file)
    pprint.pprint(data_parameters)
    
    preprocess_and_serialize(data_parameters, FLAGS.data_dir, FLAGS.image_output_dir, FLAGS.annotation_output_dir, save_mode=FLAGS.save_mode, slice_view=FLAGS.slice_view)
