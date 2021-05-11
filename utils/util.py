from __future__ import print_function
import torch
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
import json
import csv
from skimage.exposure import rescale_intensity

import torch


from batchgenerators.transforms import AbstractTransform
from batchgenerators.augmentations.utils import convert_seg_image_to_one_hot_encoding_batched, resize_segmentation

def nbp(n,pn):
    return n//pn + 0**(1*(n%pn==0))

def get_strides(n, pn):
    xn = nbp(n,pn)
    if n%pn ==0 :
        return [pn for i in range(xn-1)]
    r = xn*pn - n
    r1 = r//(xn-1)
    l = [pn - r1 for i in range(xn-1)]
    
    r2 = r%(xn-1)
    for i in range(r2):
        l[i] -= 1
    return l

def get_idx(n,pn):
    sn = get_strides(n,pn)
    #print(sn)
    l = [0]
    for i in range(len(sn)):
        l.append(l[-1]+sn[i])
    return l

def get_all_crops(inp, ps):
    h,w,d = inp.shape
    ph,pw,pd = ps
    
    nh = nbp(h, ph)
    nw = nbp(w, pw)
    nd = nbp(d, pd)

    print(nh, nw, nd)
    
    idx_h = get_idx(h,ph)
    idx_w = get_idx(w,pw)
    idx_d = get_idx(d,pd)
    
    print(idx_h, idx_w, idx_d)
    
    all_crops = np.zeros((nh,nw,nd,ph,pw,pd))

    all_count = np.zeros((h,w,d))

    for x in range(nh):
        for y in range(nw):
            for z in range(nd):
                tmp = inp[idx_h[x]:idx_h[x]+ph, idx_w[y]:idx_w[y]+pw, idx_d[z]:idx_d[z]+pd]
                all_crops[x,y,z,...] = tmp
                all_count[idx_h[x]:idx_h[x]+ph, idx_w[y]:idx_w[y]+pw, idx_d[z]:idx_d[z]+pd]+=np.ones((ph,pw,pd))
    
    
    return all_crops, all_count, idx_h, idx_w, idx_d


    
class DownsampleSegForDSTransform2(AbstractTransform):
    '''
    data_dict['output_key'] will be a list of segmentations scaled according to ds_scales
    '''
    def __init__(self, ds_scales=(1, 0.5, 0.25), order=0, cval=0, input_key="seg", output_key="seg", axes=None):
        self.axes = axes
        self.output_key = output_key
        self.input_key = input_key
        self.cval = cval
        self.order = order
        self.ds_scales = ds_scales

    def __call__(self, labels):
        labels = downsample_seg_for_ds_transform2(labels, self.ds_scales, self.order,
                                                           self.cval, self.axes)
        return labels


def downsample_seg_for_ds_transform2(seg, ds_scales=((1, 1, 1), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)), order=0, cval=0, axes=None):
    if axes is None:
        # axes = list(range(2, len(seg.shape)))
        axes = [0,1,2]
    output = []
    for s in ds_scales:
        if all([i == 1 for i in s]):
            output.append(seg)
        else:
            new_shape = np.array(seg.shape).astype(float)
            # print("#1", (new_shape))
            for i, a in enumerate(axes):
                new_shape[a] *= s[i]
            # print("#2", (new_shape))
            new_shape = np.round(new_shape).astype(int)
            out_seg = np.zeros(new_shape, dtype=seg.dtype)
            # for b in range(seg.shape[0]):
            #     for c in range(seg.shape[1]):
            out_seg = resize_segmentation(seg, new_shape, order, cval)
            # print("#3", s,new_shape, out_seg.shape)
            output.append(torch.from_numpy(out_seg))
    return output











# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imgtype='img', datatype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.ndim == 4:# image_numpy (C x W x H x S)
        mid_slice = image_numpy.shape[-1]//2
        image_numpy = image_numpy[:,:,:,mid_slice]
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    if imgtype == 'img':
        image_numpy = (image_numpy + 8) / 16.0 * 255.0
    if np.unique(image_numpy).size == int(1):
        return image_numpy.astype(datatype)
    return rescale_intensity(image_numpy.astype(datatype))


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def json_file_to_pyobj(filename):
    def _json_object_hook(d): return collections.namedtuple('X', d.keys())(*d.values())
    def json2obj(data): return json.loads(data, object_hook=_json_object_hook)
    return json2obj(open(filename).read())


def determine_crop_size(inp_shape, div_factor):
    div_factor= np.array(div_factor, dtype=np.float32)
    new_shape = np.ceil(np.divide(inp_shape, div_factor)) * div_factor
    pre_pad = np.round((new_shape - inp_shape) / 2.0).astype(np.int16)
    post_pad = ((new_shape - inp_shape) - pre_pad).astype(np.int16)
    return pre_pad, post_pad


def csv_write(out_filename, in_header_list, in_val_list):
    with open(out_filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(in_header_list)
        writer.writerows(zip(*in_val_list))
