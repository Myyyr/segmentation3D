import numpy as np
import torch
import inspect

from batchgenerators.augmentations.utils import convert_seg_image_to_one_hot_encoding_batched, resize_segmentation


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
        axes = list(range(2, len(seg.shape)))
    output = []
    for s in ds_scales:
        if all([i == 1 for i in s]):
            output.append(seg)
        else:
            new_shape = np.array(seg.shape).astype(float)
            for i, a in enumerate(axes):
                new_shape[a] *= s[i]
            new_shape = np.round(new_shape).astype(int)
            out_seg = np.zeros(new_shape, dtype=seg.dtype)
            for b in range(seg.shape[0]):
                for c in range(seg.shape[1]):
                    out_seg[b, c] = resize_segmentation(seg[b, c], new_shape[2:], order, cval)
            output.append(out_seg)
    return output








class EnhancedCompose(object):
    """Composes several transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            if isinstance(t, collections.Sequence):
                tmp_ = []
                #gen seed so that label and image will match when randomize
                seed = np.random.randint(10000)
                for i, im_ in enumerate(img):
                    if callable(t[i]):
                        #we add a seed here to generate same seed for input and target
                        if 'seed' in inspect.getargspec(t[i]).args:
                            tmp_.append(t[i](im_, seed=seed))
                        else: tmp_.append(t[i](im_))
                    else: tmp_.append(im_)
                img = tmp_
            elif callable(t):
                if isinstance(img, collections.Sequence): img[0] = t(img[0])
                else: img = t(img)
            elif t is None: continue
            else: raise Exception('unexpected type')
        return img