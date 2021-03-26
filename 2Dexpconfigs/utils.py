import numpy as np
import torch
import inspect

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