

import numpy as np
from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform

from batchgenerators.transforms import SegChannelSelectionTransform, SpatialTransform, MirrorTransform, GammaTransform, Compose

from nnunet.training.data_augmentation.custom_transforms import Convert3DTo2DTransform, Convert2DTo3DTransform

from batchgenerators.transforms.utility_transforms import RemoveLabelTransform

from batchgenerators.dataloading import MultiThreadedAugmenter
##################################################################################
## Now we have to build a new a dataloader from "batchgenerators". Don't know   ##
## how hard it is, but we went to far too stop there !!!!!!!!                   ##
##                                                                              ##
## https://github.com/MIC-DKFZ/batchgenerators/tree/master/batchgenerators      ##
##                                                                              ##
##                                                                              ##
## PS : Good Luck !                                                             ##
##################################################################################


class TransformData():
    def __init__(self):
        patch_size = (576,576,192)

        rotation_x = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
        rotation_y = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        rotation_z = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)

        scale_range = (0.7, 1.4)
        p_scale = 0.2
        p_rot = 0.2

        gamma_retain_stats = True
        gamma_range = (0.7, 1.5)
        p_gamma = 0.3


        self.tr = []

        # Precise to the transform module that segmentation have multiple channels
        # and give it the axis of this channels.
        self.tr += [SegChannelSelectionTransform([0])]


        # Spatial transormation : rotation, scaling
        # self.tr += [Convert3DTo2DTransform()]
        self.tr += [SpatialTransform(
            patch_size, patch_center_dist_from_border=None, do_elastic_deform=False,
            do_rotation=True, angle_x=rotation_x, angle_y=rotation_y,angle_z=rotation_z, 
            do_scale=True, scale=scale_range,
            border_mode_data="constant", border_cval_data=0, order_data=3, border_mode_seg="constant",
            border_cval_seg=-1,
            order_seg=1, random_crop=False, p_el_per_sample=0.2,
            p_scale_per_sample=p_scale, p_rot_per_sample=p_rot,
            independent_scale_for_each_axis=False)]
        # self.tr += [Convert2DTo3DTransform()]

        # Gamma transfomation
        self.tr += [GammaTransform(gamma_range, False, True, 
                                retain_stats=gamma_retain_stats,
                                p_per_sample=p_gamma)]

        # Mirroting
        self.tr += [MirrorTransform((0, 1, 2))]


        # Create the composed transform module
        self.tr = Compose(self.tr)
        # batchgenerator_train = MultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'),
  #                                                 params.get("num_cached_per_thread"), seeds=seeds_train,
  #                                                 pin_memory=pin_memory)

    def __call__(self, data_dict):
        return self.tr(**data_dict)



