'''
adapted from
https://arxiv.org/pdf/2108.10991
NeRP: Implicit Neural Representation Learning
with Prior Embedding for Sparsely Sampled
Image Reconstruction
Liyue Shen, John Pauly, Lei Xing
'''
import numpy as np
import torch.nn as nn
import odl
from odl.contrib import torch as odl_torch


class Initialization_FanBeam:
    def __init__(self, image_height, image_width, num_proj, start_angle, proj_size):
        '''
        image_size: [x, y]
        proj_size: h = 512
        '''
        self.param = {}

        self.num_proj = num_proj
        self.image_width = image_width
        self.image_height = image_height


        self.reso = 512. / np.max((self.image_width, 1)) # avoid div by zero

        ## Imaging object (reconstruction objective) with object center as origin
        self.param['nx'] = self.image_width
        self.param['ny'] = self.image_height
        self.param['nh'] = proj_size

        self.param['sx'] = self.param['nx'] * self.reso
        self.param['sy'] = self.param['ny'] * self.reso

        ## Projection view angles (ray directions)
        self.param['start_angle'] = start_angle            # 0
        self.param['end_angle'] = start_angle +  2 * np.pi # 360
        self.param['nProj'] = num_proj

        ## Detector
        self.param['sh'] = self.param['sx'] * (1.286) # Size of a detector pixel.

        '''
        dde = source_to_detector - source_to_isocenter = 1085,6 - 595 = 490,6
        '''
        self.param['dde'] = 490.6 * self.reso # distance between origin and detector center (assume in x axis)
        self.param['dso'] = 595 * self.reso # distance between origin and source (assume in x axis)

def build_fanbeam_geometry(param):
    # Reconstruction space:
    reco_space = odl.uniform_discr(min_pt=[-param.param['sx'] / 2.0, -param.param['sy'] / 2.0],
                                    max_pt=[param.param['sx'] / 2.0, param.param['sy'] / 2.0],
                                    shape=[param.param['nx'], param.param['ny']],
                                    dtype='float32')

    angle_partition = odl.uniform_partition(min_pt=param.param['start_angle'],
                                            max_pt=param.param['end_angle'],
                                            shape=param.param['nProj'])

    detector_partition = odl.uniform_partition(min_pt=-(param.param['sh']),
                                               max_pt=(param.param['sh']),
                                               shape=param.param['nh']) # projection size

    # Fan-beam geometry for 2D projection
    geometry = odl.tomo.FanBeamGeometry(apart=angle_partition, # partition of the angle interval
                                        dpart=detector_partition, # partition of the detector parameter interval
                                        src_radius=param.param['dso'], # radius of the source circle
                                        det_radius=param.param['dde'] # radius of the detector circle
                                        )

    ray_trafo = odl.tomo.RayTransform(vol_space=reco_space, # domain of forward projector
                                      geometry=geometry, # geometry of the transform
                                      impl='astra_cuda'
                                      ) # implementation back-end for the transform: ASTRA toolbox, using CUDA, 2D

    FBPOper = odl.tomo.fbp_op(ray_trafo=ray_trafo,
                             frequency_scaling=1.0)

    # Reconstruction space for imaging object, RayTransform operator, Filtered back-projection operator
    return reco_space, ray_trafo, FBPOper


# Projector
class Projection_FanBeam(nn.Module):
    def __init__(self, param):
        super(Projection_FanBeam, self).__init__()
        self.param = param
        self.reso = param.reso

        # RayTransform operator
        reco_space, ray_trafo, FBPOper = build_fanbeam_geometry(self.param)

        # Wrap pytorch module
        self.trafo = odl_torch.OperatorModule(ray_trafo)

        self.back_projector = odl_torch.OperatorModule(ray_trafo.adjoint)

    def forward(self, x):
        x = self.trafo(x)
        x = x / self.reso
        return x

    def back_projection(self, x):
        x = self.back_projector(x)
        return x


# FBP reconstruction
class FBP_FanBeam(nn.Module):
    def __init__(self, param):
        super(FBP_FanBeam, self).__init__()
        self.param = param

        reco_space, ray_trafo, FBPOper = build_fanbeam_geometry(self.param)

        self.fbp = odl_torch.OperatorModule(FBPOper)

    def forward(self, x):
        x = self.fbp(x)
        return x

    def filter_function(self, x):
        x_filter = self.filter(x)
        return x_filter


class FanBeam2DProjector():
    def __init__(self, image_height, image_width, proj_size, num_proj):

        self.image_height = image_height
        self.image_width = image_width
        self.proj_size = proj_size
        self.num_proj = num_proj
        self.start_angle = 0

        # Initialize required parameters for image, view, detector
        geo_param = Initialization_FanBeam(image_height=self.image_height,
                                           image_width=self.image_width,
                                            proj_size=self.proj_size,
                                            num_proj=self.num_proj,
                                            start_angle=self.start_angle
                                            )
        # Forward projection function
        self.forward_projector = Projection_FanBeam(geo_param)

        # Filtered back-projection
        self.fbp = FBP_FanBeam(geo_param)

    def forward_project(self, slice):
        '''
        Arguments:
            slice: torch tensor with input size (1, img_x, img_y)
        '''
        proj_data = self.forward_projector(slice)
        return proj_data

    def backward_project(self, projs):
        '''
        Arguments:
            projs: torch tensor with input size (B, num_proj)
        '''
        volume = self.fbp(projs)
        return volume
