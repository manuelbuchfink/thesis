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


class Initialization_ConeBeam:
    def __init__(self, image_size, num_proj, start_angle):
        '''
        image_size: [z, x, y]
        proj_size: [h, w]
        '''
        self.param = {}

        self.image_size = image_size
        self.num_proj = num_proj
        #self.reso = 512. / np.max((self.image_size[1], 1)) # avoid div by zero

        ## Imaging object (reconstruction objective) with object center as origin
        self.param['nx'] = self.image_size[1]
        self.param['ny'] = self.image_size[2]
        self.param['nz'] = self.image_size[0]

        self.param['sx'] = self.param['nx'] #* self.reso
        self.param['sy'] = self.param['ny'] #* self.reso
        self.param['sz'] = self.param['nz'] #* self.reso

        ## Projection view angles (ray directions)
        self.param['start_angle'] = start_angle            # 0
        self.param['end_angle'] = start_angle +  2 * np.pi # 360
        self.param['nProj'] = num_proj

        ## Detector
        self.param['sh'] = self.param['sx'] * (1.5) # Size of a detector pixel. 768
        self.param['sw'] = np.sqrt(self.param['sx']**2 + self.param['sy']**2) * (1.5) # 724
        self.param['nh'] = self.image_size[2]
        self.param['nw'] = self.image_size[1]
        '''
        dde = source_to_detector - source_to_isocenter = 1085,6 - 595 = 490,6
        '''
        self.param['dde'] = 500 #* self.reso # distance between origin and detector center (assume in x axis)
        self.param['dso'] = 1000 #* self.reso # distance between origin and source (assume in x axis)

def build_conebeam_geometry(param):
    # Reconstruction space:
    reco_space = odl.uniform_discr(min_pt=[-param.param['sx'] / 2.0, -param.param['sy'] / 2.0, -param.param['sz'] / 2.0],
                                    max_pt=[param.param['sx'] / 2.0, param.param['sy'] / 2.0, param.param['sz'] / 2.0],
                                    shape=[param.param['nx'], param.param['ny'], param.param['nz']],
                                    dtype='float16')

    angle_partition = odl.uniform_partition(min_pt=param.param['start_angle'],
                                            max_pt=param.param['end_angle'],
                                            shape=param.param['nProj'])

    detector_partition = odl.uniform_partition(min_pt=[-(param.param['sh'] / 2.0), -(param.param['sw'] / 2.0)],
                                               max_pt=[(param.param['sh'] / 2.0), (param.param['sw'] / 2.0)],
                                               shape=[param.param['nh'], param.param['nw']]) # projection size

    # Fan-beam geometry for 3D projection
    geometry = odl.tomo.ConeBeamGeometry(apart=angle_partition, # partition of the angle interval
                                        dpart=detector_partition, # partition of the detector parameter interval
                                        src_radius=param.param['dso'], # radius of the source circle
                                        det_radius=param.param['dde'], # radius of the detector circle
                                        axis=[0, 0, 1] # rotation axis is z-axis: (0, 0, 1)
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
class Projection_ConeBeam(nn.Module):
    def __init__(self, param):
        super(Projection_ConeBeam, self).__init__()
        self.param = param
        #self.reso = param.reso

        # RayTransform operator
        reco_space, ray_trafo, FBPOper = build_conebeam_geometry(self.param)

        # Wrap pytorch module
        self.trafo = odl_torch.OperatorModule(ray_trafo)

        self.back_projector = odl_torch.OperatorModule(ray_trafo.adjoint)

    def forward(self, x):
        x = self.trafo(x)
        x = x #/ self.reso
        return x

    def back_projection(self, x):
        x = self.back_projector(x)
        return x


# FBP reconstruction
class FBP_ConeBeam(nn.Module):
    def __init__(self, param):
        super(FBP_ConeBeam, self).__init__()
        self.param = param

        reco_space, ray_trafo, FBPOper = build_conebeam_geometry(self.param)

        self.fbp = odl_torch.OperatorModule(FBPOper)

    def forward(self, x):
        x = self.fbp(x)
        return x

    def filter_function(self, x):
        x_filter = self.filter(x)
        return x_filter


class ConeBeam3DProjector():
    def __init__(self, image_size, num_proj):

        self.image_size = image_size
        self.num_proj = num_proj
        self.start_angle = 0

        # Initialize required parameters for image, view, detector
        geo_param = Initialization_ConeBeam(image_size=image_size,
                                            num_proj=self.num_proj,
                                            start_angle=self.start_angle
                                            )
        # Forward projection function
        self.forward_projector = Projection_ConeBeam(geo_param)

        # Filtered back-projection
        self.fbp = FBP_ConeBeam(geo_param)

    def forward_project(self, volume):
        '''
        Arguments:
            volume: torch tensor with input size (B, C, img_x, img_y, img_z)
        '''
        proj_data = self.forward_projector(volume)
        return proj_data

    def backward_project(self, projs):
        '''
        Arguments:
            projs: torch tensor with input size (B, num_proj, proj_size_h, proj_size_w)
        '''
        volume = self.fbp(projs)
        return volume

