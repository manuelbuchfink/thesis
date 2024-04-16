import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import odl
from odl.contrib import torch as odl_torch


class Initialization_FanBeam:
    def __init__(self, image_size, num_proj, start_angle,  raw_reso=0.7):
        '''
        image_size: [x, y], assume x = y for each slice image
        proj_size: [h, w]
        '''
        self.param = {}
        
        self.image_size = image_size
        self.num_proj = num_proj
        self.raw_reso = raw_reso
        
        self.reso = 512. / image_size[0] * raw_reso

        ## Imaging object (reconstruction objective) with object center as origin
        self.param['nx'] = image_size[0]        
        self.param['ny'] = image_size[1]

        self.param['sx'] = self.param['nx']*self.reso
        self.param['sy'] = self.param['ny']*self.reso

        ## Projection view angles (ray directions)
        self.param['start_angle'] = start_angle
        self.param['end_angle'] = start_angle + 2 * np.pi
        self.param['nProj'] = num_proj

        ## Detector
        self.param['sh'] = self.param['sx']*(1500/1000)       
        self.param['dde'] = 500*self.reso # distance between origin and detector center (assume in x axis)
        self.param['dso'] = 1000*self.reso # distance between origin and source (assume in x axis)

def build_fanbeam_geometry(param):
    # Reconstruction space:
    reco_space = odl.uniform_discr(min_pt=[-param.param['sx'] / 2.0, -param.param['sy'] / 2.0],
                                    max_pt=[param.param['sx'] / 2.0, param.param['sy'] / 2.0], 
                                    shape=[param.param['nx'], param.param['ny']],
                                    dtype='float32')
    
    angle_partition = odl.uniform_partition(min_pt=param.param['start_angle'], 
                                            max_pt=param.param['end_angle'],
                                            shape=param.param['nProj'])

    detector_partition = odl.uniform_partition(min_pt=-(param.param['sh'] / 2.0), 
                                                 max_pt=(param.param['sh'] / 2.0),
                                                 shape=param.param['nProj'])

    # Fan-beam geometry for 2D projection
    geometry = odl.tomo.FanBeamGeometry(apart=angle_partition, # partition of the angle interval
                                          dpart=detector_partition, # partition of the detector parameter interval
                                          src_radius=param.param['dso'], # radius of the source circle
                                          det_radius=param.param['dde'] # radius of the detector circle 
                                          )
    
    ray_trafo = odl.tomo.RayTransform(vol_space=reco_space, # domain of forward projector
                                     geometry=geometry, # geometry of the transform
                                     impl='astra_cuda',
                                     ) # implementation back-end for the transform: ASTRA toolbox, using CUDA, 2D
    
    FBPOper = odl.tomo.fbp_op(ray_trafo=ray_trafo, 
                             #filter_type='Ram-Lak', currently no filter
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
        self.reso = param.reso
        
        reco_space, ray_trafo, FBPOper = build_fanbeam_geometry(self.param)
        
        self.fbp = odl_torch.OperatorModule(FBPOper)

    def forward(self, x):
        x = self.fbp(x)
        return x

    def filter_function(self, x):
        x_filter = self.filter(x)
        return x_filter


class FanBeam2DProjector():
    def __init__(self, image_size, num_proj):

        self.image_size = image_size
        self.num_proj = num_proj
        self.start_angle = 0
        self.raw_reso = 0.7

        # Initialize required parameters for image, view, detector
        geo_param = Initialization_FanBeam(image_size=self.image_size, 
                                            num_proj=self.num_proj, 
                                            start_angle=self.start_angle,
                                            raw_reso=self.raw_reso)
        # Forward projection function
        self.forward_projector = Projection_FanBeam(geo_param)

        # Filtered back-projection
        self.fbp = FBP_FanBeam(geo_param)

    def forward_project(self, volume):
        '''
        Arguments:
            volume: torch tensor with input size (B, img_x, img_y)
        '''

        proj_data = self.forward_projector(volume)

        return proj_data

    def backward_project(self, projs):
        '''
        Arguments:
            projs: torch tensor with input size (B, num_proj)
        '''

        volume = self.fbp(projs)

        return volume

