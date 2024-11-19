import numpy as np
import torch.nn as nn
import odl
from odl.contrib import torch as odl_torch

class Initialization_ConeBeam_modi:
    def __init__(self, image_size, cb_para):
        '''
        image_size: [z, x, y], assume x = y for each slice image
        proj_size: [h, w]

         cb_para=
         {
                'num_projs' :num_projs,
                'pixelSize' :pixelSize ,
                'voxelSize' : voxelSize,   #0.06400318mm,
                'Volumen_num_xz':Volumen_num_xz,
                'Volumen_num_y':Volumen_num_y,
                'SDD':  SDD,    #1281.909mm,
                'SOD' :SOD,   #  so=646.0335mm
                'detector_width':detector_width,
                'detector_height':detector_height,
                'ObjectOffsetX': 0,
                'ObjectRoll':0,
                'projectionzshift_pixel':0 ,
                'InitialAngle':0
        }
        '''
        self.param = {}
        self.image_size = image_size
        self.param['voxelSize'] = cb_para['voxelSize']

        ## Imaging object (reconstruction objective) with object center as origin
        self.param['nx'] = self.image_size[1]   #Volumen_num_xz
        self.param['ny'] = self.image_size[2]   #Volumen_num_xz
        self.param['nz'] = self.image_size[0]   #Volumen_num_y
        self.param['sx'] = self.param['nx'] * cb_para['voxelSize'] # volume real size
        self.param['sy'] = self.param['ny'] * cb_para['voxelSize'] # 32.768
        self.param['sz'] = self.param['nz'] * cb_para['voxelSize']

        ## Projection view angles (ray directions)
        self.param['start_angle'] = 0
        self.param['end_angle'] = self.param['start_angle'] + 2 * np.pi
        self.param['nProj'] = cb_para['num_projs']

        ## Detector
        self.param['nh'] = self.image_size[2]
        self.param['nw'] = self.image_size[1]

        # magnification m = SDD / SOD = 1281.909 / 646.0335 = 1.98427635719
        # pixel size = voxel size * magnifiactaion = 0.064 * 1.98427635719 = 0.12699368686

        self.param['sh'] = self.param['nh'] * cb_para['pixelSize']    #size of projection height 65
        self.param['sw'] = self.param['nw'] * cb_para['pixelSize']    # biggest width 65
        self.param['dde'] = cb_para['SDD'] - cb_para['SOD'] # distance between origin and detector center (assume in x axis)
        self.param['dso'] = cb_para['SOD'] # distance between origin and source (assume in x axis)



def build_conebeam_gemotry(param):
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
                                                 shape=[param.param['nh'], param.param['nw']])

    # Cone-beam geometry for 3D-2D projection
    geometry = odl.tomo.ConeBeamGeometry(apart=angle_partition, # partition of the angle interval
                                          dpart=detector_partition, # partition of the detector parameter interval
                                          src_radius=param.param['dso'], # radius of the source circle
                                          det_radius=param.param['dde'], # radius of the detector circle
                                          axis=[0, 0, 1]) # rotation axis is z-axis: (0, 0, 1)

    ray_trafo = odl.tomo.RayTransform(vol_space=reco_space, # domain of forward projector
                                     geometry=geometry, # geometry of the transform
                                     impl='astra_cuda') # implementation back-end for the transform: ASTRA toolbox, using CUDA, 2D or 3D

    FBPOper = odl.tomo.fbp_op(ray_trafo=ray_trafo,
                             #filter_type='Ram-Lak',
                             frequency_scaling=1.0)

    # Reconstruction space for imaging object, RayTransform operator, Filtered back-projection operator
    return reco_space, ray_trafo, FBPOper


# Projector
class Projection_ConeBeam(nn.Module):
    def __init__(self, param):
        super(Projection_ConeBeam, self).__init__()
        self.param = param
        # self.reso = param.reso

        # RayTransform operator
        reco_space, ray_trafo, FBPOper = build_conebeam_gemotry(self.param)

        # Wrap pytorch module
        self.trafo = odl_torch.OperatorModule(ray_trafo)

        self.back_projector = odl_torch.OperatorModule(ray_trafo.adjoint)

    def forward(self, x):
        x = self.trafo(x)
        return x

    def back_projection(self, x):
        x = self.back_projector(x)
        return x


# FBP reconstruction
class FBP_ConeBeam(nn.Module):
    def __init__(self, param):
        super(FBP_ConeBeam, self).__init__()
        self.param = param
        # self.reso = param.reso

        reco_space, ray_trafo, FBPOper = build_conebeam_gemotry(self.param)

        self.fbp = odl_torch.OperatorModule(FBPOper)

    def forward(self, x):
        x = self.fbp(x)
        x[x < 0] = 0 #normalize by cutting off negative values
        x[x > 1] = 1 #normalize by cutting of peaks that exceed the datarange
        return x

    def filter_function(self, x):
        x_filter = self.filter(x)
        return x_filter


class ConeBeam3DProjector():
    def __init__(self, image_size, cb_para):
        geo_param = Initialization_ConeBeam_modi(image_size=image_size,
                                                 cb_para=cb_para)

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














