import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import odl
from odl.contrib import torch as odl_torch

class Initialization_ConeBeam_modi:
    def __init__(self, cb_para):
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
        self.param['voxelSize'] = cb_para['voxelSize']

        ## Imaging object (reconstruction objective) with object center as origin
        self.param['nx'] = cb_para['Volumen_num_xz']   # number of voxels
        self.param['ny'] = cb_para['Volumen_num_xz']
        self.param['nz'] = cb_para['Volumen_num_y'] # 1 != 128
        self.param['sx'] = self.param['nx'] * cb_para['voxelSize'] # volume real size
        self.param['sy'] = self.param['ny'] * cb_para['voxelSize'] # 32.768
        self.param['sz'] = self.param['nz'] * cb_para['voxelSize']

        ## Projection view angles (ray directions)
        self.param['start_angle'] = 0
        self.param['end_angle'] = self.param['start_angle'] + 2 * np.pi
        self.param['nProj'] = cb_para['num_projs']

        ## Detector
        # self.param['nh'] = cb_para['detector_height']
        # self.param['nw'] = cb_para['detector_width']

        # magnification m = SDD / SOD = 1281.909 / 646.0335 = 1.98427635719
        # pixel size = voxel size * magnifiactaion = 0.064 * 1.98427635719 = 0.12699368686

        self.param['nh'] =cb_para['detector_width']   # wrong code defination from nerp
        self.param['nw'] = cb_para['detector_height']
        self.param['sh'] = self.param['nh'] * cb_para['pixelSize']    #size of projection height 65
        self.param['sw'] = self.param['nw'] * cb_para['pixelSize']    # biggest width 65
        self.param['dde'] = cb_para['SDD'] - cb_para['SOD'] # distance between origin and detector center (assume in x axis)
        self.param['dso'] = cb_para['SOD'] # distance between origin and source (assume in x axis)

        print(cb_para['pixelSize'],cb_para['detector_width'] ,self.param['sh'])
        print(cb_para['voxelSize'],self.param['nz'],self.param['sz'])

# class Initialization_ConeBeam:
#     def __init__(self, image_size, num_proj, start_angle, proj_size, raw_reso=0.7):
#         '''
#         image_size: [z, x, y], assume x = y for each slice image
#         proj_size: [h, w]
#         '''
#         self.param = {}

#         self.image_size = image_size
#         self.num_proj = num_proj
#         self.proj_size = proj_size
#         self.raw_reso = raw_reso

#         self.reso = 512. / image_size[1] * raw_reso  # voxelsize

#         ## Imaging object (reconstruction objective) with object center as origin
#         self.param['nx'] = image_size[1]   # number of voxels
#         self.param['ny'] = image_size[2]
#         self.param['nz'] = image_size[0]
#         self.param['sx'] = self.param['nx']*self.reso  # volume real size
#         self.param['sy'] = self.param['ny']*self.reso
#         self.param['sz'] = self.param['nz']*self.reso

#         ## Projection view angles (ray directions)
#         self.param['start_angle'] = start_angle
#         self.param['end_angle'] = start_angle + np.pi
#         self.param['nProj'] = num_proj

#         ## Detector    mag=1500/1000
#         self.param['sh'] = self.param['sx']*(1500/1000)    #size of projection height
#         self.param['sw'] = np.sqrt(self.param['sx']**2+self.param['sy']**2)*(1500/1000)   # biggest width
#         self.param['nh'] = proj_size[0] # shape of sinogram is proj_size*proj_size
#         self.param['nw'] = proj_size[1]
#         self.param['dde'] = 500*self.reso # distance between origin and detector center (assume in x axis)
#         self.param['dso'] = 1000*self.reso # distance between origin and source (assume in x axis)

def build_conebeam_gemotry(param):
    # Reconstruction space:
    reco_space = odl.uniform_discr(min_pt=[-param.param['sx'] / 2.0, -param.param['sy'] / 2.0, -param.param['sz'] / 2.0],
                                    max_pt=[param.param['sx'] / 2.0, param.param['sy'] / 2.0, param.param['sz'] / 2.0],
                                    shape=[param.param['nx'], param.param['ny'], param.param['nz']],
                                    dtype='float32')

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
                             filter_type='Ram-Lak',
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
        # x = x / self.reso
        x = x / self.param.param['voxelSize']

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
        return x

    def filter_function(self, x):
        x_filter = self.filter(x)
        return x_filter


class ConeBeam3DProjector():
    # def __init__(self, image_size, proj_size, num_proj):
        # self.image_size = image_size
        # self.proj_size = proj_size
        # self.num_proj = num_proj
        # self.start_angle = np.pi / float(self.num_proj * 2.0) * (self.num_proj - 1.0)
        # self.raw_reso = 0.7

        # Initialize required parameters for image, view, detector
        # geo_param = Initialization_ConeBeam(image_size=self.image_size,
        #                                     num_proj=self.num_proj,
        #                                     start_angle=self.start_angle,
        #                                     proj_size=self.proj_size,
        #                                     raw_reso=self.raw_reso)
    def __init__(self, cb_para):
        geo_param = Initialization_ConeBeam_modi(cb_para)

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














