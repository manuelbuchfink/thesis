detector_width=para_2d['detector_width']
angles=para_2d['num_projs']
voxelSize = para_2d['voxelSize']  #0.001    #6.40032e-05   #0.06400318mm
SDD =para_2d['SDD']    #'1085.6mm'   Distance Source to Detector
SOD= para_2d['SOD']   # 595.0mm'    Distance Source to Patient    SDD-SOD
ODD = SDD-SOD
# magnification=SDD/SOD
pixelSize = para_2d['pixelSize']  #float(voxelSize*magnification*0.6)

import astra  #module load cuda/11.8.0, use (mm)
vol_geom = astra.create_vol_geom(para_2d['Volumen_num_xz'], para_2d['Volumen_num_xz'])
# vol_geom =astra.create_vol_geom(512, 512, -256+axis_pixelposition, 256+axis_pixelposition,-256, 256)

# shift the angle to ctutils definition
angle_shift=1/4*2*np.pi
proj_angels= np.linspace(0+angle_shift, 2*np.pi+angle_shift, angles,endpoint=False)

# astra.geom_2vec()

##proj_geom['DetectorWidth']=pixelSize
proj_geom = astra.create_proj_geom('fanflat', pixelSize / voxelSize, detector_width, proj_angels, SOD / voxelSize, ODD / voxelSize)

# exit()

proj_geom = astra.geom_postalignment(proj_geom, cor_shift)

proj_id = astra.create_projector('line_fanflat', proj_geom, vol_geom)
# proj_id = astra.create_projector('strip_fanflat', proj_geom, vol_geom)

(sino_id, sino) = astra.create_sino(img, proj_id)

astra.data2d.delete(sino_id)
astra.projector.delete(proj_id)
# return np.fliplr(sino)   #as the ctutils

Volumen_num_xz=512
detector_width=1000
num_projs=1000*1
voxelSize = 0.6640625e-3*1000   #0.001    #6.40032e-05   #0.06400318mm
SDD = 1.0856*1000    #'1085.6mm'   Distance Source to Detector
SOD= 0.595*1000 #*0.4    # 595.0mm'    Distance Source to Patient    SDD-SOD
ODD = SDD-SOD
magnification=SDD/SOD
pixelSize = voxelSize*magnification

axis_pixelposition=100.5
ObjectOffsetX=(axis_pixelposition-0.5)*pixelSize/magnification

fb_para={
        'num_projs' :num_projs, # = num_proj
        'pixelSize' :pixelSize ,# =SH
        'voxelSize' : voxelSize,   #0.06400318mm, # SH X SW
        'Volumen_num_xz':Volumen_num_xz,
        'Volumen_num_y':1,
        'SDD':  SDD,    #1281.909mm, # DDE
        'SOD' :SOD,   #  so=646.0335mm #DSO
        'detector_width':detector_width,
        'detector_height':1,
        'ObjectOffsetX': ObjectOffsetX,
        'ObjectRoll':0,
        'projectionzshift_pixel':0 ,
        'InitialAngle':0
        }
'''
MAPPINGS

Volumen_num_xz          | proj_size?
Volumen_num_y           | ?
detector width          | ?
num_projs               | nProj
voxelSize               | sx * sy * sz?
SDD                     | (dso + dde)
SOD                     | dso
ODD                     | dde
magnification           | (dso + dde) / dso
pixelSize               | sh, sw?
axis_pixelposotion      | ?
ObjectOffsetX           | ?
ObjectRoll              | ?
projectionzshift_pixel  | ?
InitialAngle            | 0

'''
'''
Equal Definitions: (ctutils | my code)
        SDD | dso + dde = 1500
        SOD | dso  = 1000
        ODD | dde = 500
        magnification | (dso + dde) / ddo = 1500 / 1000 = 1.5
        num_projs | nProj
        voxelSize | nx * ny * nz (image voxel)
        pixel size | nx * ny
        Volumen_num_xz | proj_size (512)
        Volumen_num_y | 1
        detector_width | self.param['sw'] = np.sqrt(self.param['sx']**2 + self.param['sy']**2) * (1.5)
        proj_angles | odl.uniform_partition(min_pt=param.param['start_angle'], max_pt=param.param['end_angle'], shape=param.param['nProj'])
        proj_geom | odl.tomo.ConeBeamGeometry(apart=angle_partition, dpart=detector_partition, src_radius=param.param['dso'], det_radius=param.param['dde']) (fanflat)

Deviationg Definitions:
        initialAngle = 0; currently no angle shift of (1/4*2*np.pi) applied but could be done

Not specified Definitions:
        

        proj_id (not used with odl)
        (sino_id, sino) (not used with odl)

        axis_pixelposition (?)
        ObjextOffsetX (?)


'''