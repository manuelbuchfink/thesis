# Code Repository of Bachelor Thesis Streak artifact reduction in Sparse-view Computed Tomography using Deep Learning

Computed Tomography (CT) is a widely used imaging technology that
uses X-rays to gain insights into different materials in domains like medicine
and material science. Though effective, there are issues with the amount of X-
rays used in traditional CT scans, such as a large radiation load and long scan
durations. Thus, the sparse-view CT, which uses a significantly lower number
of X-rays, is a much sought-after improvement on the established procedure.
However, reconstructing images via sparse-view CT can result in streak arti-
facts in the final image. This can have several critical implications, such as
medical misdiagnoses or interpretation of structural damage in building el-
ements. The prior work by Kim et al. (2022) proposed a method for image
correction in medical image reconstruction with sparse-view CT and showed a
reduction in streak artifacts. However, this approach has four drawbacks: 1) It
is limited to single-image correction; 2) It has not been tested on data outside
of the medical domain; 3) The process is not optimized for time efficiency; 4)
The code is not publicly available. Motivated by the limitations mentioned
above, in this thesis, we adapt the image-correction approach proposed by
Kim et al. (2022) in medical single-image reconstruction with sparse-view CT
into image reconstruction for image volume data by re-implementing their
methodology. Further, we explore nine approaches to accelerate the image-
correction process with different beam geometries. Our results reveal several
viable acceleration methods. For example, for the Fan-Beam geometry, com-
bining multiple robust acceleration methods results in a much faster image
volume correction per image slice while maintaining high accuracy. We also
show that the method is particularly robust with the Cone-Beam geometry,
even for highly sparse projections.


Recreated Baseline Implemenation Fan-Beam correction:
  - train_ct_recon_2d_baseline.py
  - test_2d_baseline.sh

    
Final Implemenation Fan-Beam correction:
  - ner/train_ct_recon_2d_hdf5_multi.py
  - test_2d_everything.sh
    

Final Implementation Cone-Beam correction (ODL):
  -  train_ct_recon_3d_small.py
  -  test_3d_small.sh


Final Implementation Cone-Beam correction (ODL_ctutil):
  -  train_ct_recon_3d_hdf5_ctutils.py
  -  test_3d_ctutils.sh
    

Final Implementation Cone-Beam parallelization:
  -   train_ct_recon_3d_hdf5.py
  -   test_3d.sh
