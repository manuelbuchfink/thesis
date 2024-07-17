#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=20G
#SBATCH --gpus=rtx4090:1

source ~/.bashrc
export PATH=/miniconda3/bin:$PATH

conda deactivate
conda activate odl

module purge
module load python/3.11/pytorch/

srun --export=ALL python ner-copy-before-changing-to-projection-ssim/train_ct_recon_2d_hdf5.py --config ner-copy-before-changing-to-projection-ssim/configs/ct_recon_2d.yaml
