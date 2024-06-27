#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=7G
#SBATCH --gpus=rtx4090:1

source ~/.bashrc
export PATH=/miniconda3/bin:$PATH

conda deactivate
conda activate odl

module purge
module load python/3.11/pytorch/

srun --export=ALL python ner/train_ct_recon_2d_fast.py --config ner/configs/ct_recon_2d.yaml
