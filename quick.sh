#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=5G
#SBATCH --gpus=rtx4090:2

source ~/.bashrc
export PATH=/miniconda3/bin:$PATH

conda deactivate
conda activate odl

module purge
module load python/3.11/pytorch/

srun --export=ALL python ner/sample.py --config ner/configs/ct_recon_2d.yaml
