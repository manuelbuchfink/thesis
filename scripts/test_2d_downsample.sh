#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=rtx4090:1
#SBATCH --mem=24G

#SBATCH -o ./Reports/slurm-%j.out # STDOUT

source ~/.bashrc
export PATH=/miniconda3/bin:$PATH

conda deactivate
conda activate odl

module purge
module load python/3.11/pytorch/

srun --unbuffered --exclusive --gpus=rtx4090:1 --ntasks=1 --cpus-per-task=32 --mem=24G python ner/train_ct_recon_2d_hdf5_downsample.py --config ner/configs/ct_recon_2d.yaml

echo "File save complete!"