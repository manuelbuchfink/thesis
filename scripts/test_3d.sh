#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --gpus=rtx4090:4
#SBATCH -o ./Reports/slurm-%j.out # STDOUT

source ~/.bashrc
export PATH=/miniconda3/bin:$PATH

conda deactivate
conda activate odl

module purge
module load python/3.11/pytorch/

srun --exclusive --unbuffered --gpus=rtx4090:1 --ntasks=1 --cpus-per-task=32 --mem=24G python ner/create_fbp_volume_3d.py --config ner/configs/ct_recon_3d.yaml

wait
echo FBP saved

srun --exclusive --unbuffered --gpus=rtx4090:1 --ntasks=1 --cpus-per-task=32 --mem=24G python ner/train_ct_recon_3d_hdf5.py --config ner/configs/ct_recon_3d.yaml --id 0 &
srun --exclusive --unbuffered --gpus=rtx4090:1 --ntasks=1 --cpus-per-task=32 --mem=24G python ner/train_ct_recon_3d_hdf5.py --config ner/configs/ct_recon_3d.yaml --id 1 &
srun --exclusive --unbuffered --gpus=rtx4090:1 --ntasks=1 --cpus-per-task=32 --mem=24G python ner/train_ct_recon_3d_hdf5.py --config ner/configs/ct_recon_3d.yaml --id 2 &
srun --exclusive --unbuffered --gpus=rtx4090:1 --ntasks=1 --cpus-per-task=32 --mem=24G python ner/train_ct_recon_3d_hdf5.py --config ner/configs/ct_recon_3d.yaml --id 3 &

wait
echo Priors trained

srun --exclusive --unbuffered --gpus=rtx4090:1 --ntasks=1 --cpus-per-task=32 --mem=24G python ner/join_parallel_recon_3d_hdf5.py --config ner/configs/ct_recon_3d.yaml

wait
echo Volume Corrected