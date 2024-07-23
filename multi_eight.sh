#!/bin/bash

#SBATCH --nodes=1

#SBATCH --ntasks=4
#SBATCH --cpus-per-task=80
#SBATCH --gpus=rtx4090:4
#SBATCH --mem=20G

#SBATCH -o ./Reports/slurm-%j.out # STDOUT

source ~/.bashrc
export PATH=/miniconda3/bin:$PATH

conda deactivate
conda activate odl

module purge
module load python/3.11/pytorch/

srun --exclusive --gpus=rtx4090:1 --ntasks=1 --cpus-per-task=10 --mem=5G python ner/train_ct_recon_2d_hdf5.py --config ner/configs/ct_recon_2d.yaml --start 0 --end 128 &
srun --exclusive --gpus=rtx4090:1 --ntasks=1 --cpus-per-task=10 --mem=5G python ner/train_ct_recon_2d_hdf5.py --config ner/configs/ct_recon_2d.yaml --start 129 --end 256 &
srun --exclusive --gpus=rtx4090:1 --ntasks=1 --cpus-per-task=10 --mem=5G python ner/train_ct_recon_2d_hdf5.py --config ner/configs/ct_recon_2d.yaml --start 257 --end 384 &
srun --exclusive --gpus=rtx4090:1 --ntasks=1 --cpus-per-task=10 --mem=5G python ner/train_ct_recon_2d_hdf5.py --config ner/configs/ct_recon_2d.yaml --start 385 --end 512 &

echo "Waiting for parallel job steps to complete..."
wait
echo "All parallel job steps completed!"
# srun python ner/stitch_volume <-- write script to combine above parallel processed data to one  output file