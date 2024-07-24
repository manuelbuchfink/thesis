#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=6
#SBATCH --gpus=rtx4090:6
#SBATCH --mem=18G

#SBATCH -o ./Reports/slurm-%j.out # STDOUT

source ~/.bashrc
export PATH=/miniconda3/bin:$PATH

conda deactivate
conda activate odl

module purge
module load python/3.11/pytorch/

echo "Waiting for setup to complete..."

srun --exclusive --gpus=rtx4090:1 --ntasks=1 --cpus-per-task=6 --mem=18G python ner/setup.py --config ner/configs/ct_recon_2d.yaml

wait
echo "Setup completed!"
echo "Waiting for parallel job steps to complete..."

srun --exclusive --gpus=rtx4090:1 --ntasks=1 --cpus-per-task=6 --mem=3G python ner/train_ct_recon_2d_hdf5_multi.py --config ner/configs/ct_recon_2d.yaml --start 0 --end 86 --id 0 &
srun --exclusive --gpus=rtx4090:1 --ntasks=1 --cpus-per-task=6 --mem=3G python ner/train_ct_recon_2d_hdf5_multi.py --config ner/configs/ct_recon_2d.yaml --start 86 --end 172 --id 1 &
srun --exclusive --gpus=rtx4090:1 --ntasks=1 --cpus-per-task=6 --mem=3G python ner/train_ct_recon_2d_hdf5_multi.py --config ner/configs/ct_recon_2d.yaml --start 172 --end 258 --id 2 &
srun --exclusive --gpus=rtx4090:1 --ntasks=1 --cpus-per-task=6 --mem=3G python ner/train_ct_recon_2d_hdf5_multi.py --config ner/configs/ct_recon_2d.yaml --start 258 --end 344 --id 3 &
srun --exclusive --gpus=rtx4090:1 --ntasks=1 --cpus-per-task=6 --mem=3G python ner/train_ct_recon_2d_hdf5_multi.py --config ner/configs/ct_recon_2d.yaml --start 344 --end 430 --id 4 &
srun --exclusive --gpus=rtx4090:1 --ntasks=1 --cpus-per-task=6 --mem=3G python ner/train_ct_recon_2d_hdf5_multi.py --config ner/configs/ct_recon_2d.yaml --start 430 --end 512 --id 5 &

wait
echo "All parallel job steps completed!"
echo "Waiting for file save to complete!"

srun --exclusive --gpus=rtx4090:1 --ntasks=1 --cpus-per-task=6 --mem=18G python ner/join_parallel_recon_2d_hdf5.py --config ner/configs/ct_recon_2d.yaml # <-- write script to combine above parallel processed data to one  output file

echo "File save complete!"