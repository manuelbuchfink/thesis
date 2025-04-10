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

echo "Waiting for setup to complete..."

#srun --unbuffered --exclusive --gpus=rtx4090:1 --ntasks=1 --cpus-per-task=32 --mem=24G python ner/setup.py --config ner/configs/ct_recon_2d.yaml

wait
echo "Setup completed!"
echo "Waiting for parallel job steps to complete..."
#srun --unbuffered --exclusive --gpus=rtx8000:1 --ntasks=1 --cpus-per-task=32 --mem=20G python ner/sparsify.py --config ner/configs/ct_recon_2d.yaml --start 0 --end 128 --id 0 &

#srun --unbuffered --exclusive --gpus=rtx4090:1 --ntasks=1 --cpus-per-task=32 --mem=24G python ner/train_ct_recon_2d_hdf5_parallel.py --config ner/configs/ct_recon_2d.yaml --start 0 --end 128 --id 0 &
#srun --unbuffered --exclusive --gpus=rtx4090:1 --ntasks=1 --cpus-per-task=10 --mem=5G python ner/train_ct_recon_2d_hdf5_parallel.py --config ner/configs/ct_recon_2d.yaml --start 128 --end 256 --id 1 &
#srun --unbuffered --exclusive --gpus=rtx4090:1 --ntasks=1 --cpus-per-task=10 --mem=5G python ner/train_ct_recon_2d_hdf5_parallel.py --config ner/configs/ct_recon_2d.yaml --start 256 --end 384 --id 2 &
#srun --unbuffered --exclusive --gpus=rtx4090:1 --ntasks=1 --cpus-per-task=10 --mem=5G python ner/train_ct_recon_2d_hdf5_parallel.py --config ner/configs/ct_recon_2d.yaml --start 384 --end 512 --id 3 &

wait
echo "All parallel job steps completed!"
echo "Waiting for file save to complete!"

srun --unbuffered --exclusive --gpus=rtx4090:1 --ntasks=1 --cpus-per-task=32 --mem=24G python ner/join_parallel_recon_2d_hdf5.py --config ner/configs/ct_recon_2d.yaml # <-- write script to combine above parallel processed data to one  output file

echo "File save complete!"