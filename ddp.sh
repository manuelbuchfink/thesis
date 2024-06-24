#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=10G
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=rtx4090:2

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source ~/.bashrc
export PATH=/miniconda3/bin:$PATH

conda deactivate
conda activate odl

module purge
module load python/3.11/pytorch/

srun --export=ALL python ner/ddp_train.py --config ner/configs/ct_recon_2d.yaml
#srun --export=ALL python ner/ddp_train.py --config ner/configs/ct_recon_2d.yaml