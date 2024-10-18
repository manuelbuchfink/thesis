#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --gpus=rtx4090:1
#SBATCH -o ./Reports/slurm-%j.out # Save the output to the Reports folder



# Ensure the conda environment is activated properly
source ~/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate NER

# Print Python path to verify it's using the correct environment
which python
echo "Python version: $(python --version)"

# Step 1: Run the first script to create the FBP volume
/zhome/ahmadfn/miniconda3/bin/python3 ner/create_fbp_volume_3d_single.py --config ner/configs/ct_recon_3d_single.yaml
wait
echo "FBP saved successfully"

# Step 2: Run the training script to train the priors
/zhome/ahmadfn/miniconda3/bin/python3 ner/train_ct_recon_3d_hdf5_single.py --config ner/configs/ct_recon_3d_single.yaml --id 0
wait
echo "Priors trained successfully"

# Step 3: Run the final step to finish the reconstruction
/zhome/ahmadfn/miniconda3/bin/python3 ner/finish_recon_3d_hdf5_single.py --config ner/configs/ct_recon_3d_single.yaml
wait
echo "Volume corrected successfully"