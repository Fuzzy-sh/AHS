#!/bin/bash
# ====================================
#SBATCH --nodes=1
#SBATCH --mem=128GB
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=inclusion_exclusion_cpu
#SBATCH --output=inclusion_exclusion_cpu%A.out
#SBATCH --cpus-per-task=1
# ====================================

# Check the GPUs with the nvidia-smi command.



date
id


echo start initialization



which python
conda env list

# Print some job information

echo
echo "main preprocess file with task Id: $SLURM_ARRAY_TASK_ID"
echo "My hostname is: $(hostname -s)"
echo

# run the python program 

echo starting preprocess program python code for inclusion_exclusion

echo python inclusion_exclusion.py 

CUDA_LAUNCH_BLOCKING=1 python inclusion_exclusion.py 



echo ending slurm script to do preprocessing