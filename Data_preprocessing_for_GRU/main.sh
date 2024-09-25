#!/bin/bash
# ====================================
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --gpus-per-node=h100:4
#SBATCH --ntasks-per-node=32
#SBATCH --gpus-per-node=1
#SBATCH --job-name=main_GPU
#SBATCH --output=main_GPU_%A_%a.out
#SBATCH --account=def-memacdon
# ====================================


# Check the GPUs with the nvidia-smi command.

nvidia-smi

date
id

echo start initialization


which python
conda env list

# Print some job information

echo
echo "mainGRU file with task Id: $SLURM_ARRAY_TASK_ID"
echo "My hostname is: $(hostname -s)"
echo

# run the python program 

index=0
while read line ; do
        LINEARRAY[$index]="$line"
        index=$(($index+1))
done < params_main.txt

echo $((${SLURM_ARRAY_TASK_ID}-1))
echo ${LINEARRAY[$((${SLURM_ARRAY_TASK_ID}-1))]}

echo starting main program python code for main

echo python main.py ${LINEARRAY[$((${SLURM_ARRAY_TASK_ID}-1))]} 

CUDA_LAUNCH_BLOCKING=1 python main.py ${LINEARRAY[$((${SLURM_ARRAY_TASK_ID}-1))]} 



echo ending slurm script to do training for main 