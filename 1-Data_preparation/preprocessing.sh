#!/bin/bash
# ====================================
#SBATCH --nodes=1
#SBATCH --mem=128GB
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=preprocess_cpu
#SBATCH --output=/work/messier_lab/preprocess_cpu_%A_%a.out
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

index=0
while read line ; do
        LINEARRAY[$index]="$line"
        index=$(($index+1))
done < preprocessing_params.txt

echo $((${SLURM_ARRAY_TASK_ID}-1))
echo ${LINEARRAY[$((${SLURM_ARRAY_TASK_ID}-1))]}

echo starting preprocess program python code for preprocess

echo python preprocessing.py ${LINEARRAY[$((${SLURM_ARRAY_TASK_ID}-1))]} 

CUDA_LAUNCH_BLOCKING=1 python preprocessing.py ${LINEARRAY[$((${SLURM_ARRAY_TASK_ID}-1))]} 



echo ending slurm script to do preprocessing