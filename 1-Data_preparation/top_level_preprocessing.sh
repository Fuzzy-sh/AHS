#!/bin/bash


echo Starting Preprocessing from top level Fuzzy script
# print out the date and time for now. 
date

# Define the chunk size
CHUNK_SIZE=10000
# Define dataset names
# Define dataset names using an associative array
# Define dataset names using a simple array
dataset_names=("Claim" "NACRS" "DAD" "PIN" "Form10")


# Define paths
root_path="AHS_data/raw_data/"
file_name="list_sub_id.h5"

# Determine the total number of subjects in list_sub_id.h5
# total_subjects=$(h5ls -d ${dataset_path} | grep "total_subjects" | awk '{print $2}')



jobCounter=-1  # Correcting the starting point
echo "startig point for the jobcounter for main_GRU is: $jobCounter"

rm preprocessing_params.txt
# Loop over dataset_name_input values
for dataset_name_input in {0..4}

do
    # Get the dataset name
    dataset=${dataset_names[$dataset_name_input]}
    dataset_path="${root_path}${dataset}/$file_name"
    # Get the total number of subjects from the list_sub_id.h5 file


    TOTAL_SUBJECTS=`python get_total_subjects.py ${dataset_path}`
    echo ${TOTAL_SUBJECTS}
    # echo "python get_total_subjects.py ${dataset_path}"
    # TOTAL_SUBJECTS= $("python get_total_subjects.py ${dataset_path}")
    echo "Total subjects: $TOTAL_SUBJECTS"
    echo "Dataset: $dataset"
    echo "Dataset path: $dataset_path"
    # Calculate the number of full chunks
    NUM_FULL_CHUNKS=$((TOTAL_SUBJECTS / CHUNK_SIZE))
    # Loop over chunks
    for i in $(seq 0 $NUM_FULL_CHUNKS)
    do
        start_chunk=$((i * CHUNK_SIZE))
        end_chunk=$((start_chunk + CHUNK_SIZE))

        # If this is the last chunk, set end_chunk to -1 to indicate the end
        if [ $i -eq $NUM_FULL_CHUNKS ]; then
            end_chunk=${TOTAL_SUBJECTS}
        fi

        # Run your command with the specified hyperparameters
        echo "--start_chunck $start_chunk --end_chunck $end_chunk --dataset_name_input $dataset_name_input"
        echo "--start_chunck $start_chunk --end_chunck $end_chunk --dataset_name_input $dataset_name_input" >> preprocessing_params.txt

        jobCounter=$((jobCounter+1))
        
        # You would replace the echo command with the actual command you want to run
        # your_command --start_chunk $start_chunk --end_chunk $end_chunk --dataset_name_input $dataset_name_input
    done
done



echo "Number of the parameters created in preprocessing_params.txt are: $jobCounter"
echo ------------------------------------------------------------------------
# ########################################################################################
# Submit the job array
jobID1=$(sbatch --array=0-$jobCounter  preprocessing.sh)

echo $jobID1

