echo Starting Preprocessing from top level Fuzzy script
echo Today is 
date
echo ---------------------------------------------------

#############################################################################
# send the parameters and write them down to the main_GRU.txt file
# We send the parameters from the top level part of the program to know the number of the job array counter 


#Remove the file if it exists
rm params_main.txt



# echo Creating the params for params_main_GRU

train_observatoin_length=( 30 )
train_followup_length=( 15 )
train_stride=( 5 )
test_observatoin_length=( 30 )
test_followup_length=( 15 )
test_stride=( 5 )
train_batch_size=( 12 )
test_batch_size=( 1 )
train_type=( 0 )
test_type=( 1 )
hidden_size=( 62 )
num_layers=( 1 )
batch_first=( 1 )
bias=( 1 )

epochs=100

num_classes=( 3 )

optimizer=( 0 )
initial_lr=( 0.01 )


momentum=( 0.9 )
scheduler=( 0 )


pre_processing_input=( 2 )
model_input=( 2 3 )


echo "train_observatoin_length parameters are: ${train_observatoin_length[@]}"
echo "train_followup_length parameters are: ${train_followup_length[@]}"
echo "train_stride parameters are: ${train_stride[@]}"
echo "test_observatoin_length parameters are: ${test_observatoin_length[@]}"
echo "test_followup_length parameters are: ${test_followup_length[@]}"
echo "test_stride parameters are: ${test_stride[@]}"
echo "train_batch_size parameters are: ${train_batch_size}"
echo "test_batch_size parameters are: ${test_batch_size[@]}"
echo "type of training windowing is: ${train_type[@]}"
echo "type of test windowing is: ${test_type[@]}"
echo "hidden_size parameters are: ${hidden_size[@]}"
echo "num_layers parameters are: ${num_layers[@]}"
echo "batch_first parameters are: ${batch_first[@]}"
echo "bias parameters are: ${bias[@]}"
echo "initial_lr parameters are: ${initial_lr[@]}"
echo "epochs parameters are: ${epochs[@]}"
echo "pre_processing_input parameters are: ${pre_processing_input[@]}"
echo "model_input parameters are: ${model_input[@]}"
echo "num_classes parameters are: ${num_classes[@]}"
echo "optimizer parameters are: ${optimizer[@]}"
echo "momentum parameters are: ${momentum[@]}"
echo "scheduler parameters are: ${scheduler[@]}"


echo ---------------------------------------------------
# ##################################################################################
jobCounter=-1  # Correcting the starting point
echo "startig point for the jobcounter for main_GRU is: $jobCounter"



# for train_followup_length in "${train_followup_length[@]}" ; do
for train_stride in "${train_stride[@]}" ; do
# for test_observatoin_length in "${test_observatoin_length[@]}" ; do
# for test_followup_length in "${test_followup_length[@]}" ; do
for test_stride in "${test_stride[@]}" ; do

for test_batch_size in "${test_batch_size[@]}" ; do
for train_type in "${train_type[@]}" ; do
for test_type in "${test_type[@]}" ; do
for hidden_size in "${hidden_size[@]}" ; do
for num_layers in "${num_layers[@]}" ; do
for batch_first in "${batch_first[@]}" ; do 

for bias in "${bias[@]}" ; do

# for model_input in "${model_input[@]}" ; do
for num_classes in "${num_classes[@]}" ; do
# for initial_lr in "${initial_lr[@]}" ; do
for momentum in "${momentum[@]}" ; do
for scheduler in "${scheduler[@]}" ; do
# for pre_processing_input in "${pre_processing_input[@]}" ; do
for train_batch_size in "${train_batch_size[@]}" ; do 

for ((i = 0; i < ${#train_observatoin_length[@]}; i++)); do
for ((j = 0; j < ${#optimizer[@]}; j++)); do
for ((k = 0; k < ${#train_followup_length[@]}; k++)); do

    echo "--train_observatoin_length ${train_observatoin_length[i]} --train_followup_length ${train_followup_length[k]} --train_stride $train_stride --test_observatoin_length ${test_observatoin_length[i]} --test_followup_length ${test_followup_length[k]} --test_stride $test_stride --train_batch_size $train_batch_size --test_batch_size $test_batch_size --train_type_input $train_type --test_type_input $test_type --hidden_size $hidden_size --num_layers $num_layers --batch_first $batch_first --bias $bias --epochs $epochs --model_input ${model_input[i]} --num_classes $num_classes  --optimizer ${optimizer[j]} --initial_lr ${initial_lr[j]} --momentum $momentum --scheduler $scheduler --pre_processing_input ${pre_processing_input[i]}"

    echo "--train_observatoin_length ${train_observatoin_length[i]} --train_followup_length ${train_followup_length[k]} --train_stride $train_stride --test_observatoin_length ${test_observatoin_length[i]} --test_followup_length ${test_followup_length[k]} --test_stride $test_stride --train_batch_size $train_batch_size --test_batch_size $test_batch_size --train_type_input $train_type --test_type_input $test_type --hidden_size $hidden_size --num_layers $num_layers --batch_first $batch_first --bias $bias --epochs $epochs --model_input ${model_input[i]} --num_classes $num_classes  --optimizer ${optimizer[j]} --initial_lr ${initial_lr[j]} --momentum $momentum --scheduler $scheduler --pre_processing_input ${pre_processing_input[i]}" >> params_main.txt

    jobCounter=$((jobCounter+1))


done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
# done
# done
# done

# done
# done

echo "Number of the parameters created in params_main.txt are: $jobCounter"
echo ------------------------------------------------------------------------
# ########################################################################################
# # the max of submitting job arrays is 3999, so at each step, we will sumbit only 3999 jobs 
# # then sleep for about one hour till all the jobs are finished

jobID1=$(sbatch --array=0-$jobCounter  main.sh)
echo $jobID1

# echo ---------------------------------------------------------------------------
# echo "End of training xGboost"
# echo ---------------------------------------------------------------------------
# ###########################################################################################