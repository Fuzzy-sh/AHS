import os
import glob
import sys
import argparse

# import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, MultiStepLR
# import torch.optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, MultiStepLR

from model import HybridModel
import tensorLoader as TL
import modelTrainer as MT
from utils import parse_args, print_gpu_info, get_device, get_optimizer, generate_filename, get_column_indices

from lossFunc import CosineSimilarityLoss, CombinedLoss
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"


# usage


def dataloader_size_in_gb(dataloader):
    total_bytes = 0
    for batch in dataloader:
        for sample in batch:
            sample_bytes = sum(p.element_size() * p.nelement() for p in sample)
            total_bytes += sample_bytes
    total_gb = total_bytes / (2**30)  # Convert bytes to GB
    return total_gb




def main(args):

   
    device = get_device()
    print(device)
    print_gpu_info()
    
    train_files = glob.glob(os.path.join(args.train_folder_path, args.file_pattern))
    test_files = glob.glob(os.path.join(args.test_folder_path, args.file_pattern))
    
    observation_windows_start_column, observation_windows_end_column, prediction_windows_start_column, prediction_windows_end_column, demographic_start_column, demographic_end_column, prediction_lable1, prediction_lable2,prediction_value1, prediction_value2=get_column_indices(args.pre_processing_method)
    print(args)
    loader = TL.TimeSeriesTensorDataLoader(args.train_type, args.test_type, train_files, args.train_observatoin_length, args.train_followup_length,
                                           args.train_stride, args.test_observatoin_length, args.test_followup_length,
                                           args.test_stride, args.train_batch_size, args.num_workers, test_files, args.test_batch_size,
                                           args.pin_memory, device, observation_windows_start_column, observation_windows_end_column, prediction_windows_start_column, prediction_windows_end_column, demographic_start_column, demographic_end_column, prediction_lable1, prediction_lable2,prediction_value1, prediction_value2, args.onClinicDemCol)

    concatenated_trainLoaders = loader.build_loader_train_set()
    concatenated_testLoaders = loader.build_loader_test_set()

    size_in_gb = dataloader_size_in_gb(concatenated_trainLoaders)
    print("Size of concatenated_trainLoaders in GB:", size_in_gb)
    
    size_in_gb = dataloader_size_in_gb(concatenated_testLoaders)
    print("Size of concatenated_testLoaders in GB:", size_in_gb)
    
    
    
    train_iter = iter(concatenated_trainLoaders)
    (input, label, demographic) = next(train_iter)

    _ ,  batch_size, seq_length, input_size = input.shape
    demographics_size = demographic.shape[-1]
    input_size=input_size-1
    print(f"Input Shape: {input.shape}\n"
      f"Label Shape: {label.shape}\n"
      f"Demographic Shape: {demographic.shape}\n"
      f"Batch Size: {batch_size}, Sequence Length: {seq_length}, Input Size: {input_size}")
    
    model= HybridModel(input_size, demographics_size, args.hidden_size, args.num_layers, args.batch_first, args.bias, args.num_classes, args.nhead, seq_length, args.components, args.num_GRUs)

    model.to(device)

    params = list(model.parameters())
    for param in params:
        param.required_grad = True

    optimizer = get_optimizer(args.optimizer, params, args.initial_lr, args.momentum)
    schedulers = {
        'ReduceLROnPlateau': ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.1, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08),
        'ExponentialLR': ExponentialLR(optimizer, gamma=0.1),
        'MultiStepLR': MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    }
    if args.scheduler not in schedulers:
        raise ValueError(f'Scheduler {args.scheduler} not recognized')
    scheduler = schedulers[args.scheduler]
    loss_function = nn.NLLLoss()
    # loss_function = CombinedLoss


    model_filename = generate_filename(args)
    print(model_filename)
    fpr_file = f"{args.dir_results}{model_filename}fpr.xlsx"
    tpr_file = f"{args.dir_results}{model_filename}tpr.xlsx"
    prec_file = f"{args.dir_results}{model_filename}prec.xlsx"
    rec_file = f"{args.dir_results}{model_filename}rec.xlsx"


    trainer = MT.modelTrainer(model, concatenated_trainLoaders, concatenated_testLoaders, optimizer, loss_function, scheduler, device, batch_size, args.hidden_size, args.num_classes, args.epochs, args.dir_checkpoints, args.dir_results, model_filename, fpr_file, tpr_file, prec_file, rec_file, args.components, args.num_GRUs)
    
    trainer.train()

if __name__ == "__main__":



    
    args = parse_args()
    main(args)



    #     ['subject_id', 'sex', 'age', 'start_date', 'visit_emr_MH_non_elect',
    #    'visit_emr_NonMH', 'visit_emr_visit', 'visit_family_gp',
    #    'visit_hospitalized_NonMH', 'visit_im', 'visit_neurology',
    #    'visit_other', 'visit_psychiatry', 'visit_hosp_visit',
    #    'visit_hospitalized_MH', 'visit_pharmacy', 'substance', 'mood',
    #    'anxiety', 'psychotic', 'cognitive', 'otherpsych', 'selfharm', 'CHF',
    #    'Arrhy', 'VD', 'PCD', 'PVD', 'HPTN_UC', 'HPTN_C', 'Para', 'OthND',
    #    'COPD', 'Diab_UC', 'Diab_C', 'Hptothy', 'RF', 'LD', 'PUD_NB', 'HIV',
    #    'Lymp', 'METS', 'Tumor', 'Rheum_A', 'Coag', 'Obesity', 'WL', 'Fluid',
    #    'BLA', 'DA', 'Alcohol', 'Drug', 'Psycho', 'Dep', 'Stroke', 'Dyslipid',
    #    'Sleep', 'IHD', 'Fall', 'Urinary', 'Visual', 'Hearing', 'Tobacco',
    #    'Delirium', 'MS', 'parkinsons', 'homeless', 'police_interaction',
    #    'events', 'events_id'

        


    # python main.py --train_observatoin_length 20 --train_followup_length 15 --train_stride 5 --test_observatoin_length 20 --test_followup_length 15 --test_stride 5 --train_batch_size 24 --test_batch_size 1 --train_type_input 0 --test_type_input 2 --hidden_size 62 --num_layers 1 --batch_first True --bias True --epochs 100 --pre_processing_input 2 --model_input 2
    
    # python main.py --train_observatoin_length 75 --train_followup_length 15 --train_stride 5 --test_observatoin_length 75 --test_followup_length 15 --test_stride 5 --train_batch_size 24 --test_batch_size 1 --train_type_input 0 --test_type_input 2 --hidden_size 62 --num_layers 1 --batch_first True --bias True --epochs 100 --pre_processing_input 1 --model_input 1

    # python main.py --train_observatoin_length 20 --train_followup_length 15 --train_stride 5 --test_observatoin_length 20 --test_followup_length 15 --test_stride 5 --train_batch_size 24 --test_batch_size 1 --train_type_input 0 --test_type_input 2 --hidden_size 62 --num_layers 1 --batch_first True --bias True --epochs 100 --pre_processing_input 2 --model_input 3
    # python main.py --train_observatoin_length 30 --train_followup_length 15 --train_stride 5 --test_observatoin_length 30 --test_followup_length 15 --test_stride 5 --train_batch_size 24 --test_batch_size 1 --train_type_input 0 --test_type_input 2 --hidden_size 62 --num_layers 1 --batch_first True --bias True --epochs 100 --pre_processing_input 2 --model_input 8
    

    #python main.py --train_observatoin_length 30 --train_followup_length 15 --train_stride 5 --test_observatoin_length 30 --test_followup_length 15 --test_stride 5 --train_batch_size 24 --test_batch_size 1 --train_type_input 0 --test_type_input 2 --hidden_size 62 --num_layers 1 --batch_first True --bias True --epochs 100 --pre_processing_input 2 --model_input 9


    #python main.py --train_observatoin_length 30 --train_followup_length 15 --train_stride 5 --test_observatoin_length 30 --test_followup_length 15 --test_stride 5 --train_batch_size 24 --test_batch_size 1 --train_type_input 0 --test_type_input 2 --hidden_size 62 --num_layers 1 --batch_first True --bias True --epochs 100 --pre_processing_input 2 --model_input 12
    

    #python main.py --train_observatoin_length 30 --train_followup_length 15 --train_stride 5 --test_observatoin_length 30 --test_followup_length 15 --test_stride 5 --train_batch_size 24 --test_batch_size 1 --train_type_input 0 --test_type_input 2 --hidden_size 62 --num_layers 1 --batch_first True --bias True --epochs 100 --pre_processing_input 2 --model_input 13