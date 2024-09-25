import argparse
import torch
import numpy as np

def get_column_indices(pre_processing_method):
    # the 0 column is id; the 1 column is the label; the columns 1 and 2 are the demographic information; the column 3 is time stamps
    # the last column (-1) is event id, the column -2 is event names, the column -3 is and -4 are outcomes of interest in the prediction;
    
    prediction_lable1='homeless'
    prediction_lable2='police_interaction'
    prediction_value1=1
    prediction_value2=2
    demographic_start_column=1
    demographic_end_column=3
    prediction_windows_start_column=-4 
    prediction_windows_end_column=-2

    if pre_processing_method=="tensor_without_timestamps":
        observation_windows_start_column=4 
        observation_windows_end_column=-4
    elif pre_processing_method=="tensor_with_timestamps":
        observation_windows_start_column = 3 # time will be included in the observation window
        observation_windows_end_column=-4
    elif pre_processing_method=="sequnce_without_timestamps":
        observation_windows_start_column = -1 # for the sequnces, we only need the sequnces of the events.
        observation_windows_end_column = None
    else:
        raise ValueError(f"Invalid pre_processing_method: {pre_processing_method}")

    return (observation_windows_start_column, observation_windows_end_column, prediction_windows_start_column, prediction_windows_end_column, demographic_start_column, demographic_end_column, prediction_lable1, prediction_lable2,prediction_value1, prediction_value2)



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_observatoin_length', type=int, help='Length of the observation period during training')
    parser.add_argument('--train_followup_length', type=int, help='Length of the follow-up period during training')
    parser.add_argument('--train_stride', type=int, help='Stride length for training data')
    parser.add_argument('--test_observatoin_length', type=int, help='Length of the observation period during testing')
    parser.add_argument('--test_followup_length', type=int, help='Length of the follow-up period during testing')
    parser.add_argument('--test_stride', type=int, help='Stride length for testing data')
    parser.add_argument('--train_batch_size', type=int, help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, help='Batch size for testing')
    parser.add_argument('--train_type_input', type=int, help='Type of training input')
    parser.add_argument('--test_type_input', type=int, help='Type of testing input')
    parser.add_argument('--hidden_size', type=int, help='Size of the hidden layer')
    parser.add_argument('--num_layers', type=int, help='Number of layers in the model')
    parser.add_argument('--batch_first', type=bool, help='Whether the batch dimension comes first in the input data')
    parser.add_argument('--bias', type=bool, help='Whether to use bias in the model')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')

    parser.add_argument('--file_pattern', type=str, default="*.parquet", help='File pattern to match for data files')
    parser.add_argument('--train_folder_path', type=str, default="data/train/", help='Path to the training data folder')
    parser.add_argument('--test_folder_path', type=str, default="data/test/", help='Path to the testing data folder')
    parser.add_argument('--dir_checkpoints', type=str, default="model/", help='Directory to save model checkpoints')
    parser.add_argument('--dir_results', type=str, default="results/", help='Directory to save results')
    parser.add_argument('--model_input', type=int, default=0, help='Name of the model')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes in the output')
    parser.add_argument('--pin_memory', type=bool, default=False, help='Whether to pin memory in PyTorch')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--optimizer_input', type=int, default=0 , help='Optimizer to use')
    parser.add_argument('--initial_lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--scheduler_input', type=int, default=0, help='Learning rate scheduler')
    parser.add_argument('--pre_processing_input', type=int, default=0, help='Pre-processing method')
    parser.add_argument('--components', nargs='+', help='List of components', required=True)
    args = parser.parse_args()

    train_type_map = {0: "window_length", 1: "whole_length", 2: "whole_length_with_batch", 3: "window_length_fixed_startpoint"}
    test_type_map = {0: "oneWholeWindow", 1: "onClinicDem", 2: "batchWindow"}
    pre_processing_method_map = {0: "tensor_without_timestamps", 1: "sequnce_without_timestamps", 2: "tensor_with_timestamps" , 3: "sequnce_with_timestamps"}
    model_name_map = {0: "hybrid", 1: "GRU_EmbeddingLayer", 2: "GRU_with_timestamps", 3: "GRU_attention_with_timestamps", 4: "Atten_GRU_D_1l_CNN_parallel", 5:"Atten_GRU_D_Atten_1l_CNN_parallel", 6: "Atten_GRU_D_2l_CNN_parallel", 7:"Atten_GRU_D_Atten_2l_CNN_parallel",8: "Stack_1l_CNN_GRU_D_Atten", 9:"Stack_1l_CNN_Atten_GRU_D_Atten", 10: "Stack_2l_CNN_GRU_D_Atten", 11:"Stack_2l_CNN_Atten_GRU_D_Atten", 12: "Stack_GRU_D_Atten_1l_CNN", 13:"Stack_GRU_D_Atten_1l_CNN_Atten", 14: "Stack_GRU_D_Atten_2l_CNN", 15:"Stack_GRU_D_Atten_2l_CNN_Atten"}
    optimizer_map = {0: "SGD", 1: "Adam", 2: "AdamW", 3: "RMSprop", 4: "Adagrad", 5: "Adadelta", 6: "Adamax"}
    scheduler_map = {0: "ReduceLROnPlateau", 1: "ExponentialLR", 2: "MultiStepLR"}
    #components = ['GRU', 'GR_D', 'G_A', 'G_D_A', 'Cv1D', 'C1D_A']
    # Filter the components that contain 'G'
    components_with_hidden = [component for component in args.components if 'G' in component]

    # Count the number of components with 'G'
    args.num_GRUs = len(components_with_hidden)
    # components = ['GRU', 'GRU_D', 'GRU_Attention', 'GRU_D_Attention', 'Conv1D', 'Conv1D_Attention']

    args.train_type = train_type_map[args.train_type_input]
    args.test_type = test_type_map[args.test_type_input]
    args.pre_processing_method = pre_processing_method_map[args.pre_processing_input]
    args.model_name = model_name_map[args.model_input]
    args.optimizer = optimizer_map[args.optimizer_input]
    args.scheduler = scheduler_map[args.scheduler_input]
    args.onClinicDemCol= ['visit_emr_NonMH', 'visit_emr_MH_non_elect' , 'substance' , 'Drug']
    args.nhead=1

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

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    return args

def print_gpu_info():
    
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        print(f"Current GPU Index: {torch.cuda.current_device()}")
        for gpu_index in range(torch.cuda.device_count()):
            gpu_info = torch.cuda.get_device_properties(gpu_index)
            print(f"  GPU {gpu_index} Information:")
            print(f"  Name: {gpu_info.name}")
            print(f"  Total Memory: {gpu_info.total_memory / (1024 ** 3):.2f} GB")
            print(f"  CUDA Capability: {gpu_info.major}.{gpu_info.minor}")
            print(f"  Multiprocessors: {gpu_info.multi_processor_count}\n")
    else:
        print("No GPU available.")

def get_device():
    cuda_available = torch.cuda.is_available()
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if cuda_available else 'cpu'))
    device='cpu'

    return device

def get_optimizer(optimizer_name, parameters, lr, weight_decay=0.0):
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr)
    elif optimizer_name == 'adam':
        return torch.optim.Adam(parameters, lr=lr)
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr)
    elif optimizer_name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr)
    elif optimizer_name == 'adadelta':
        return torch.optim.Adadelta(parameters, lr=lr)
    elif optimizer_name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr)
    else:
        raise ValueError(f'Optimizer {optimizer_name} not recognized')


def generate_filename(args):
    return (f"{args.model_name}_"
            #f"TrainType_{args.train_type}_" # batchWindow
            f"TestType_{args.test_type}_"
            f"OL_{args.train_observatoin_length}_{args.test_observatoin_length}_"
            f"FL_{args.train_followup_length}__{args.test_followup_length}_"
            #f"S_{args.train_stride}_{args.test_stride}_" # stride is always 5
            f"BS_{args.train_batch_size}_{args.test_batch_size}_" # batch size is always 12
            #f"H_{args.hidden_size}_" # hidden size is always 62
            #f"L_{args.num_layers}_" # number of layers is always 1
            f"OptLR_{args.optimizer}_{args.initial_lr}_"  # optimizer is always SDG
            #f"Sch_{args.scheduler}_" # scheduler is always ReduceLROnPlateau
            #f"PrePrc_{args.pre_processing_method}_" # pre-processing method is always tensor_with_timestamps
            f"{'_'.join(args.components)}" # components are always GRU, GRU_D, GRU_Attention, GRU_D_Attention, Conv1D, Conv1D_Attention
            )