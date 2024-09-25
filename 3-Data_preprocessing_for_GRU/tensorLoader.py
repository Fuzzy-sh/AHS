
from typing import Dict, Any
import dask
import dask.dataframe as dd
import numpy as np
from tqdm import tqdm
from timeSeriesGenerator import TimeSeriesTensorGenerator
import torch
from torch.utils.data import (
    DataLoader, TensorDataset, ConcatDataset, WeightedRandomSampler
)

import torch.nn.functional as F
# from memory_profiler import profile

class TimeSeriesTensorDataLoader:

    def __init__(self, traintype, testtype, train_files, train_observatoin_length, train_followup_length, train_stride,
                 test_observatoin_length, test_followup_length, test_stride, train_batch_size, num_workers, test_files,
                 test_batch_size, pin_memory, device, observation_windows_start_column, observation_windows_end_column, prediction_windows_start_column, prediction_windows_end_column, demographic_start_column, demographic_end_column, prediction_lable1, prediction_lable2,prediction_value1, prediction_value2, onClinicDemCol):

        super(TimeSeriesTensorDataLoader, self).__init__()

        self.traintype = traintype
        self.testtype = testtype
        self.train_files = train_files
        self.test_files = test_files
        self.train_observatoin_length = train_observatoin_length
        self.train_followup_length = train_followup_length
        self.train_stride = train_stride

        self.test_observatoin_length = test_observatoin_length
        self.test_followup_length = test_followup_length
        self.test_stride = test_stride

        self.train_batch_size = train_batch_size
        self.num_workers = num_workers
        self.test_batch_size = test_batch_size
        self.pin_memory = pin_memory
        self.device = device
        self.trainLoader = {}
        self.ts_processor = TimeSeriesTensorGenerator(observation_windows_start_column, observation_windows_end_column, prediction_windows_start_column, prediction_windows_end_column, demographic_start_column, demographic_end_column, prediction_lable1, prediction_lable2,prediction_value1, prediction_value2 , onClinicDemCol)


    @staticmethod
    def create_weighted_sampler(targets: object) -> object:
        
        # Calculate the number of samples in each class
        class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
        print("Class Sample Count:", class_sample_count)
    
        # Calculate the weight for each class
        class_weights = 1.0 / class_sample_count
        print("Class Weights:", class_weights)
    
        # Assign weights to samples based on their class
        samples_weight = np.array([class_weights[t] for t in targets])
    
        # Convert numpy array to a PyTorch tensor
        samples_weight = torch.from_numpy(samples_weight)
    
        # Create a WeightedRandomSampler
        sampler = WeightedRandomSampler(samples_weight.type(torch.DoubleTensor), len(samples_weight))
        
        return sampler
    
    @staticmethod
    def custom_collate(batch: object) -> object:
    
        # Unpack the batch into inputs, labels, and demographics
        inputs, labels, demographics = zip(*batch)
        
        # Find the maximum length among input sequences
        max_len = max([l.shape[1] for l in inputs])
        
        # Pad the input sequences to have consistent lengths
        padded_inputs = torch.stack([F.pad(input[0], (0, 0, 0, max_len - len(input[0])), "constant", 0) for input in inputs])
        
        # Stack labels and demographics
        labels = torch.stack(labels).squeeze()
        demographics = torch.stack(demographics).squeeze(dim=1)
        
        return padded_inputs, labels, demographics
    
    
    def build_loader_train_set(self):
        
        train_loader = {}

        if self.traintype == 'window_length':
            train_loader = self._concatenate_train_loader_window_length()
    
        # elif self.type == 'whole_length':
        #     train_loader = self._concatenate_train_loader_whole_length()
    
        # elif self.type == 'whole_length_with_batch':
        #     train_loader = self._concatenate_train_loader_whole_length_with_batches()
    
        # elif self.type == 'window_length_fixed_startpoint':
        #     train_loader = self._concatenate_train_loader_window_length_fixed_startpoint()
    
        return train_loader
    
    def _concatenate_train_loader_window_length(self):
    
        train_loader = {}

        for train_file in self.train_files:
            train_ddf = dd.read_parquet(train_file, engine='pyarrow').compute()
            train_loader[train_file] = self.build_loader_train_windowlength(train_ddf)
            
            # print(train_file)
            # break
      
          

        return self._finalize_train_loader(train_loader)
    
    # def generate_tensors(self, ddf):
    #     for id, data in ddf.groupby('subject_id'):
    #         if len(data) > 1:
    #             x_subject, y_subject, demographics_subject = self.ts_processor.generate_tensors_by_windowlength(data, self.train_observatoin_length, self.train_followup_length, self.train_stride)
    #             x, y, demographics = map(np.concatenate, ([x_subject], [y_subject], [demographics_subject]))
    #     return x, y, demographics
    

    # @profile
    def build_loader_train_windowlength(self, ddf):

        
        # Initialize an empty list to store subject-specific data
        x_list = []
        y_list=[]
        demographics_list=[]
        
        ddf_groupedby_subject=ddf.groupby('subject_id')
        i=0
        # for id, data in tqdm(ddf_groupedby_subject):
        for id, data in ddf_groupedby_subject:
            
            if len(data)>1:
                
                # Iterate over subjects and append x_subject to the list
                x_subject, y_subject, demographics_subject = self.ts_processor.generate_tensors_by_windowlength(data, self.train_observatoin_length, self.train_followup_length, self.train_stride)
             
                # Append x,y, and demographics for subject to the list
                x_list.append(x_subject)
                y_list.append(y_subject)
                demographics_list.append(demographics_subject)

            # if i==500:
            #     break 
            # else:
            #     i+=1

        # Concatenating all subjects' data...'
    
        x,y,demographics = np.concatenate(x_list, axis=0),np.concatenate(y_list, axis=0),np.concatenate(demographics_list, axis=0)
    
        sampler=self.create_weighted_sampler(y)

        
        '# Converting to the tensor...'
        # print (x)
        x,y,demographics = torch.from_numpy(x),torch.from_numpy(y.astype(int)).long(),torch.from_numpy(demographics)
       
        '# Move to the device... '
        x,y,demographics=x.to(self.device),y.to(self.device),demographics.to(self.device)
        
        # print(f' The total size of the x dataset is : {x.size()}')
        # print(f' The total size of the y dataset is : {y.size()}')
        # print(f' The total size of the demographics dataset is : {demographics.size()}')
        
        '# Building the dataset...'
        dataset = torch.utils.data.TensorDataset(x,y.clone().detach(),demographics)
        
        '# Building the loader..'
        dataLoader = torch.utils.data.DataLoader(dataset = dataset, batch_size=self.train_batch_size,sampler=sampler, drop_last=True)
    
        # print (f' Each of the x sizes in loader is: {next(iter(dataLoader))[0].size()}')
        # print (f' Each of the y sizes in loader is: {next(iter(dataLoader))[1].size()}')
        # print (f' Each of the demographic in loader size is: {next(iter(dataLoader))[2].size()}')
        
        '# Returning back the dataLoader...'
        
        return dataLoader


    # @profile
    # def build_loader_train_windowlength(self, ddf): #, observatoin_length, followup_length, stride, batch_size):
        
    #     # Initialize an empty list to store subject-specific data
    #     x_list,y_list,demographics_list = [],[],[]
        
    #     ddf_groupedby_subject=ddf.groupby('subject_id')
  
    #     # for id, data in tqdm(ddf_groupedby_subject):
    #     for id, data in ddf_groupedby_subject:
            
    #         if len(data)>1:
    #             # Generate tensors by windowlength and append to the lists
    #             x_subject, y_subject, demographics_subject = self.ts_processor.generate_tensors_by_windowlength(data, self.train_observatoin_length, self.train_followup_length, self.train_stride)
    #             x_list.append(x_subject)
    #             y_list.append(y_subject)
    #             demographics_list.append(demographics_subject)
                
            # if i==100:
            #     break 
            # else:
            #     i+=1

    #     # Concatenating all subjects' data...'
    #     # x,y,demographics = np.concatenate(x_list, axis=0),np.concatenate(y_list, axis=0),np.concatenate(demographics_list)
    #     x, y, demographics = map(np.concatenate, (x_list, y_list, demographics_list))
    #     sampler=self.create_weighted_sampler(y)

    #     '# Converting to the tensor...'
      
    #     x, y, demographics = map(lambda arr: torch.from_numpy(arr).to(self.device).long(), (x, y.astype(int), demographics))

    #     '# Building the dataset...'
    #     dataset = torch.utils.data.TensorDataset(x,y.clone().detach(),demographics)
        
    #     '# Building the loader..'
    #     dataLoader = torch.utils.data.DataLoader(dataset = dataset, batch_size=self.train_batch_size,sampler=sampler, drop_last=True)

    #     '# Returning back the dataLoader...'
        
    #     return dataLoader
    
        
    def _finalize_train_loader(self, train_loader):
    
        train_loaders = ConcatDataset([[loader] for file in train_loader.keys() for loader in train_loader[file]])
        
        concatenated_train_loaders = DataLoader(train_loaders, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
    
        for i in concatenated_train_loaders:
            if i[0].shape[1] < self.train_batch_size:
                print("There are some batches with a size less than assumed")

        
        # Assuming next(iter(concatenated_train_loaders))[1] is your tensor
        tensor_to_check = next(iter(concatenated_train_loaders))[1]
        
        # Get unique values and their counts
        unique_values, counts = torch.unique(tensor_to_check, return_counts=True)

    
        print('the unique values in each batch and their counts')
        for value, count in zip(unique_values, counts):
            print(f"Value: {value}, Count: {count}")
    
        return concatenated_train_loaders


    # def build_loader_test(self, ddf):#, batch_size):
        
    #     # Initialize an empty list to store subject-specific data
    #     x_list = []
    #     y_list=[]
    #     demographics_list=[]
        
    #     # Iterate over subjects and append x_subject to the list
    #     x_subject, y_subject, demographics_subject = self.ts_processor.generate_tensors_for_test_set(ddf,
    #                                                                                                  self.train_observatoin_length,
    #                                                                                                  self.train_followup_length,
    #                                                                                                  self.train_stride)
        
    #     # Append x,y, and demographics for subject to the list
    #     x_list.append(x_subject)
    #     y_list.append(y_subject)
    #     demographics_list.append(demographics_subject)
    
    #     # Concatenating all subjects' data...'
    #     x = np.concatenate(x_list, axis=0)
    #     y = np.concatenate(y_list, axis=0)
    #     demographics = np.concatenate(demographics_list)


    #     '# Converting to the tensor...'
    #     x,y,demographics = torch.from_numpy(x),torch.from_numpy(y.astype(int)).long(),torch.from_numpy(demographics)


    #     '# Move to the device... '
    #     x,y,demographics=x.to(self.device),y.to(self.device),demographics.to(self.device)

    #     # print(f' The total size of the x dataset is : {x.size()}')
    #     # print(f' The total size of the y dataset is : {y.size()}')
    #     # print(f' The total size of the demographics dataset is : {demographics.size()}')
        
    #     '# Building the dataset...'
    #     dataset = torch.utils.data.TensorDataset(x,y.clone().detach(),demographics)
        
        
    #     '# Building the loader..'
    #     dataLoader = torch.utils.data.DataLoader(dataset = dataset, batch_size=self.test_batch_size)
    
    #     # print (f' Each of the x sizes in loader is: {next(iter(dataLoader))[0].size()}')
    #     # print (f' Each of the y sizes in loader is: {next(iter(dataLoader))[1].size()}')
    #     # print (f' Each of the demographic in loader size is: {next(iter(dataLoader))[2].size()}')
        
    #     '# Returning back the dataLoader...'
    #     return dataLoader

    # @profile
    def build_loader_test_set(self):
        
        test_loader = {}
        positive_H_label_count=0
        positive_P_label_count=0
        negative_label_count=0
        function = self.ts_processor.generate_oneWholeWindow_tensors_for_test_set
        pred_len = self.test_followup_length
        assert isinstance(self.testtype, object)
        if self.testtype == 'onClinicDem':
            function = self.ts_processor.generate_onClinicDem_tensor_for_test_set
            pred_len = self.test_followup_length
        elif self.testtype == 'batchWindow':
            function = self.ts_processor.generate_batchWindow_tensors_for_test_set
            pred_len = self.test_followup_length
        test_ddf = dd.read_parquet(self.test_files, engine='pyarrow')
        test_ddf = test_ddf.compute()
        ddf_groupedby_subject = test_ddf.groupby('subject_id')
        i=0


        for key, test_file in tqdm(ddf_groupedby_subject):
            
            x_list = []
            y_list = []

            demographics_list = []

            if len(test_file) > pred_len:
                
                x_subject, y_subject, demographics_subject = function(test_file, self.test_observatoin_length, self.test_followup_length, self.test_stride)

                if len(x_subject) > 0:

                    batch_size = len(x_subject)
                    
                    
                    # batch size is the number of the window returning from "generate_tensor_for_test_set"

                    # Append x,y, and demographics for subject to the list
                    x_list.append(x_subject)
                    y_list.append(y_subject)
                    demographics_list.append(demographics_subject)
                    # batch size is the number of the window returning from "generate_tensor_for_test_set"
                    # print(len(x_list),len(x_list[0]), x_list[0][0].shape) #tuple
                    # convert the lists to the tensor and send them to the deivce.
                    # print([i.shape for i in x_list[0]])
                    x = np.concatenate(x_list, axis=0)
                    y= np.concatenate(y_list, axis=0)
                    demographics=np.concatenate(demographics_list, axis=0)

                    x, y, demographics = torch.from_numpy(x), torch.from_numpy(y.astype(int)).long(), torch.from_numpy(
                        demographics)
                    x, y, demographics = x.to(self.device), y.to(self.device), demographics.to(self.device)

                    '# Building the dataset...'
                    # dataset = torch.utils.data.TensorDataset(x,y.clone().detach(),demographics)
                    dataset = torch.utils.data.TensorDataset(x, y.clone().detach(), demographics)
                    '# Building the loader..'
                   
                    positive_H_label_count += (y == 1).sum().item()
                    positive_P_label_count += (y == 2).sum().item()
                    negative_label_count += (y == 0).sum().item()

                    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True)

                    test_loader[key] = data_loader
                   
         
            
            # if i == 5000:
            
            #     break
            # else:
            #     i += 1

        test_loaders = ConcatDataset([[loader] for file in test_loader.keys() for loader in test_loader[file]])
        total = positive_H_label_count + positive_P_label_count + negative_label_count
        print(f"positive_H_label_count: {positive_H_label_count};   with percentage: {positive_H_label_count*100/total}")
        print(f"positive_P_label_count: {positive_P_label_count};   with percentage: {positive_P_label_count*100/total}")
        print(f"negative_label_count: {negative_label_count};       with percentage: {negative_label_count*100/total}")
     
        concatenated_test_loaders = DataLoader(test_loaders, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

        return concatenated_test_loaders
