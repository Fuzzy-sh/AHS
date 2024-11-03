import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
from tqdm import tqdm
import time
import pandas as pd
from itertools import combinations
import torch.nn.functional as F




class modelTrainer:

    def __init__(self, model, train_loader, test_loader, optimizer, loss_function, scheduler, device, batch_size, hidden_size, num_classes, epochs, dir_checkpoints, dir_results, model_filename, fpr_file, tpr_file, prec_file, rec_file, components, num_huddin):
        
        self.model = model
        self.concatenated_trainLoaders = train_loader
        self.concatenated_testLoaders = test_loader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.scheduler = scheduler
        self.device = device
        self.dir_checkpoints = dir_checkpoints
        self.dir_results = dir_results
        self.num_classes=num_classes
        self.hidden_size=hidden_size
        self.batch_size=batch_size
        self.epochs=epochs
        self.model_filename=model_filename
        self.min_val_loss = np.Inf
        self.fpr_file=fpr_file
        self.tpr_file=tpr_file
        self.prec_file=prec_file
        self.rec_file=rec_file
        self.components=components
        self.num_huddin=num_huddin
        
    def train(self):
        # Training loop
        
        results= {'epoch':[],'training_loss':[], 'test_loss':[], 'test_accuracy':[], 'training_accuracy':[],
         'training_precision_noOutcome':[],'training_precision_homelessness':[],'training_precision_policeInteraction':[],
         'training_sensitivity_noOutcome':[],'training_sensitivity_homelessness':[],'training_sensitivity_policeInteraction':[],
         'test_precision_noOutcome':[],'test_precision_homelessness':[],'test_precision_policeInteraction':[],
         'test_sensitivity_noOutcome':[],'test_sensitivity_homelessness':[],'test_sensitivity_policeInteraction':[],
         'test_auc_noOutcome':[],'test_auc_homelessness':[],'test_auc_policeInteraction':[],
         'test_Pair':[],
         'test_roc_auc_p0':[],'test_roc_auc_p1':[],'test_roc_auc_p2':[],

        }
        
        for e in range(1, self.epochs + 1):
            
            start_time = time.time()
            running_results,hidden_list = self._train_one_epoch(e)
            self._print_metrics(running_results, 'train')
            elapsed_time = time.time() - start_time
            print(f"Training time: {elapsed_time:.2f} seconds")

            start_time = time.time()
            valing_results,fpr,tpr,roc_auc,prec, rec  = self._validate_one_epoch([hidden for hidden in hidden_list])
            val_loss = valing_results['test_loss'] / valing_results['steps']

            if val_loss < self.min_val_loss:
                self.min_val_loss = val_loss
                print(f'\nSaving the model with min loss of: {self.min_val_loss:.4f} as {self.model_filename}')
                torch.save(self.model.state_dict(), f"{self.dir_checkpoints}{self.model_filename}.pth")

            self._print_metrics(valing_results, 'test', True)
            elapsed_time = time.time() - start_time
            print(f"Validation time: {elapsed_time:.2f} seconds")

            results = self._save_results(e,results, running_results, valing_results, fpr, tpr,prec, rec, roc_auc, self.fpr_file, self.tpr_file, self.prec_file, self.rec_file)
            self.scheduler.step(val_loss)

    def _cosine_similarty_loss(self, preds, targets, inputs, num_classes=3):

                # Calculate TP, FP, FN for each class
        tp_masks = [(preds == targets) & (targets == i) for i in range(num_classes)]
        fp_masks = [(preds != targets) & (preds == i) for i in range(num_classes)]
        fn_masks = [(preds != targets) & (targets == i) for i in range(num_classes)]

        # Calculate cosine similarity losses for each class
        cos_similarities = []
        for i in range(1, num_classes):  # Skip class 0 for cosine similarity as TP
            inputs_tp = inputs[tp_masks[i]]
            inputs_fp = inputs[fp_masks[i]]
            inputs_fn = inputs[fn_masks[i]]
            
          # Check lengths and adjust
            min_len_fp = min(inputs_tp.size(0), inputs_fp.size(0))
            min_len_fn = min(inputs_tp.size(0), inputs_fn.size(0))

            if min_len_fp > 0:
                # Truncate to the smallest length
                inputs_tp_fp = inputs_tp[:min_len_fp]
                inputs_fp = inputs_fp[:min_len_fp]
                cos_sim_tp_fp = F.cosine_similarity(inputs_tp_fp, inputs_fp, dim=1).mean()
            else:
                cos_sim_tp_fp = torch.tensor(0.0)
            
            if min_len_fn > 0:
                # Truncate to the smallest length
                inputs_tp_fn = inputs_tp[:min_len_fn]
                inputs_fn = inputs_fn[:min_len_fn]
                cos_sim_tp_fn = F.cosine_similarity(inputs_tp_fn, inputs_fn, dim=1).mean()
            else:
                cos_sim_tp_fn = torch.tensor(0.0)

            # Aggregate cosine similarities with some weights
            alpha, beta = 0.5, 0.5
            cos_similarity_loss = cos_sim_tp_fp + (1-cos_sim_tp_fn)
            cos_similarities.append(cos_similarity_loss)

        # Combine all losses
        total_cos_similarity_loss = sum(cos_similarities) / (num_classes - 1)  # Average over the classes
        return total_cos_similarity_loss
    
    def _train_one_epoch(self, e):
        
        self.model.train()  
        
        running_results = {'loss': 0, 'accuracy': 0, 'steps': 0, 
                           'precision_noOutcome': 0, 'precision_homelessness': 0, 'precision_policeInteraction': 0, 
                           'sensitivity_noOutcome': 0, 'sensitivity_homelessness': 0, 'sensitivity_policeInteraction': 0,
                           'auc_noOutcome': 0, 'auc_homelessness': 0, 'auc_policeInteraction': 0}

        
        # Initialize variables to accumulate metrics across batches
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        actual_probabilities =[] #[[] for _ in range(num_classes)]  # List to accumulate actual probabilities
      
        train_bar = tqdm(self.concatenated_trainLoaders)
        
        # h = self.model.init_hidden(self.batch_size, self.device)
        
        weight = next(self.model.parameters()).data
        hidden_list=[]
        hidden_new_list=[]
        for i in self.components:
            hidden = weight.new(self.model.num_layers, self.batch_size, self.hidden_size).zero_()
            hidden_list.append(hidden)
            hidden_new=hidden.clone().detach().to(self.device)
            hidden_new_list.append(hidden_new)
    
        # hidden = torch.stack([torch.randn(self.batch_size, self.hidden_size) for _ in range(num_layers)])
       
        # with profile(with_stack=True, profile_memory=True ) as prof:
        #     with record_function("model_interface"):
        labels=[]
        
        for i, (inputs, actual_labels, demographics) in enumerate(train_bar):
            
            
            # h = h.data
            input, actual_label,demographic = inputs[0].clone().detach().float().requires_grad_(True), actual_labels[0].clone().detach(), demographics[0].clone().detach()
            # input, actual_label,demographic= input.to(self.device), actual_label.to(self.device),demographic.to(self.device)
          
            input=input.squeeze(-1)
            
            actual_label=actual_label.squeeze()
            demographic=demographic.squeeze()

            running_results['steps'] += 1
    
            # optimizer.zero_grad()
            self.optimizer.zero_grad(set_to_none=True)
            
           
            
            output,hidden_new = self.model(input, demographic, hidden_list)
            # Predicted labels
            ps = torch.exp(output)
            
            predicted_label=ps.max(dim=1)[1]
            # try:
            #     hidden_new=hidden_new[0]
                
            # except:
            #     pass
   
            # output, h = self.model.forward(input, h)
            # print(output)
            # loss = criterion(output, target, labels)
            loss = self.loss_function(output, actual_label)

            cosin_loss=self._cosine_similarty_loss(predicted_label, actual_label, input)
            loss+=cosin_loss
            loss.backward(retain_graph=True)
    
            hidden_list=[hidden_new.clone().detach().to(self.device) for hidden_new in hidden_new_list]
  
            # # Optionally, you can clip gradients using clip_grad_norm
            
            max_norm = 1.0  # Define the maximum gradient norm
            clip_grad_norm_(self.model.parameters(), max_norm)
    
            # loss.backward(retain_graph=False)
            self.optimizer.step()
            running_results['loss'] += loss.item()
    

            equality = (predicted_label == actual_label).float()
            
            # Update the confusion matrix for the batch
            batch_confusion_matrix = self._calculate_multi_class_confusion_matrix(actual_label, predicted_label)
            confusion_matrix += batch_confusion_matrix
    
            # Update accuracy and other metrics
            running_results['accuracy']+=equality.type(torch.FloatTensor).mean()
            train_bar.set_description(desc=f'Training - Epoch [{e}/{self.epochs}], Loss: {running_results["loss"] / running_results["steps"]:.3f}, Accuracy: {running_results["accuracy"] / running_results["steps"]:.3f}')
            
            # if running_results['steps'] ==100:
            #     break
            # actual_probabilities+=ps.tolist()
            # labels+=actual_labels.tolist()
                
            # if running_results['steps'] ==10:
    
            #     break
                    
        # num_rows=len(actual_probabilities)
        # actual_probabilities=np.array(actual_probabilities).reshape(self.batch_size,num_rows, self.num_classes)
        # print(actual_probabilities)    
        # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=3))

        
        # Calculate and print sensitivity, precision, and AUC-ROC for each class
        precisions, sensitivities, auc_roc_scores, fpr, tpr, roc_auc,prec, rec = self._calculate_metrics_for_multi_class(confusion_matrix)
        
        running_results['precision_noOutcome'],running_results['precision_homelessness'],running_results['precision_policeInteraction']= precisions[0],precisions[1],precisions[2]
        
        running_results['sensitivity_noOutcome'],running_results['sensitivity_homelessness'],running_results['sensitivity_policeInteraction']= sensitivities[0],sensitivities[1],sensitivities[2]
        
        # running_results['auc_noOutcome'],running_results['auc_homelessness'],running_results['auc_policeInteraction']= auc_roc_scores[0],auc_roc_scores[1],auc_roc_scores[2]
    

        # print(precisions, sensitivities)
        return running_results, hidden_list


    
    def _validate_one_epoch(self, hidden_list):
        
        self.model.eval()
        
        valing_results = {
            'test_loss': 0,'test_accuracy': 0,'min_loss': 0,'steps': 0,
            'precision_noOutcome': 0,'precision_homelessness': 0,'precision_policeInteraction': 0,
            'sensitivity_noOutcome': 0,'sensitivity_homelessness': 0,'sensitivity_policeInteraction': 0,
            'auc_noOutcome': 0, 'auc_homelessness': 0, 'auc_policeInteraction': 0,
        }
        
        # Initialize variables to accumulate metrics across batches
        
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        actual_probabilities = [] #[[] for _ in range(self.num_classes)]  # List to accumulate actual probabilities
        labels=[]
    
        with torch.no_grad():
            # with torch.inference_mode():
            
            test_bar = tqdm(self.concatenated_testLoaders, desc='Validation Results:')
            # hidden = torch.stack([torch.randn(1, self.hidden_size) for _ in range(num_layers)]).to(self.device)
            # weight = next(self.model.parameters()).data
    
            # hidden = weight.new(self.model.num_layers, 1, self.hidden_size).zero_().to(self.device)
            
            for i, (inputs, actual_labels, demographics) in enumerate(test_bar):


                valing_results['steps'] += 1
                
                # inputs, actual_labels,demographics = inputs[0].to(self.device), actual_labels[0].to(self.device), demographics[0].to(self.device)

                inputs, actual_labels, demographics = inputs[0].float(), actual_labels[0], demographics[0]

                # if len(inputs.size()) > 2 and inputs.size(-1) == 1:
                    # Remove the last dimension
                inputs = inputs.squeeze(-1)
                
                # print(actual_labels.size())
                # print(inputs.size())
                # print(demographics.size())
                
                # # Check for NaN values in input data
                # if torch.isnan(inputs).any():
                #     print("Input data contains NaN values.")
                    
                # # Check for extremely large values
                # if torch.abs(inputs).max() > 1e6:
                #     print("Input data contains extremely large values.")
                # hidden_list = [hidden.squeeze(0) if hidden.dim() == 3 and hidden.size(0) == 1 else hidden for hidden in hidden_list]
                # print(hidden_list[0].size())
         
                hidden_batch_size_list=[hidden.size(-2) for hidden in hidden_list]
                
                input_batch_size=inputs.size(0)
                


                # Calculate the difference in batch sizes
                batch_size_diff_list = [int(input_batch_size - hidden_batch_size) for hidden_batch_size in hidden_batch_size_list]
                
                for i, batch_size_diff in enumerate(batch_size_diff_list):
            
                    # Slice or concatenate based on batch size difference
                    if batch_size_diff < 0:
                        
                        # Shrink the hidden tensor
                        hidden_list[i] = hidden_list[i][ :, :input_batch_size, :]
                        
                    elif batch_size_diff > 0:
                        
                  
                        # Expand the hidden tensor with zeros
                        hidden_list[i] = torch.cat(
                            (hidden_list[i], torch.zeros(1, batch_size_diff, hidden_list[i].size(-1), device=hidden_list[i].device)),
                            dim=1)
                 
                        
                        

                # # Slice or concatenate based on batch size difference
                # if batch_size_diff < 0:
                #     # Shrink the hidden tensor
                #     hidden = hidden[ :input_batch_size, :]
                # elif batch_size_diff > 0:
                #     # Expand the hidden tensor with zeros
                #     hidden = torch.cat(
                #         (hidden, torch.zeros( batch_size_diff, hidden.size(1), device=hidden.device)),
                #         dim=0)
              
                output, hidden_list= self.model(inputs, demographics, hidden_list)
               
                loss = self.loss_function(output, actual_labels)
  
                
    
                # Predicted labels
                ps = torch.exp(output)
               
                predicted_labels=ps.max(dim=1)[1]
                
                equality = (predicted_labels == actual_labels).float()

                cosin_loss=self._cosine_similarty_loss(predicted_labels, actual_labels, inputs)
                loss+=cosin_loss
                
                valing_results['test_loss'] += loss.item()

               # Update the confusion matrix for the batch
                batch_confusion_matrix = self._calculate_multi_class_confusion_matrix(actual_labels, predicted_labels )
                confusion_matrix += batch_confusion_matrix
    
                valing_results['test_accuracy'] += equality.type(torch.FloatTensor).mean()
                test_bar.set_description(desc='Validation - Loss: %.3f, Accuracy: %.3f' % (valing_results['test_loss'] / valing_results['steps'],valing_results['test_accuracy'] / valing_results['steps']))
                
                actual_probabilities+=ps.tolist()
                labels+=actual_labels.tolist()
                
                
                # if valing_results['steps'] ==100:
                    
                #     break
                    
            
            num_rows=len(actual_probabilities)
            actual_probabilities=np.array(actual_probabilities).reshape(num_rows, self.num_classes)
            # Calculate and print sensitivity, precision, and AUC-ROC for each class
            
            precisions, sensitivities, auc_roc_scores, fpr, tpr, roc_auc, prec, rec = self._calculate_metrics_for_multi_class(confusion_matrix, actual_probabilities, labels)
            
            valing_results['precision_noOutcome'],valing_results['precision_homelessness'],valing_results['precision_policeInteraction']= precisions[0],precisions[1],precisions[2]
            valing_results['sensitivity_noOutcome'],valing_results['sensitivity_homelessness'],valing_results['sensitivity_policeInteraction']= sensitivities[0],sensitivities[1],sensitivities[2]
            valing_results['auc_noOutcome'],valing_results['auc_homelessness'],valing_results['auc_policeInteraction']= auc_roc_scores[0],auc_roc_scores[1],auc_roc_scores[2]

    
            return (valing_results,fpr,tpr,roc_auc, prec, rec )
            
    def _update_data(self, data_dict, values_dict, file_name):
        
        """
        Update the data dictionary with the values from values_dict and save to the Excel file.
        
        Parameters:
            data_dict (dict): Dictionary containing data for each class.
            values_dict (dict): Dictionary containing values for each class to be added.
            file_name (str): Name of the Excel file to save the data.

        """
       
        for key, arr in values_dict.items():
        # for key, arr in zip(['0','1','2'], values_dict.values()):
            # Check if the sheet already exists
            if key in data_dict:
                # If the sheet exists, create a new DataFrame with the next epoch values
                new_epoch_df = pd.DataFrame({f'epoch_{len(data_dict[key].columns) + 1}': arr})
                # # Concatenate the new DataFrame with the existing data
                data_dict[key] = pd.concat([data_dict[key], new_epoch_df], axis=1)
                # data_dict[key][f'epoch_{len(data_dict[key].columns) + 1}'] = arr
            else:
                # If the sheet doesn't exist, create a new DataFrame
                data_dict[key] = pd.DataFrame({f'epoch_1': arr})

        # Save the updated data to the Excel file
        # for sheet_name, df in data_dict.items():
        #     df.to_excel(file_name, sheet_name=sheet_name, index=False)
        
        with pd.ExcelWriter(file_name) as writer:
            for sheet_name, df in data_dict.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)


    def _save_results(self,e,results, running_results, valing_results, fpr,tpr,prec, rec,roc_auc, fpr_file, tpr_file, prec_file, rec_file ):
    
        results['training_loss'].append(running_results['loss']/running_results['steps'])
        results['test_loss'].append(valing_results['test_loss']/valing_results['steps'])
        results['test_accuracy'].append(valing_results['test_accuracy'].item()/valing_results['steps'])
        results['training_accuracy'].append(running_results['accuracy'].item()/running_results['steps'])
        
        
        results['training_precision_noOutcome'].append(running_results['precision_noOutcome'].item())
        results['training_precision_homelessness'].append(running_results['precision_homelessness'].item())
        results['training_precision_policeInteraction'].append(running_results['precision_policeInteraction'].item())
        
        results['training_sensitivity_noOutcome'].append(running_results['sensitivity_noOutcome'].item())
        results['training_sensitivity_homelessness'].append(running_results['sensitivity_homelessness'].item())
        results['training_sensitivity_policeInteraction'].append(running_results['sensitivity_policeInteraction'].item())
        
        results['test_precision_noOutcome'].append(valing_results['precision_noOutcome'].item())
        results['test_precision_homelessness'].append(valing_results['precision_homelessness'].item())
        results['test_precision_policeInteraction'].append(valing_results['precision_policeInteraction'].item())
        
        results['test_sensitivity_noOutcome'].append(valing_results['sensitivity_noOutcome'].item())
        results['test_sensitivity_homelessness'].append(valing_results['sensitivity_homelessness'].item())
        results['test_sensitivity_policeInteraction'].append(valing_results['sensitivity_policeInteraction'].item())
        
        results['test_auc_noOutcome'].append(valing_results['auc_noOutcome'].item())
        results['test_auc_homelessness'].append(valing_results['auc_homelessness'].item())
        results['test_auc_policeInteraction'].append(valing_results['auc_policeInteraction'].item())
       

        # Combine dictionaries into one

        # results['test_Pair'].append(list(roc_auc.keys()))
        
        # results['test_roc_auc_p0'].append(list(roc_auc.values())[0])
        # results['test_roc_auc_p1'].append(list(roc_auc.values())[1])
        # results['test_roc_auc_p2'].append(list(roc_auc.values())[2])
        
        
        # Convert dict_values to a list
        fpr_list = list(fpr.values())
        tpr_list = list(tpr.values())
        prec_list = list(prec.values())
        rec_list = list(rec.values())

        # print(tpr.values())

        
        # Create a dictionary with keys 'p0', 'p1', 'p2' and corresponding values from the list
        fpr_dict = {
            '0' : fpr_list[0],
            '1' : fpr_list[1],
            '2' : fpr_list[2]
        }
        
        tpr_dict = {
            '0' : tpr_list[0],
            '1' : tpr_list[1],
            '2' : tpr_list[2]
        }
        
        prec_dict = {
            '0' : prec_list[0],
            '1' : prec_list[1],
            '2' : prec_list[2]
        }
        
        rec_dict = {
            '0' : rec_list[0],
            '1' : rec_list[1],
            '2' : rec_list[2]
        }

        fpr_data = {}
        tpr_data = {}
        prec_data = {}
        rec_data = {}
        

        if e > 1:
            try:
                # Try reading the existing Excel file
                fpr_data = pd.read_excel(fpr_file, sheet_name=None)
                tpr_data = pd.read_excel(tpr_file, sheet_name=None)
                prec_data = pd.read_excel(prec_file, sheet_name=None)
                rec_data = pd.read_excel(rec_file, sheet_name=None)
                
            except FileNotFoundError:
                pass  # If the file doesn't exist, dictionaries remain empty
        

        self._update_data(fpr_data, fpr_dict, fpr_file)
        self._update_data(tpr_data, tpr_dict, tpr_file)
        self._update_data(prec_data, prec_dict, prec_file)
        self._update_data(rec_data, rec_dict, rec_file)

        # # Update the fpr_data dictionary with the next epoch values
        # for key, arr in zip(['0','1','2'], fpr_dict.values()):
        #     # Check if the sheet already exists
        #     if key in fpr_data:
                
        #         # If the sheet exists, create a new DataFrame with the next epoch values
        #         new_epoch_fpr_df = pd.DataFrame({f'epoch_{len(fpr_data[key].columns) + 1}': arr})
        #         # Concatenate the new DataFrame with the existing data
        #         fpr_data[key] = pd.concat([fpr_data[key], new_epoch_fpr_df], axis=1)
                
        #     else:
                
        #         # If the sheet doesn't exist, create a new DataFrame
        #         fpr_data[key] = pd.DataFrame({f'epoch_1': arr})

        # # Update the fpr_data dictionary with the next epoch values
        # for key, arr in zip(['0','1','2'], tpr_dict.values()):
        #     # Check if the sheet already exists
        #     if key in tpr_data:
  
        #         # If the sheet exists, create a new DataFrame with the next epoch values
        #         new_epoch_tpr_df = pd.DataFrame({f'epoch_{len(tpr_data[key].columns) + 1}': arr})
        #         # Concatenate the new DataFrame with the existing data
        #         tpr_data[key] = pd.concat([tpr_data[key], new_epoch_tpr_df], axis=1)
                
        #     else:
        #         # If the sheet doesn't exist, create a new DataFrame
        #         tpr_data[key] = pd.DataFrame({f'epoch_1': arr})
        
        # # Save the updated data to the Excel file
        # with pd.ExcelWriter(fpr_file) as writer:
        #     for sheet_name, df in fpr_data.items():
        #         df.to_excel(writer, sheet_name=sheet_name, index=False)

        # with pd.ExcelWriter(tpr_file) as writer:
        #     for sheet_name, df in tpr_data.items():
        #         df.to_excel(writer, sheet_name=sheet_name, index=False)


    # the dictionary of the results will be saved to a csv file
        data_frame=pd.DataFrame(
            data={
               
                'Training_Loss':results['training_loss'],
                'Test_Loss': results['test_loss'],
                'Test_Accuracy':results['test_accuracy'],
                'Training_Accuracy':results['training_accuracy'],
                
                'Training_precision_no_outcome': ["%.4f" % elem for elem in results['training_precision_noOutcome']],
                'Training_precision_homelessness': ["%.4f" % elem for elem in results['training_precision_homelessness']],
                'Training_precision_policeInteraction': ["%.4f" % elem for elem in results['training_precision_policeInteraction']],
                
                'Training_sensitivity_no_outcome': ["%.4f" % elem for elem in  results['training_sensitivity_noOutcome']],
                'Training_sensitivity_homelessness':   ["%.4f" % elem for elem in  results['training_sensitivity_homelessness']],
                'Training_sensitivity_policeInteraction':  ["%.4f" % elem for elem in  results['training_sensitivity_policeInteraction']],
                
                'Test_precision_no_outcome':  ["%.4f" % elem for elem in  results['test_precision_noOutcome']],
                'Test_precision_homelessness': ["%.4f" % elem for elem in results['test_precision_homelessness']],
                'Test_precision_policeInteraction': ["%.4f" % elem for elem in  results['test_precision_policeInteraction']],
                
                'Test_sensitivity_no_outcome': ["%.4f" % elem for elem in  results['test_sensitivity_noOutcome']],
                'Test_sensitivity_homelessness': ["%.4f" % elem for elem in  results['test_sensitivity_homelessness']],
                'Test_sensitivity_policeInteraction': ["%.4f" % elem for elem in  results['test_sensitivity_policeInteraction']],
    
                'Test_auc_no_outcome': ["%.4f" % elem for elem in  results['test_auc_noOutcome']],
                'Test_auc_homelessness': ["%.4f" % elem for elem in  results['test_auc_homelessness']],
                'Test_auc_policeInteraction': ["%.4f" % elem for elem in  results['test_auc_policeInteraction']],

            },
            
        index=range(1,e+1)

        )
        
        data_frame.to_csv(f"{self.dir_results}{self.model_filename}.csv",  index_label="Epoch")
        # data_frame.to_csv(f"{self.dir_results}test.csv",  index_label="Epoch")


        return results
    
    
    def _print_metrics(self,results, dataset_name, auc_dict=None):
    
        # Create dictionaries to store precision and sensitivity values
  
        precision_dict = {
            'NoOutcome': results['precision_noOutcome'] ,
            'homelessness': results['precision_homelessness'] ,
            'policeInteraction': results['precision_policeInteraction'] 
        }
        
        sensitivity_dict = {
            'NoOutcome': results['sensitivity_noOutcome'] ,
            'homelessness': results['sensitivity_homelessness'] ,
            'policeInteraction': results['sensitivity_policeInteraction'] 
        }
    
        
        # Print precision values
        print(f"{dataset_name}_precisions:", end=" ")
        
        for label, precision in precision_dict.items():
            print(f"{label}: {precision:.3f},", end=" ")
        
        # Print sensitivity values
        print(f"\n{dataset_name}_sensitivity:", end=" ")
        
        for label, sensitivity in sensitivity_dict.items():
            print(f"{label}: {sensitivity:.3f},", end=" ")

        # Print auc values if auc_dict is provided
        if auc_dict:
            
            auc_dict = {
            'NoOutcome': results['auc_noOutcome'] ,
            'homelessness': results['auc_homelessness'] ,
            'policeInteraction': results['auc_policeInteraction'] 
                
            }
            
            # Print auc values
            print(f"\n{dataset_name}_auc:", end=" ")
            
            for label, auc in auc_dict.items():
                print(f"{label}: {auc:.3f},", end=" ")
            
        print()  # Print a new line to end the line
    
    
    def _calculate_multi_class_confusion_matrix(self,actual_labels, predicted_labels):
        
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        
        actual_labels=actual_labels.cpu()
        predicted_labels=predicted_labels.cpu()
       
        for actual, predicted in zip(actual_labels, predicted_labels):
            
            confusion_matrix[actual, predicted] += 1
    
        return confusion_matrix
    
    
    def _calculate_metrics_for_multi_class(self, confusion_matrix, actual_probabilities = None, actual_labels=None):
    
        # Initialize lists to store per-class metrics
        precisions = []
        sensitivities = []
        auc_roc_scores = []
        
        fpr={}
        tpr={}
        prec={}
        rec={}
        roc_auc={}
        
        for i in range(self.num_classes):
            
            TP = confusion_matrix[i, i]
            FP = np.sum(confusion_matrix[:, i]) - TP
            FN = np.sum(confusion_matrix[i, :]) - TP
    
            # Calculate precision for class i
            precision_i = TP / (TP + FP)
            precisions.append(precision_i)
    
            # Calculate sensitivity (recall) for class i
            sensitivity_i = TP / (TP + FN)
            sensitivities.append(sensitivity_i)
            
            # Calculate AUC-ROC for class i
            if actual_probabilities is not None and actual_probabilities.size > 0:
                
                actual_probabilities_i = actual_probabilities[:,i]
                actual_probabilities_i = np.nan_to_num(actual_probabilities_i, nan=0)
                
                auc_roc_i = roc_auc_score([1 if label == i else 0 for label in actual_labels], actual_probabilities_i)
                fpr[str(i)], tpr[str(i)], _ = roc_curve([1 if label == i else 0 for label in actual_labels], actual_probabilities_i)
                prec[str(i)], rec[str(i)], _ = precision_recall_curve([1 if label == i else 0 for label in actual_labels], actual_probabilities_i)
              
                
                auc_roc_scores.append(auc_roc_i)
    
        #using one-vs-one strategy (2)
        # for i, j in combinations(range(self.num_classes), 2):

        #     if actual_probabilities is not None and actual_probabilities.size > 0:
        #         key = (i, j)

        #         fpr[key], tpr[key], _ = roc_curve([1 if label == i else 0 for label in actual_labels], actual_probabilities[:, j] - actual_probabilities[:, i])
                
        #         roc_auc[key] = auc(fpr[key], tpr[key])
    
        return precisions, sensitivities, auc_roc_scores, fpr, tpr, roc_auc, prec, rec
