from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor, nn
from torch.autograd import Variable
import torch.jit as jit
from torch.jit import ScriptModule, script_method
from torch.nn import Parameter
import math

from GRUlayer import GRU_Component, GRU_D_Component
from attention import MultiHeadAttention
from conv1D import conv_1d, conv_1d_withAttention

# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# ----------------------------------------------------------------------------------------------------------------------

class HybridModel(ScriptModule):
    __constants__ = ['hidden_size', 'num_layers', 'batch_first']

    def __init__(self, input_size, demographics_size, hidden_size, num_layers, batch_first, bias, output_size, nhead, sequence_length, components, num_huddin):
        super(HybridModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.output_size = output_size
        self.demographics_size = demographics_size
        self.input_size = input_size
        self.nhead = nhead
        self.sequence_length = sequence_length
        self.components = components
        self.num_huddin = num_huddin

        # Initialize selected components
        # if 'GRU' in components:
        self.gru = GRU_Component(input_size, demographics_size, hidden_size, num_layers, batch_first, bias, output_size, nhead)
        # if 'GRU_D' in components:
        self.gru_d = GRU_D_Component(input_size, demographics_size, hidden_size, num_layers, batch_first, bias, output_size, nhead)
        # if 'GRU_Attention' in components:
        self.gru_attention = GRU_Component(input_size, demographics_size, hidden_size, num_layers, batch_first, bias, output_size, nhead)
        # if 'GRU_D_Attention' in components:
        self.gru_d_attention = GRU_D_Component(input_size, demographics_size, hidden_size, num_layers, batch_first, bias, output_size, nhead)
        # if 'Conv1D' in components:
        self.conv1d = conv_1d(input_size, sequence_length, batch_first)
        # if 'Conv1D_Attention' in components:
        self.conv1d_attention = conv_1d_withAttention(input_size, sequence_length, batch_first, nhead)

        self.fc1 = nn.Linear(hidden_size * len(components) + demographics_size, hidden_size * (len(components)))
        self.fc2 = nn.Linear(hidden_size * (len(components)) , output_size)
        self.relu = nn.LeakyReLU(negative_slope=0.01)

    @script_method
    def forward(self, x, d, h=None):
        # type: (Tensor, Tensor, Optional[List[Tensor]]) -> Tuple[Tensor, List[Tensor]]
        outputs = jit.annotate(List[Tensor], [])
        hidden_states = jit.annotate(List[Tensor], [])
        hidden_num=0
        
        
        if 'GRU' in self.components:


            
            gru_output, gru_hn, _, _, _ = self.gru(x[:, :, 1:], d, h[hidden_num] if h is not None else None)
            
            outputs.append(gru_output)
            hidden_states.append(gru_hn)
            hidden_num+=1

        if 'GR_D' in self.components:

            gru_d_output, gru_d_hn, _, _, _ = self.gru_d(x, d, h[hidden_num] if h is not None else None)
            
            outputs.append(gru_d_output)
            hidden_states.append(gru_d_hn)
            hidden_num+=1

        if 'G_A' in self.components:
            _,_,gru_attention_output, gru_attention_hn, _ = self.gru_attention(x[:, :, 1:], d, h[hidden_num] if h is not None else None)
            outputs.append(gru_attention_output)
            hidden_states.append(gru_attention_hn)
            hidden_num+=1

        if 'G_D_A' in self.components:
            _,_,gru_d_attention_output, gru_d_attention_hn, _ = self.gru_d_attention(x, d, h[hidden_num] if h is not None else None)
            outputs.append(gru_d_attention_output)
            hidden_states.append(gru_d_attention_hn)
            hidden_num+=1

        if 'Cv1D' in self.components:
            cnn_1d_output = self.conv1d(x[:, :, 1:])
            outputs.append(cnn_1d_output)

        if 'C1D_A' in self.components:
            cnn_1d_attention_output, _= self.conv1d_attention(x[:, :, 1:])
            outputs.append(cnn_1d_attention_output)

        # Concatenate all outputs

        outputs = torch.cat((outputs), dim=1)

      
        combined_output = torch.cat((outputs , d), dim=1)

        # Fully connected layers
        out = self.relu(self.fc1(combined_output))
   
        out = self.fc2(out)
      

        # Compute log probabilities
        log_probs = F.log_softmax(out, dim=1)
        
        return log_probs, hidden_states



# print(hidden_states)



# # Assume attention_scores is the attention matrix of size [24, 2, 30, 30]
# # Generate dummy data for illustration
# attention_scores = np.random.rand(24, 2, 30, 30)

# # Function to plot attention heatmaps
# def plot_attention_heatmaps(attention_scores, sequence_index, head_index):
#     scores = attention_scores[sequence_index, head_index, :, :]
    
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(scores, cmap='viridis', annot=True, fmt=".2f")
#     plt.xlabel('Key (Input Sequence)')
#     plt.ylabel('Query (Output Sequence)')
#     plt.title(f'Attention Heatmap for Sequence {sequence_index}, Head {head_index}')
#     plt.show()

# # Plotting for a specific sequence and attention head
# sequence_index = 0  # Choose the sequence in the batch
# head_index = 0      # Choose the attention head
# plot_attention_heatmaps(attention_scores, sequence_index, head_index)
