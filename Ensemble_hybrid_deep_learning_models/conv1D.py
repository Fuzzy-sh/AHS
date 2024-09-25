


from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor, nn
from torch.autograd import Variable
import torch.jit as jit
from torch.nn import Parameter
import math
from attention import MultiHeadAttention



'''
1. Convolution & Pooling 1D
torch.nn.Conv1d(): 1D convolution
Input: (N, Fi, Li): basically, each input is Fi vectors of length Li
N: batch size
Fi: number of input filters (or channels)
Li: length of input sequence
Output: (N, Fo, Lo): each output is Fo vectors of length Lo
N: batch size
Fo: number of output filters (or channels)
Lo: length of output sequence


# case 1 - kernel size = 1
conv1d = nn.Conv1d(16, 32, kernel_size = 1)

x = torch.ones(128, 16, 10)   # input: batch_size = 128, num_filters = 16, seq_length = 10
print(conv1d(x).size())       # input and output size are equal when kernel_size = 1 (assuming no padding)
torch.Size([128, 32, 10])



# case 2 - kernel size = 2, stride = 1
conv1d = nn.Conv1d(16, 32, kernel_size = 2, padding = 2)

x = torch.ones(128, 16, 10)   # input: batch_size = 128, num_filters = 16, seq_length = 10
print(conv1d(x).size())

torch.Size([128, 32, 13])


# case 2 - kernel size = 2, stride = 2
conv1d = nn.Conv1d(16, 64, kernel_size = 2, stride = 2, padding = 2)

x = torch.ones(128, 16, 10)   # input: batch_size = 128, num_filters = 16, seq_length = 10
print(conv1d(x).size())

torch.Size([128, 64, 7])



conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=3, padding=1) # sequence_lengh /2
sequence_length_out = math.ceil(sequence_length /2)
relu = nn.LeakyReLU(negative_slope=0.01)
m=nn.MaxPool1d(3, stride=3, padding=1) # sequence_lengh /2

sequence_length_out = math.ceil(sequence_length_out /2)

fc=nn.Linear(sequence_length_out * in_channels, in_channels)
# Pass the input tensor through the Conv1D instance


'''

class conv_1d(jit.ScriptModule):
    def __init__(self, input_size, sequence_length, batch_first): 
        super(conv_1d, self).__init__()

        in_channels = input_size # 62
        out_channels = input_size * 2 # 124
        self.batch_first = batch_first
       

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=3, padding=1) # sequence_lengh /2
        sequence_length_out = math.ceil(sequence_length /2)
        
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.maxpool_1 = nn.MaxPool1d(2, stride=3, padding=1) # sequence_lengh /2
        sequence_length_out = math.ceil(sequence_length_out /2)
        
        self.fc = nn.Linear(sequence_length_out * in_channels, in_channels)  # Adjust output size to match the desired output shape [24, 62]
        
    @jit.script_method
    def forward(self, x):
        # type: (Tensor) -> Tensor

    
        if self.batch_first: # x [batch, seq_len, input_size]
            x=x.permute(0,2,1) # [batch, input_size , seq_len]
        else: # x [seq_len, batch, input_size]
            x=x.permute(1,2,0) # [batch, input_size , seq_len]
        
        # Apply convolutional layers
        x = self.conv1(x)  # [batch, input_size (-1) , seq_len]
    
        # attention_scores=0
        # x=> (batch, input_size, input_size)
        x = self.relu(x)
        
        x = self.maxpool_1(x) # [batch, input_size/2, input_size]
        
        # Reshape for fully connected layer
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layer
        x = self.fc(x)
        
       
        return x # [batch , input_size]
    


class conv_1d_withAttention(jit.ScriptModule):
    def __init__(self, input_size, sequence_length, batch_first, nhead): 
        super(conv_1d_withAttention, self).__init__()
        in_channels = input_size # 62
        out_channels = input_size *2  # 124
        self.batch_first = batch_first
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.multi_head_attention_1 = MultiHeadAttention(out_channels, n_head=nhead)
        self.fc = nn.Linear(out_channels * sequence_length, in_channels)  # Adjust output size to match the desired output shape [24, 62]

    @jit.script_method
    def forward(self, x):
        # type: (Tensor) -> Tuple[Tensor, Tensor]

    
        if self.batch_first: # x [batch, seq_len, input_size]
            x=x.permute(0,2,1) # [batch, input_size (-1) , seq_len]
        else: # x [seq_len, batch, input_size]
            x=x.permute(1,2,0) # [batch, input_size (-1) , seq_len]
        
        # Apply convolutional layers
        x = self.conv1(x)  # [batch, input_size (-1) , seq_len]
        # x = self.relu(x)
        x = x.permute(0,2,1) # [batch, seq_len, input_size]
     
        attn_weights, attention_scores = self.multi_head_attention_1(x, x, x, mask=None) # [batch_size, timesteps, feature_dim]
        
        # context= 
        # context = attn_weights.bmm(x.permute(0,2,1)) # [batch_size, timesteps, feature_dim] * [batch_size, feature_dim, timesteps] = [batch_size, timesteps, timesteps]
        # context= context.squeeze(1)
     
        
        attn_weights = attn_weights.view(attn_weights.size(0), -1)

        out= self.fc(attn_weights)
     
        return out, attention_scores # [batch , input_size]
    


    


