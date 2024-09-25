

from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor, nn
from torch.autograd import Variable
import torch.jit as jit
from torch.nn import Parameter
import math
from GRUcell import JitGRUCell, TimeJitGRUCell

from attention import MultiHeadAttention


# ----------------------------------------------------------------------------------------------------------------------
class JitGRULayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(JitGRULayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, x, hidden):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        inputs = x.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])

        for i in range(len(inputs)):
            
            hidden = self.cell(inputs[i], hidden)
            outputs += [hidden]

        
        return torch.stack(outputs), hidden # list of [25,32,62] and last output [32,62]
    

# ----------------------------------------------------------------------------------------------------------------------
class TimeJitGRULayer(jit.ScriptModule):
    
    # *cell_args:  a variable-length argument list (also known as varargs or positional arguments). It allows the constructor to accept
    # an arbitrary number of additional arguments, which will be passed to the cell object when it is initialized.
 
    
    def __init__(self, cell, *cell_args):
        
        super(TimeJitGRULayer, self).__init__()
        
        # the TimeJitGRUCell is instantiated withe the args sent to it [input_size, hidden_size]
        self.cell = cell(*cell_args)
    
        
    @jit.script_method
    def compute_delta(self,x, x_last_obsv, t, delta_last_obsv):
        # type: (Tensor, Tensor, int, Tensor) -> Tuple[Tensor, Tensor,Tensor]
        # dynamic featur starts from 1 , timestamps are in in the first columen
        
        
        
        x_t = x[:, 1:]# the size is [b,63] Remove the last column [b]
        
        # Extract s_t and s_t_1
        s_t = x[:, 0].unsqueeze(1)  # Extract last column [1,32,1]-->[32,1]
        
        # if t==0,x_t_1,s_t_1, and delta would be all zero
        x_t_1 = (x_last_obsv[:, 1:])* (torch.tensor(t >= 1)*1.0)
        s_t_1 = (x_last_obsv[:, 0].unsqueeze(1))* (torch.tensor(t >= 1)*1.0)
        
        
        # Compute mask for time t-1
        mask_t_1= x_t_1.ne(0).float()  # Create mask where elements with value 0 remain 0 and
        
        delta= (s_t - s_t_1 + delta_last_obsv * (1 - mask_t_1))* (torch.tensor(t >= 1)*1.0)
        max_value = delta.max()
        max_value = max_value + (max_value==0)*1.0
        delta= delta / max_value
        # Modify the return statement to use torch.tensor()
        return (x_t, x_t_1, delta)

    
    @jit.script_method
    def forward(self, x, hidden):
        
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]

        
         # in this line the number of cell, in the GRU model is exactly the same as the number of records.[1,64]
        inputs = x.unbind(0) # x is [25,32,64] -->   inputs is 25 [1,32,64]
       

        # allows you to provide type information about variables in your code for better readability
        # outputs is a list containing tensors (Tensor).[]: This initializes an empty list.
        # It serves as the initial value for the outputs variable.
        
        outputs = torch.jit.annotate(List[Tensor], [])

        # Initial computation for the first input
        x, x_last_obsv, delta = self.compute_delta(inputs[0],inputs[0], 0,torch.tensor(0))
        hidden = self.cell(x,x_last_obsv,delta, hidden) # cell (inputs[0]:[1,32,64] , h:[32,62]) --> [32,62]
        outputs += [hidden]  # there wil be 25 in the list of outputs for each record in chronologically
        
        # Computation for the remaining inputs
        for i in range(1,len(inputs)):
            x, x_last_obsv, delta = self.compute_delta(inputs[i], inputs[i-1],i, delta)
            hidden = self.cell(x,x_last_obsv,delta, hidden) # cell (inputs[0]:[1,32,64] , h:[32,62]) --> [32,62]
            outputs += [hidden]  # there wil be 25 in the list of outputs for each record in chronologically
        
        # convert the output to the tensors
        outputs = torch.stack(outputs)
        return outputs, hidden # list of [25,32,62] and [32,62]

# ----------------------------------------------------------------------------------------------------------------------

class JitGRU_EmbeddingLayer(jit.ScriptModule):

    __constants__ = ['hidden_size', 'num_layers', 'batch_first', 'layers']

    def __init__(self, seq_length, embedding_dim ,demographics_size, hidden_size, num_layers, batch_first, bias, output_size=3):
        super(JitGRU_EmbeddingLayer, self).__init__()
 
        assert bias

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.output_size = output_size
        self.demographics_size = demographics_size
        self.embedding_dim=embedding_dim
        self.seq_length=seq_length
        
        
        self.embedding = torch.nn.Embedding(self.seq_length,self.embedding_dim)
        
        if num_layers == 1:
            self.layers = nn.ModuleList([JitGRULayer(JitGRUCell, embedding_dim, hidden_size)])
        else:
            self.layers = nn.ModuleList([JitGRULayer(JitGRUCell, embedding_dim, hidden_size)] + [JitGRULayer(JitGRUCell, hidden_size, hidden_size)
                                                                                              for _ in range(num_layers - 1)])
        self.fc1 = nn.Linear( self.hidden_size + self.demographics_size , self.hidden_size  )
        self.fc2 = nn.Linear( self.hidden_size  , self.output_size )
        
        self.relu = nn.LeakyReLU(negative_slope=0.01)
       
    @jit.script_method
    def forward(self, x, d, h=None):

        # type: (Tensor,Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        output_states = jit.annotate(List[Tensor], [])


        # shape of x is [24, 65]
        # shape of d is [1, 24, 2]
        # Handle batch_first cases
     
        if self.batch_first:
            x = x.permute(1, 0) # --> [65, 24]
        
        
        x = x.to(torch.long)
        if h is None:
            batch_size = x.size(1)
            h = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=x.dtype, device=x.device)

        # the sequence size should be greater than the max of eventid, that is 65
        # Ensure indices are within the valid range
        # assert torch.max(embedding) < self.sequnce_size, "Index out of range"
        
        # print(self.sequnce_size) # print(torch.max(x)) # print(torch.min(x))
        
        embedding = self.embedding(x) # --> [65, 24, 62]

        output = embedding

        
      
        for i, rnn_layer in enumerate(self.layers):
            output, hidden = rnn_layer(output, h[i])
            output_states.append(hidden)

        # output=output[-1, :, :]
        
        d = d.expand(x.shape[0], -1, -1)        #  [1, 24, 2] --> [65, 24, 2]
        
        if self.batch_first:
            output = output.permute(1, 0, 2)
            d = d.permute(1, 0, 2) # [24, 30, 2]

        
        # [24, 65, 62]
        # [24, 65, 2]

        # final_output = torch.cat((output, d), dim=2) # --> [24, 65, 64]
        final_output=torch.cat((output, d), dim=2)
        
        out = self.relu(self.fc1(final_output))
  
        # output = output[-1]#.squeeze()
        out = self.fc2(out)
        # Reshape the tensor
        out_reshaped = out.reshape(x.shape[1], -1, self.output_size)[:, 0, :]

        log_probs = F.log_softmax(out_reshaped, dim=1)

        return log_probs, torch.stack(output_states)
    

# ----------------------------------------------------------------------------------------------------------------------

class GRU_Component(jit.ScriptModule): # GRU_Component recieves input without time  returens the output from the last time step and the output states with and without attention 
    __constants__ = ['hidden_size', 'num_layers', 'batch_first', 'layers']

    def __init__(self, input_size, demographics_size, hidden_size, num_layers, batch_first, bias, output_size, nhead):
        """
        Initialize the GRU with Attention and Demographics model.

        Parameters:
        - input_size: int, size of the input features
        - demographics_size: int, size of the demographic features
        - hidden_size: int, size of the hidden layer in the GRU cells
        - num_layers: int, number of GRU layers
        - batch_first: bool, if True, the input and output tensors are provided as (batch, seq, feature)
        - bias: bool, if True, adds a bias term to the linear layers
        - output_size: int, size of the output layer
        """


        super(GRU_Component, self).__init__()
        assert bias, "Bias term must be True"

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.output_size = output_size
        self.demographics_size = demographics_size
        self.nhead = nhead
        # Create GRU layers
        if self.num_layers == 1:
            self.layers = nn.ModuleList([JitGRULayer(JitGRUCell, self.input_size, self.hidden_size)])
        else:
            self.layers = nn.ModuleList([JitGRULayer(JitGRUCell, self.input_size, self.hidden_size)] + [JitGRULayer(JitGRUCell, self.hidden_size, self.hidden_size) for _ in range(self.num_layers - 1)])
        
        # Define the multi-head attention block
        self.multi_head_attention = MultiHeadAttention(input_size, n_head=nhead)

    @jit.script_method
    def forward(self, x, d,h=None):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        """
        Forward pass of the model.

        Parameters:
        - x: Tensor, input sequence data
        - d: Tensor, demographic data
        - h: Optional[Tensor], hidden state

        Returns:
        - output: Tensor, output from the last time step after attention
        - multi_head_output: Tensor, output from the multi-head attention block
        - attention_scores: Tensor, attention scores from the multi-head attention block
        """
       
        output_states = jit.annotate(List[Tensor], [])
      
        # Handle batch_first cases
        if self.batch_first:
            x = x.permute(1, 0, 2)  # Change to (seq_len, batch, input_size) if batch_first

        if h is None:
            batch_size = x.size(1)
            h = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
   
        output = x
        
        for i, rnn_layer in enumerate(self.layers):
            output, hidden = rnn_layer(output, h[i])  # Forward pass through GRU layers : return the list of hidden states and the last hidden state
            output_states.append(hidden)


        output_states=torch.stack(output_states) # hiddens



        # attention for output
        output_attenion = output.permute(1,0,2)  # [batch_size, , timesteps , feature_dim]
        output_attention_weights, output_attention_scores = self.multi_head_attention(
        output_attenion, output_attenion, output_attenion, mask=None)  # [batch_size, , timesteps , feature_dim]
        output_attention_weights=output_attention_weights.permute(1,0,2)  # [timesteps, batch_size, feature_dim]
        # attention for hidden states
        hidden_attenion = output_states.permute(1,0,2)  # [batch_size, , timesteps , feature_dim]
        hidden_attention_weights, hidden_attention_scores = self.multi_head_attention(
        hidden_attenion, hidden_attenion, hidden_attenion, mask=None)  # [batch_size, , timesteps , feature_dim]
        hidden_attention_weights=hidden_attention_weights.permute(1,0,2)  # [timesteps, batch_size, feature_dim]
       
 

        # Get the last time step output from the output_states
        output_last= output[-1, :, :]  # Get the last time step output from the output_states
        output_attention_last= output_attention_weights[-1, :, :]  # Get the last time step output from the output_states

        
        # RETURN THE OUTPUT FROM THE LAST TIME STEP, THE OUTPUT STATES, THE OUTPUT FROM THE MULTI-HEAD ATTENTION BLOCK, AND THE ATTENTION SCORES    
        return output_last, output_states, output_attention_last, hidden_attention_weights, output_attention_scores

    
# ----------------------------------------------------------------------------------------------------------------------
class GRU_D_Component(jit.ScriptModule): # GRU_Component recieves input with time returens the output from the last time step and the output states with and without attention
    __constants__ = ['hidden_size', 'num_layers', 'batch_first', 'layers']

    def __init__(self, input_size, demographics_size, hidden_size, num_layers, batch_first, bias, output_size, nhead):
        """
        Initialize the GRU with Attention and Demographics model.

        Parameters:
        - input_size: int, size of the input features
        - demographics_size: int, size of the demographic features
        - hidden_size: int, size of the hidden layer in the GRU cells
        - num_layers: int, number of GRU layers
        - batch_first: bool, if True, the input and output tensors are provided as (batch, seq, feature)
        - bias: bool, if True, adds a bias term to the linear layers
        - output_size: int, size of the output layer
        """


        super(GRU_D_Component, self).__init__()
        assert bias, "Bias term must be True"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.output_size = output_size
        self.demographics_size = demographics_size
        self.nhead = nhead
        # Define GRU layers
        self.layers = nn.ModuleList([TimeJitGRULayer(TimeJitGRUCell, self.input_size, self.hidden_size)])
        if self.num_layers > 1:
            self.layers.extend([TimeJitGRULayer(TimeJitGRUCell, self.hidden_size, self.hidden_size) for _ in range(self.num_layers - 1)])

        # Define the multi-head attention block
        self.multi_head_attention = MultiHeadAttention(input_size, n_head=nhead)

    @jit.script_method
    def forward(self, x, d, h=None):
        # type: (Tensor, Tensor,  Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        """
        Forward pass of the model.

        Parameters:
        - x: Tensor, input sequence data
        - d: Tensor, demographic data
        - h: Optional[Tensor], hidden state

        Returns:
        - output: Tensor, output from the last time step after attention
        - multi_head_output: Tensor, output from the multi-head attention block
        - attention_scores: Tensor, attention scores from the multi-head attention block
        """
        output_states = jit.annotate(List[Tensor], [])
      
        
        # Handle batch_first cases
        if self.batch_first:
            x = x.permute(1, 0, 2)  # Change to (seq_len, batch, input_size) if batch_first

        if h is None:
            h = torch.zeros(self.num_layers, x.shape[1], self.hidden_size, dtype=x.dtype, device=x.device)  # Initialize hidden state
        # else:
        #     h = h.unsqueeze(0)  # Add a dimension for the number of layers
        
        output = x

        for i, rnn_layer in enumerate(self.layers):
            output, hidden = rnn_layer(output, h[i])  # Forward pass through GRU layers
            output_states.append(hidden)



        output_states=torch.stack(output_states) # hiddens



        # attention for output
        output_attenion = output.permute(1,0,2)  # [batch_size, , timesteps , feature_dim]
        output_attention_weights, output_attention_scores = self.multi_head_attention(
        output_attenion, output_attenion, output_attenion, mask=None)  # [batch_size, , timesteps , feature_dim]
        output_attention_weights=output_attention_weights.permute(1,0,2)  # [timesteps, batch_size, feature_dim]
        # attention for hidden states
        hidden_attenion = output_states.permute(1,0,2)  # [batch_size, , timesteps , feature_dim]
        hidden_attention_weights, hidden_attention_scores = self.multi_head_attention(
        hidden_attenion, hidden_attenion, hidden_attenion, mask=None)  # [batch_size, , timesteps , feature_dim]
        hidden_attention_weights=hidden_attention_weights.permute(1,0,2)  # [timesteps, batch_size, feature_dim]
       
 

        # Get the last time step output from the output_states
        output_last= output[-1, :, :]  # Get the last time step output from the output_states
        output_attention_last= output_attention_weights[-1, :, :]  # Get the last time step output from the output_states

        
        # RETURN THE OUTPUT FROM THE LAST TIME STEP, THE OUTPUT STATES, THE OUTPUT FROM THE MULTI-HEAD ATTENTION BLOCK, AND THE ATTENTION SCORES    
        return output_last, output_states, output_attention_last, hidden_attention_weights, output_attention_scores

    