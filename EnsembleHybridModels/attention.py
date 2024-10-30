
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor, nn
from torch.autograd import Variable
import torch.jit as jit
from torch.nn import Parameter
import math




# # that should be for each t in the observation window; [25,32,62] -->[32,62]
# class MultiHeadAttention(jit.ScriptModule):
    
#     def __init__(self, input_size:int, n_head:int, dropout:float=0.1):
#         super().__init__()

#         self.n_head = n_head # number of heads 
#         self.input_size = input_size # input size in our version without using embedding layer which is the number of dynamic features
#         # Make sure d_model is divisible by h
#         assert input_size % n_head == 0, "input_size is not divisible by h"
        
#         self.d_k = input_size//n_head # dimension of input size seen by each head
        
#         self.w_qs = nn.Linear(input_size, input_size, bias=False)
#         self.w_ks = nn.Linear(input_size, input_size, bias=False)
#         self.w_vs = nn.Linear(input_size, input_size, bias=False)
#         self.w_os = nn.Linear(input_size, input_size, bias=False)
#         self.dropout_percent=dropout
#         self.dropout = nn.Dropout(self.dropout_percent)

#     @jit.script_method
#     def attention(self, query, key, value, mask):
#     # type: (Tensor, Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]   
       
#         d_k = query.shape[-1]
#         # (batch, n_head, hidden_size, d_k) --> (batch, n_head, hidden_size, hidden_size)
#         # attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # scaled dot product
#         attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

#         if mask is not None:
#              # Write a very low value (indicating -inf) to the positions where mask == 0
#             attention_scores=attention_scores.masked_fill(mask == 0, -1e9)

#         attention_scores = attention_scores.softmax(dim=-1) # (batch, n_head, hidden_size, hidden_size) # Apply softmax to get the attention weights

#         # if self.dropout is not None:
#         #     attention_scores = self.dropout(attention_scores)

#         # (batch, h, hidden_size, hidden_size) --> (batch, h, hidden_size, d_k)
#         # return attention scores which can be used for visualization

#         output = (attention_scores @ value) # (batch, n_head, hidden_size, d_k)

#         return output, attention_scores
    
#     @jit.script_method
#     def forward(self, q, k, v, mask):
#         # type: (Tensor, Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]

#         query = self.w_qs(q) # (batch, hidden_size, hidden_size) --> (batch, hidden_size, hidden_size)
#         key = self.w_ks(k)   # (batch, hidden_size, hidden_size) --> (batch, hidden_size, hidden_size)
#         value = self.w_vs(v) # (batch, hidden_size, hidden_size) --> (batch, hidden_size, hidden_size)

#         # (batch, hidden_size, hidden_size) --> (batch, hidden_size, n_head, d_k) --> (batch, n_head, hidden_size, d_k)
       
#         query = query.view(query.shape[0], query.shape[1], self.n_head, self.d_k).transpose(1, 2)
    
#         key = key.view(key.shape[0], key.shape[1], self.n_head, self.d_k).transpose(1, 2)
#         value = value.view(value.shape[0], value.shape[1], self.n_head, self.d_k).transpose(1, 2)

#         # claculate attention 
#         x, attention_scores = self.attention(query, key, value, mask)

#         # combine all the heads together
#         # (batch, h, hidden_size, d_k) --> (batch, hidden_size, h, d_k) --> (batch, hidden_size, hidden_size)
#         x = x.transpose(1, 2).contiguous().view(x.shape[0],-1 , self.n_head * self.d_k)

#         # multiply by the output weights
#         # (batch, hidden_size, hidden_size) --> (batch, hidden_size, hidden_size)
       
#         attention_weights = self.w_os(x)
        
#         return attention_weights, attention_scores

class LayerNorm(jit.ScriptModule):
    """
    Layer Normalization
    """

    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    @jit.script_method
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

class ScaleDotProductAttention(jit.ScriptModule):
    
    """
    compute scale dot product attention

    Query : given input that we focused on (model's output)
    Key : every input to check relationship with Qeury(model's input)
    Value : every input same with Key (model's input)

    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    
    @jit.script_method
    def forward(self, q, k, v, mask=None, e=1e-12):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], float) -> Tuple[Tensor, Tensor]
        
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score


class MultiHeadAttention(jit.ScriptModule):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        self.layer_norm = LayerNorm(d_model)
        # self.dropout = nn.Dropout(0.1)
       

    @jit.script_method
    def forward(self, q, k, v, mask=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)
        
        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization
        # out = self.layer_norm(out)
        # out = out + out_
        # out = self.dropout(out)


        return out, attention
    
    @jit.script_method  
    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor
    
    @jit.script_method
    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
    


    