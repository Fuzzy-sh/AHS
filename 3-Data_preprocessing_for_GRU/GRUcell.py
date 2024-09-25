from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor, nn
from torch.autograd import Variable
import torch.jit as jit
from torch.nn import Parameter
import math



# ----------------------------------------------------------------------------------------------------------------------
class JitGRUCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):   
   
        super(JitGRUCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.weight_ih = Parameter(torch.Tensor(3 * self.hidden_size, self.input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * self.hidden_size, self.hidden_size))
        self.bias_ih = Parameter(torch.Tensor(3 * self.hidden_size ))
        self.bias_hh = Parameter(torch.Tensor(3 * self.hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    @jit.script_method
    def forward(self, x, hidden):
        # type: (Tensor, Tensor) -> Tensor
        
        # Remove extra dimension if present
        # if x.dim() == 3 and x.size(0) == 1:
        #     x = x.squeeze(0)
        # if hidden.dim() == 3 and hidden.size(0) == 1:
        #     hidden = hidden.squeeze(0)

        x = x.view(-1, x.size(1))
        hidden = hidden.to(x.device)



        x_results = torch.addmm(self.bias_ih, x, self.weight_ih.t())
        h_results = torch.addmm(self.bias_hh, hidden, self.weight_hh.t())
       

        i_r, i_z, i_n = x_results.chunk(3, 1)
        h_r, h_z, h_n = h_results.chunk(3, 1)

        r = torch.sigmoid(i_r.add(h_r))
        z = torch.sigmoid(i_z.add(h_z))
        n = torch.tanh(i_n.addcmul(r, h_n))


        return n.sub(n.mul(z)).addcmul(z, hidden)
    




class TimeJitGRUCell(jit.ScriptModule):

    def __init__(self, input_size, hidden_size):
        
        super(TimeJitGRUCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x_mean=0
        
        # 3 weights for update gate (i_z), reset gate (i_r), and new memory content (i_n) +
        self.weight_ih = Parameter(torch.Tensor(3 * self.hidden_size, self.input_size)) # [3*62, 64]
        
        # weights for decay rates as input
        self.weight_i_dg = Parameter(torch.Tensor(self.hidden_size, self.input_size))
        
        # 3 weights for update gate (h_z), reset gate (h_r), and new memory content (h_n)  weights for decay rates as hidden
        self.weight_hh = Parameter(torch.Tensor(3 * self.hidden_size, self.hidden_size)) # [3*62, 62]
        
        # weights for decay rates as hidden
        self.weight_h_dg = Parameter(torch.Tensor(self.hidden_size, self.hidden_size))

        # 3 biased for z, r ,n + decay rates gamma
        self.bias_ih = Parameter(torch.Tensor(3 * self.hidden_size))
        self.bias_hh = Parameter(torch.Tensor(3 * self.hidden_size))
        
        # biased for decay rates as input and hidden
        self.bias_i_dg = Parameter(torch.Tensor(self.hidden_size))
        self.bias_h_dg = Parameter(torch.Tensor(self.hidden_size))
        
        self.reset_parameters()

    # Parameters such as weight_ih, weight_hh, bias_ih, and bias_hh are typically randomly initialized to small values before training the model.
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    @jit.script_method
    def forward(self, x, x_last_obsv,delta,hidden):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor

        # print("x:",x.size())
        # print("x_last_obsv:",x_last_obsv.size())
        # print("delta:",delta.size())
        # print("hidden:",hidden.size())
        # if hidden.dim() == 3 and hidden.size(0) == 1:
        #     hidden = hidden.squeeze(0)
        hidden=hidden.to(x.device) # h:[1,32,62]# x : [1,32,64], hiddin: [32,62]

       # decay added based on the paper "Recurrent Neural Networks for Multivariate Time Series with Missing Values"
        #################################################
        
        #(10)
        gamma_x =torch.addmm(self.bias_i_dg, delta, self.weight_i_dg.t())
        # gamma_x = torch.mm(delta,self.weight_i_dg.t()) + self.bias_i_dg
        gamma_x = torch.exp(-torch.max(torch.zeros_like(gamma_x), gamma_x) ) #[b,input_size]
        gamma_h= torch.addmm(self.bias_h_dg, delta, self.weight_h_dg.t())
        # gamma_h = torch.mm(delta,self.weight_h_dg.t())+self.bias_h_dg
        gamma_h = torch.exp(-torch.max(torch.zeros_like(gamma_h),gamma_h) ) #[b,input_size]
        
        #(11)
        mask = x.ne(0).float()  # Create mask where elements with value 0 remain 0 and others are 1 [32,64]

        x = mask * x + (1 - mask) * (gamma_x * x_last_obsv + (1 - gamma_x) * torch.mean(x))

        #(12)Based on our work, we dont use the mask as there is no missing values, we we remove the mask from the following formula
        hidden = gamma_h * hidden

        # 13,14,15,16 --> in these formulas, mask has been given to the models, we did not.
        ###################################################

        # mm : matrix multiplication for the x input [32,64] * [64, 3*62] --> [32,3*62]
        # x_results = torch.mm(x, self.weight_ih.t()) + self.bias_ih  # (b*r,c) * [input_size, hidden_size]

         # mm : matrix multiplication for the hidden states  [32,62] * [62, 3*62] --> [32,3*62]
        # h_results = torch.mm(hidden, self.weight_hh.t()) + self.bias_hh
        x_results = torch.addmm(self.bias_ih, x, self.weight_ih.t())
        h_results = torch.addmm(self.bias_hh, hidden, self.weight_hh.t())

        i_r, i_z, i_n = x_results.chunk(3, 1) # [ch1,ch2,ch3] -- ch1: [32,62]
        h_r, h_z, h_n = h_results.chunk(3, 1)

        r = torch.sigmoid(i_r.add(h_r))
        z = torch.sigmoid(i_z.add(h_z))
        n = torch.tanh(i_n.addcmul(r, h_n))

        # relavant gated cell that remember "cat" as a singular and forget the rest and rememeber "was" for singular
        
        # r = torch.sigmoid(i_r + h_r) # [32,62]

        # # update gate <forget or remember
        # z = torch.sigmoid(i_z + h_z) # [32,62]

        # # new memory content [ tanh(i_n+ ((torch.sigmoid(i_r + h_r))*h_n)) : reset the hidden states for the parts that we dont want
        # # how relavant the the c<t-1> is to computing the next candidate for c<t>
        # n = torch.tanh(i_n + r * h_n) # [32,62]

        # torch.mul(n, z): The scaled candidate memory content based on the update gate.
        # n - torch.mul(n, z): removes the portion of the new candidate memory content that should be forgotten according to the update gate.
        # torch.mul(z, hidden): This performs element-wise multiplication between the update gate (z) and the previous hidden state (hidden).
        # This operation computes how much of the previous hidden state should be retained in the output based on the update gate.
        # + torch.mul(z, hidden): adds the portion of the previous hidden state that should be retained according to the update gate.
        # hidden_tilda= (n - torch.mul(n, z)) + torch.mul(z, hidden) # [32,62]
        return n.sub(n.mul(z)).addcmul(z, hidden)
