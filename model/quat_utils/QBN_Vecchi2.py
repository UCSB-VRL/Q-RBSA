# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 00:39:36 2020

@author: Edoardo
"""

import torch
from torch.nn import Module, Parameter
# import math
from einops import rearrange

def moving_average_update(statistic, curr_value, momentum):
    
    new_value = (1 - momentum) * statistic + momentum * curr_value
    
    return  new_value.data

class QuaternionBatchNorm2d(Module):
    r"""Applies a 2D Quaternion Batch Normalization to the incoming data.
        """

    def __init__(self, num_features, gamma_init=1., beta_param=True, momentum=0.1):
        super(QuaternionBatchNorm2d, self).__init__()
        self.num_features = num_features // 4
        self.gamma_init = gamma_init
        self.beta_param = beta_param
        self.gamma = Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)
        self.eps = torch.tensor(1e-5)
        
        self.register_buffer('moving_var', torch.ones(1) )
        self.register_buffer('moving_mean', torch.zeros(4))
        self.momentum = momentum

    def reset_parameters(self):
        self.gamma = Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)

    def forward(self, input):
        # print(self.training)
        #import pdb; pdb.set_trace()
        if self.training:
            quat_components = torch.chunk(input, 4, dim=1)
    
            r, i, j, k = quat_components[0], quat_components[1], quat_components[2], quat_components[3]
            
            mu_r = torch.mean(r)
            mu_i = torch.mean(i)
            mu_j = torch.mean(j)
            mu_k = torch.mean(k)
            mu = torch.stack([mu_r,mu_i, mu_j, mu_k], dim=0)
            # print('mu shape', mu.shape)
            
            delta_r, delta_i, delta_j, delta_k = r - mu_r, i - mu_i, j - mu_j, k - mu_k
    
            quat_variance = torch.mean(delta_r**2 + delta_i**2 + delta_j**2 + delta_k**2)
            var = quat_variance
            denominator = torch.sqrt(quat_variance + self.eps)
            
            # Normalize
            r_normalized = delta_r / denominator
            i_normalized = delta_i / denominator
            j_normalized = delta_j / denominator
            k_normalized = delta_k / denominator
    
            beta_components = torch.chunk(self.beta, 4, dim=1)
    
            # Multiply gamma (stretch scale) and add beta (shift scale)
            new_r = (self.gamma * r_normalized) + beta_components[0]
            new_i = (self.gamma * i_normalized) + beta_components[1]
            new_j = (self.gamma * j_normalized) + beta_components[2]
            new_k = (self.gamma * k_normalized) + beta_components[3]
    
            new_input = torch.cat((new_r, new_i, new_j, new_k), dim=1)
        

            # with torch.no_grad():
            self.moving_mean.copy_(moving_average_update(self.moving_mean.data, mu.data, self.momentum))
            self.moving_var.copy_(moving_average_update(self.moving_var.data, var.data, self.momentum))
                
            # print(var, self.moving_var)

            
            return new_input
        
        else:
            with torch.no_grad():
                # print(input.shape, self.moving_mean.shape)
                r,i,j,k = torch.chunk(input, 4, dim=1)
                quaternions = [r,i,j,k]
                output = []
                denominator = torch.sqrt(self.moving_var + self.eps)
                beta_components = torch.chunk(self.beta, 4, dim=1)
                # print(torch.tensor(quaternions).shape)
                # print(quaternions[0].shape, self.moving_mean.shape, self.moving_var.shape, torch.squeeze(self.beta).shape)
                for q in range(4):
                    new_quat = self.gamma * ( (quaternions[q] - self.moving_mean[q]) / denominator ) + beta_components[q]
                    output.append(new_quat)
                output = torch.cat(output, dim=1)

                return  output 

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'num_features=' + str(self.num_features) \
               + ', gamma=' + str(self.gamma.shape) \
               + ', beta=' + str(self.beta.shape) \
               + ', eps=' + str(self.eps.shape) + ')'
               

def to_3d(x):
    # b, c, h, w --> b, (h w), c
    x = rearrange(x, 'b c h w -> b (h w) c')
    return x
   
def to_4d(x, h, w):
    x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
    return x

class QuatLayerNorm(Module):

    def __init__(self, num_features):
        super(QuatLayerNorm, self).__init__()
        #import pdb; pdb.set_trace() 
        quat_num_features = num_features // 4
        self.weight_r =  Parameter(torch.ones(quat_num_features))
        self.weight_i =  Parameter(torch.ones(quat_num_features))
        self.weight_j =  Parameter(torch.ones(quat_num_features))
        self.weight_k =  Parameter(torch.ones(quat_num_features))

        self.bias_r = Parameter(torch.zeros(quat_num_features))
        self.bias_i = Parameter(torch.zeros(quat_num_features))
        self.bias_j = Parameter(torch.zeros(quat_num_features))
        self.bias_k = Parameter(torch.zeros(quat_num_features))

   
    def forward(self, input):
        #import pdb; pdb.set_trace()
        b, c, h, w = input.shape
                 
        quat_components = torch.chunk(input, 4, dim=1)
        r, i, j, k = quat_components[0], quat_components[1], quat_components[2], quat_components[3]
      
         
        r = to_3d(r)
        i = to_3d(i)
        j = to_3d(j)
        k = to_3d(k)

 
        mu_r = r.mean(-1, keepdim=True)
        mu_i = i.mean(-1, keepdim=True)
        mu_j = j.mean(-1, keepdim=True)
        mu_k = k.mean(-1, keepdim=True)
 
        sigma_r = r.var(-1, keepdim=True, unbiased=False)
        sigma_i = i.var(-1, keepdim=True, unbiased=False)
        sigma_j = j.var(-1, keepdim=True, unbiased=False)
        sigma_k = k.var(-1, keepdim=True, unbiased=False)

        layer_norm_r = (r - mu_r) / torch.sqrt(sigma_r+1e-5) * self.weight_r + self.bias_r 
        layer_norm_i = (i - mu_i) / torch.sqrt(sigma_i+1e-5) * self.weight_i + self.bias_i 
        layer_norm_j = (j - mu_j) / torch.sqrt(sigma_j+1e-5) * self.weight_j + self.bias_j 
        layer_norm_k = (k - mu_k) / torch.sqrt(sigma_k+1e-5) * self.weight_k + self.bias_k

        layer_norm_r = to_4d(layer_norm_r, h, w)
        layer_norm_i = to_4d(layer_norm_i, h, w)
        layer_norm_j = to_4d(layer_norm_j, h, w)
        layer_norm_k = to_4d(layer_norm_k, h, w)
        
        new_input = torch.cat((layer_norm_r, layer_norm_i, layer_norm_j, layer_norm_k), dim=1)

        return new_input

               
               
# class model(torch.nn.Module):
#     def __init__(self):
#         super(model, self).__init__()
#         self.l = torch.nn.Linear(4,4)
#         self.b = QuaternionBatchNorm2d(4)
#     def forward(self, x):
#         return self.b(self.l(x))
# x = torch.ones(2, 4).to('cuda')
# # L = torch.nn.Linear(4,4).to('cuda')(x)
# # # print(model)
# # B = QuaternionBatchNorm2d(4).to('cuda').eval()(L)
# # model = B
# model = model().to('cuda')
# model.eval()
# # print(model.moving_var)
# y = model(x)
# print(y)                

# loss = y - torch.ones(2,4).to('cuda')      

# loss.backward()       
    
