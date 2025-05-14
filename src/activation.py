import torch
import torch.nn as nn
from einops import einsum, reduce
from jaxtyping import Float
from math import sqrt

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class Softmax(nn.Module) :
    def __init__(self):
        super().__init__()
    
    def forward(self, x , dim):
        c =  torch.max(x, dim=dim, keepdim=True).values
        x_c  = torch.exp(x - c)

        return x_c / torch.sum(x_c, dim=dim, keepdim=True)
    

class DpAttention(nn.Module) : 
    def __init__(self):
        super().__init__()
    
    def forward(self, Q, K, V, mask=None):
        qk: Float[torch.Tensor, "... keys queries"] = einsum(Q, K , "... queries d_k, ... keys d_k -> ... queries keys")
        sdk = sqrt(K.shape[-1])
        l = qk / sdk

        if mask is not None :
            l = l.masked_fill(~mask, float('-inf'))


        sft : Float[torch.Tensor, "... queries keys"] = Softmax().forward(l, -1)
        attention : Float[torch.Tensor, "... queries dv"] = einsum(sft ,V, "... queries keys, ... keys dv -> ... queries dv")

        return attention

