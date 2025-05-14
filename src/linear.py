import torch
import torch.nn as nn
import numpy as np
from einops import einsum
from jaxtyping import Float





class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        '''
        Construct a linear transformation module. This function should accept the following parameters:
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        super(Linear, self).__init__()
        std = np.sqrt(2 / (in_features + out_features))
        self.weight: Float[torch.Tensor, "out_feature in_feature"] = nn.Parameter(torch.empty(size=(out_features, in_features), dtype=dtype, device=device))
        torch.nn.init.trunc_normal_(self.weight, mean = 0, std = std, a = -3 * std , b = 3* std)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        Y = einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out" )
        return Y 
