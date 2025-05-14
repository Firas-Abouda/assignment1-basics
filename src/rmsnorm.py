import torch
import torch.nn as nn
from einops import einsum, reduce
from jaxtyping import Float


class RMSNorm(nn.Module) : 


    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None): 
        '''
        Construct the RMSNorm module. This function should accept the following parameters:
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        super(RMSNorm,self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma : Float[torch.Tensor, "d_model"] = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        '''
        Process an input tensor of shape batch_size, sequence_length, d_model) and return a tensor of the same shape.
        '''
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x2 = torch.pow(x, 2)
    
        rmsa = torch.sqrt(reduce(x2, "batch_size sequence_length d_model -> batch_size sequence_length 1 ", 'mean' )   + self.eps)
        result =  einsum(x, self.gamma, "batch_size sequence_length d_model, d_model -> batch_size sequence_length d_model") / rmsa

        return result.to(in_dtype)
