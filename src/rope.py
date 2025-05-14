import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Float


class RoPE(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        '''
        Construct the RoPE module and create buffers.
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        '''
        super(RoPE, self).__init__()
        assert d_k %2 ==0 , "query & key vector must have an even length"
        #self.register_buffer("R", ,persistent=False)
        #to multiply by i the token position
        t: Float[torch.Tensor, "d_k"] = 1.0 / torch.pow(theta, ( 2/ d_k) * torch.arange(0, d_k//2 , device = device, dtype=torch.float32).repeat_interleave(2) )
        i: Float[torch.Tensor, "max_seq_len"] =  torch.arange(0, max_seq_len , device = device)


        theta_tensor: Float[torch.Tensor, "max_seq_len d_k"] = einsum(i, t, "max_seq_len, d_k -> max_seq_len d_k" )


        cos_tensor = torch.cos(theta_tensor)
        sin_tensor = torch.sin(theta_tensor)

        self.register_buffer("cos", cos_tensor, persistent=False)
        self.register_buffer("sin", sin_tensor, persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor: 
        '''
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. 
        '''
        x_neg_shift = x.clone()

        even_idx = torch.arange(0, x.shape[-1], 2, device=x.device)
        odd_idx = even_idx + 1

        x_neg_shift[..., even_idx] = -x[..., odd_idx]
        x_neg_shift[..., odd_idx] = x[..., even_idx]

        return x * self.cos[token_positions] + x_neg_shift * self.sin[token_positions]

