import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Float

from src.rmsnorm import RMSNorm
from src.multihead_self_attention import MHSA
from src.swiglu import SwiGLU
from src.embedding import Embedding
from src.linear import Linear

class Transformer_block(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, device= None, dtype=None):
        '''
        Construct the Pre-norm Transformer block
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        d_ff: int Dimensionality of the position-wise feed-forward inner layer.
        max_seq_len: int Maximum sequence length that will be inputted
        theta: float Θ value for the RoPE
        device: torch.device | None = None Device to store the buffer on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        super(Transformer_block, self).__init__()

        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attention = MHSA(d_model, num_heads, True, max_seq_len, theta, device, dtype)

        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.swiglu = SwiGLU(d_model, d_ff, device, dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        att_x = self.norm1(x)
        att_x = self.attention(att_x)

        x = x +att_x

        fx = self.norm2(x)
        fx = self.swiglu(fx)

        x = x +fx 
        return x



class Transformer_LM(nn.Module):

    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, theta, device= None, dtype=None):
        '''
        vocab_size: int The size of the vocabulary, necessary for determining the dimensionality of the token
        embedding matrix.
        context_length: int The maximum context length, necessary for determining the dimensionality of
        the position embedding matrix.
        num_layers: int The number of Transformer blocks to use.
        num_heads: int Number of heads to use in multi-head self-attention.
        d_ff: int Dimensionality of the position-wise feed-forward inner layer.
        theta: float Θ value for the RoPE
        device: torch.device | None = None Device to store the buffer on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        super(Transformer_LM, self).__init__()
        self.embedding = Embedding(vocab_size, d_model, device, dtype)
        self.transformer_blocks = nn.ModuleList([
            Transformer_block(d_model, num_heads, d_ff, context_length, theta, device, dtype)
            for i in range(num_layers)
        ])
        self.norm = RMSNorm(d_model=d_model, device= device, dtype=dtype)
        self.linear = Linear(d_model, vocab_size)


    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.embedding(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.norm(x)
        x = self.linear(x)

        return x