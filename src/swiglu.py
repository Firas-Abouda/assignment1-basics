import torch
import torch.nn as nn


from src.linear import Linear
from src.activation import Swish



class SwiGLU(nn.Module) : 

    def __init__(self, d_model, d_ff = None,  device=None, dtype=None):
        '''
        Construct a SwiGLU transformation module. 
        '''


        super().__init__()

        assert d_model %64 == 0, "model dimension should be a multiple of 64 for hardware optimization"

        if d_ff is None :
            d_ff = int( 8 * d_model / 3)

        self.w1  = Linear(d_model, d_ff, device, dtype)
        self.w3  = Linear(d_model, d_ff, device, dtype)

        self.w2  = Linear(d_ff, d_model, device, dtype)
        
        self.swish = Swish()

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.w2(self.swish(self.w1(x)) * self.w3(x))


