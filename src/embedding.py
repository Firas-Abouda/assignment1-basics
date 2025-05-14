import torch
import torch.nn as nn
from jaxtyping import Float




class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None) :
        '''
        Construct an embedding module. This function should accept the following parameters:
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        super(Embedding, self).__init__()
        self.embedding: Float[torch.Tensor, "vocab_size dmodel"] = nn.Parameter(torch.empty(size=(num_embeddings, embedding_dim), dtype=dtype, device=device))
        torch.nn.init.trunc_normal_(self.embedding, mean = 0, std = 1, a = -3  , b = 3)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor: 
        '''
        Lookup the embedding vectors for the given token IDs.
        '''
        return self.embedding[token_ids]