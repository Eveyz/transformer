import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocal_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocal_size
        self.embedding = nn.Embedding(vocal_size, d_model)

    def forward(self, x):
        # from the original paper, they weight the embedding by the sqrt root of the d_model, which is 512 in this case 
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):

    def __init__(self, 
                 d_model: int, 
                 seq_len: int, # length of the input sentence
                 dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model)
        # https://www.youtube.com/watch?v=ISNdQcPhsts&ab_channel=UmarJamil
        pe = torch.zeros(seq_len, d_model)

        # PE(pos, 2i) = sin( pos / 10000^(2i/d_model) )
        # PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )

        # create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model ))
        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):

    # item 1 / item 2 / item 3
    # u1 u2 u3
    # o1^2 o2^2 o3^2
    # ^xj = (xj - uj)/sqrt(oj^2 + eps) 
    # need eps in case oj is very close to zero, xj will be large

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True) # last dimension
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x-mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (Batch, seq_len, d_model) ---> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))