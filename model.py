# https://www.youtube.com/watch?v=ISNdQcPhsts&ab_channel=UmarJamil
# https://zhuanlan.zhihu.com/p/368920094

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
        # (batch, seq_len, d_model) ---> (batch, seq_len, d_ff) ---> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv
        self.w_o = nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):

        # Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
        # head_i = Attention(QW_i^Q, KW_i^K, VW_tV)

        d_k = query.shape[-1] # the last in shape, probably vocab size ?
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) ---> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # @ ---> matmul
        if mask is not None:
            # write a very low value (indicating -inf) to the positions where mask = 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) ---> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) ---> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) ---> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) ---> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) ---> (batch, seq_len, h, d_k) ---> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)