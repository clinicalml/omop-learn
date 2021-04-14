import torch
from torch import nn
import math
import torch.nn.functional as F


'''
@article{wolf2019transformers,
  title={Transformers: State-of-the-art Natural Language Processing},
  author={Wolf, Thomas and Debut, Lysandre and Sanh, Victor and Chaumond, Julien and Delangue, Clement and Moi, Anthony and Cistac, Pierric and Rault, Tim and Louf, R{\'e}mi and Funtowicz, Morgan and others},
  journal={arXiv preprint arXiv:1910.03771},
  year={2019}
}
'''

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Attention(nn.Module):

    def forward(self, query, key, value, mask=None, dropout=None, backwards_only=False):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
            / math.sqrt(query.size(-1))

        if mask is not None:
            
            if not backwards_only:
                mask_shaped = torch.einsum(
                    'bi,bj->bij', (mask, mask)
                ).unsqueeze(1).expand(scores.shape)
            else:
                mask_shaped = torch.triu(torch.einsum(
                    'bi,bj->bij', (mask, mask)
                )).unsqueeze(1).expand(scores.shape)
                
            scores = scores.masked_fill(mask_shaped == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, hidden_size = 300, num_attention_heads = 4, dropout=0.3, backwards_only=False):
        super().__init__()
        assert hidden_size % num_attention_heads == 0

        self.backwards_only = backwards_only
                                       
        self.d_k = hidden_size // num_attention_heads
        self.h = num_attention_heads


        self.linear_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(3)]
        )
        self.output_linear = nn.Linear(hidden_size, hidden_size)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)
        
        self.attn = None

    def forward(self, query, key, value, mask=None, backwards_only=False):
        batch_size = query.size(0)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        x, attn = self.attention(
            query, key, value, mask=mask, dropout=self.dropout, backwards_only=self.backwards_only)
        x = x.transpose(1, 2).contiguous().view(
            batch_size, -1, self.h * self.d_k)

        self.attn = attn
        
        return self.output_linear(x)

class SublayerConnection(nn.Module):

    def __init__(self, hidden_size=300,dropout=0.3):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):

    def __init__(self,hidden_size=300,dropout=0.3):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.w_2 = nn.Linear(4 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(gelu(self.w_1(x))))


class TransformerBlock(nn.Module):

    def __init__(self, hidden, heads, dropout=0.3, tr_dropout=0.3, backwards_only=False):

        super().__init__()
        self.attention = MultiHeadedAttention(hidden, heads, tr_dropout, backwards_only=backwards_only)
        self.feed_forward = PositionwiseFeedForward(hidden, tr_dropout)
        self.input_sublayer = SublayerConnection(hidden, tr_dropout)
        self.output_sublayer = SublayerConnection(hidden, tr_dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        x = self.input_sublayer(
            x, lambda y: self.attention.forward(y, y, y, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
    
class RNNBlock(nn.Module):
    
    def __init__(self, hidden, dropout=0.3, tr_dropout=0.3):
        super().__init__()
        self.rnn = torch.nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            batch_first=True
        )
        
    def forward(self, x, mask=None):
        return self.rnn(x)[0]
        
    
class BertLMPredictionHead(nn.Module):
    
    def __init__(self, voc_size=None, hidden_size=300):
        super(BertLMPredictionHead, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = gelu
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        

        self.decoder = nn.Linear(hidden_size, voc_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        
        hidden_states = self.decoder(hidden_states)
        return hidden_states
    
