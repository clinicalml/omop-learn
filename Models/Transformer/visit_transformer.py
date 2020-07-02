import Models.Transformer.transformers as transformers
from gensim.models import KeyedVectors
import numpy as np
import torch
import math
import importlib
importlib.reload(transformers)


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class VTClassifer(torch.nn.Module):
    def __init__(self, bert_model, n_targets=1):
        super(VTClassifer, self).__init__()
        self.n_targets = n_targets
        self.emb_size = bert_model.embedding_dim 
        self.bert = bert_model
        self.pooler = torch.nn.Linear(self.bert.max_visits, 10)
        self.linear = torch.nn.Linear(self.emb_size * 10, n_targets)
        self.dropout = torch.nn.Dropout(bert_model.dropout)
    def forward(self, x):
        x = self.bert(x, train=False)
        pooled = self.pooler(
            x.transpose(1,2)
        ).view(-1, 10 * self.emb_size)
        pooled = self.dropout(pooled)
        y_pred = self.linear(torch.nn.ReLU()(pooled))
        if self.n_targets == 1:
            return y_pred.flatten(0, -1)
        return y_pred
    
class VisitTransformer(torch.nn.Module):
    def __init__(
        self, featureSet,
        embedding_dim=300,
        n_heads=2, attn_depth=2,
        dropout=0.3,
        concept_embedding_path=None,
        time_emb_type='sin',
        use_RNN=False,
        use_mask=False,
        max_days=365
    ):
        super(VisitTransformer, self).__init__()
        
        self.time_emb_type = time_emb_type
        
        self.data_set = False
        self.concept_embedding_path = concept_embedding_path
        self.featureSet = featureSet
        
        
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim * n_heads
        self.dropout = dropout
        
        self.max_days = max_days
        self.max_visits = 512 - 2
        self.attn_depth = attn_depth

        self.mask_prob = 0.15
        self.rand_prob = 0.1
        self.keep_prob = 0.1
        
        self.use_mask = use_mask
        
        self._initialize_concept_embeddings()
        
        if self.time_emb_type == 'sin':
            self.timescales = ((
                (1 / 10000) ** (1 / self.embedding_dim)
                ) ** torch.FloatTensor(range(self.embedding_dim // 2))).cuda()
            
        elif self.time_emb_type == 'learnfreq':
            self.ampl = torch.nn.Parameter(torch.randn(self.embedding_dim // 2))
            self.freq = torch.nn.Parameter((
                (1 / 10000) ** (1 / self.embedding_dim)
                ) ** torch.FloatTensor(range(self.embedding_dim // 2)))
            self.phase = torch.nn.Parameter(torch.randn(self.embedding_dim // 2))
            
        else:
            self.time_embedder = torch.nn.Embedding(
                self.max_days,
                self.embedding_dim)
            
        if not self.use_mask:
            self.start_embedding = torch.nn.Parameter(torch.randn(self.embedding_dim))
            self.pad_embedding = torch.nn.Parameter(torch.zeros(self.embedding_dim))
            self.mask_embedding = torch.nn.Parameter(torch.randn(self.embedding_dim))
        else:
            self.pad_embedding = torch.zeros(self.embedding_dim).cuda()

        if use_RNN:
            self.tfs = torch.nn.ModuleList([
                transformers.RNNBlock(self.embedding_dim, self.dropout)
            ])
        else:
            self.tfs = torch.nn.ModuleList([
                transformers.TransformerBlock(self.embedding_dim, self.n_heads, self.dropout)
                for _ in range(self.attn_depth)
            ])
        
    def _initialize_concept_embeddings(self):
        
        self.concept_embedder = torch.nn.Embedding(
            len(self.featureSet.concept_map),
            self.embedding_dim
        )
        if self.concept_embedding_path is not None:
            wv = KeyedVectors.load(self.concept_embedding_path, mmap='r')
            for i in range(len(self.featureSet.concept_map)):
                try:
                    self.concept_embedder.weight.data[
                        i, :
                    ] = torch.FloatTensor(wv[str(i)])
                except KeyError:
                    pass
                
    def set_data(
        self, all_codes_tensor,
        person_indices, visit_chunks,
        visit_time_rel, n_visits
    ):
        
        self.all_codes_tensor = all_codes_tensor
        self.person_indices = person_indices
        self.visit_chunks = visit_chunks
        self.visit_time_rel = visit_time_rel
        self.n_visits = n_visits
        
        self.data_set = True
        
    def forward(self, person_range, train=True):
        
        use_mask = self.use_mask
        
        assert(self.data_set)
        
        embedded_raw = self.concept_embedder(self.all_codes_tensor[
            np.concatenate([
                np.array(range(
                    self.person_indices[p] + self.visit_chunks[p][0],
                    self.person_indices[p] + self.visit_chunks[p][-1]
                )) for p in person_range
            ])
        ])

        curr = 0
        person_ix = []
        sum_indices = []
        for p in person_range:
            for v in range(len(self.visit_chunks[p]) - 1):
                
                sum_indices += [curr for _ in range(
                    self.visit_chunks[p][v + 1]
                    - self.visit_chunks[p][v]
                )]
                
                if self.visit_chunks[p][v + 1] > self.visit_chunks[p][v]:
                    person_ix.append(p)
                    
                curr += 1
        
        summed = torch.zeros(len(person_ix), self.embedding_dim).cuda()
        summed = summed.index_add_(
            0, torch.tensor(sum_indices).cuda(),
            embedded_raw
        )
        
        reshaped = self.pad_embedding.unsqueeze(0).repeat(
            len(person_range), self.max_visits, 1
        )
        
        if use_mask:
            mask = torch.zeros(reshaped.shape[:-1]).cuda()
        
        curr = 0 
        for i, p in enumerate(person_range):
            curr += self.n_visits[p]
            seq_len = min(self.max_visits, self.n_visits[p])
            reshaped[i, -seq_len: , :] = summed[curr - seq_len: curr, :]

        times = torch.cat([
            torch.FloatTensor(self.visit_time_rel[p]).cuda()
            for p in person_range
        ]).clamp(0, self.max_days - 1)
        
        
        if self.time_emb_type == 'sin':
            time_embedding_unshaped = torch.cat([
                torch.sin(torch.ger(times, self.timescales)),
                torch.cos(torch.ger(times, self.timescales))
            ], 1)
            
        elif self.time_emb_type == 'learnfreq':
            time_embedding_unshaped = torch.cat([
                self.ampl * torch.sin(self.phase + torch.ger(times, self.freq)),
                self.ampl * torch.cos(self.phase + torch.ger(times, self.freq))
            ], 1)

        else:
            time_embedding_unshaped = self.time_embedder(
                torch.cat([
                    torch.tensor(self.visit_time_rel[p]).cuda()
                    for p in person_range
                ]).clamp(0, self.max_days - 1))
        if self.time_emb_type == 'none':
            time_embedding_unshaped = 0 * time_embedding_unshaped
            
        
        
        time_embedding = torch.zeros(reshaped.shape).cuda()
        
        mask = torch.zeros(reshaped.shape[:-1]).cuda()
        
        curr = 0 
        for i, p in enumerate(person_range):
            curr += self.n_visits[p]
            seq_len = min(self.max_visits, self.n_visits[p])
            time_embedding[i, -seq_len: , :] = time_embedding_unshaped[curr - seq_len: curr, :]
            if use_mask:
                mask[i, -seq_len: ] = 1
        
        output_emb = reshaped + time_embedding
        for tf in self.tfs:
            if use_mask:
                output_emb = tf(output_emb, mask)
            else:
                output_emb = tf(output_emb)
        return output_emb
    
    
