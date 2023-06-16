import torch
import pandas as pd
from transformers import BertModel, BertConfig
import pdb

from omop_learn.utils.date_utils import to_unixtime


eps = 1e-8


class OnehotModel(torch.nn.Module):
    def __init__(self, featdim, outdim):
        super().__init__()
        self.featdim = featdim
        self.outdim = outdim

    def _make_matrix_onehot(self, tensor_3d):
        device = tensor_3d.device
        bsz, d1, d2 = tensor_3d.shape

        matrix_view = tensor_3d.view((-1, d2))
        N = matrix_view.shape[0]
        xoh = torch.zeros((N, self.featdim), device=device)

        inds = torch.arange(N)[:, None].expand_as(matrix_view).to(device)
        xoh[inds, matrix_view] = 1
        matrix_onehot = xoh.view((bsz, d1, -1)).sum(dim=1).bool().float()
        return matrix_onehot


class OnehotLinear(OnehotModel):
    def __init__(self, featdim, outdim, nonzeros_mask=None):
        super().__init__(featdim, outdim)
        self.w = torch.nn.Linear(featdim, outdim)
        self.nonzeros_mask = nonzeros_mask

    def forward(self, concept_tensor, nvisits):  # take nvisits to match signature of xformer
        patient_onehot = self._make_matrix_onehot(concept_tensor)
        if self.nonzeros_mask is not None:
            out = patient_onehot * self.nonzeros_mask[None, :]  # broadcast mask along batch dim
        else:
            out = patient_onehot
        return self.w(patient_onehot)

    def get_top_feat(self, k, tokenizer):
        top_pos_ids = torch.topk(self.w.weight, k).indices[0]
        top_neg_ids = torch.topk(-self.w.weight, k).indices[0]
        pos_concepts = tokenizer.ids_to_concepts(top_pos_ids.cpu().tolist())
        neg_concepts = tokenizer.ids_to_concepts(top_neg_ids.cpu().tolist())
        return pos_concepts, neg_concepts


class WindowedLinear(OnehotModel):
    def __init__(self, featdim, outdim, pred_day, windows):
        super().__init__(featdim, outdim)

        self.windows_day = torch.LongTensor(windows)
        windows_datetime = [pd.Timedelta(f"{d}d") for d in windows]
        cutoffs_datetime = [-w + pred_day for w in windows_datetime]
        self.windows = to_unixtime(cutoffs_datetime)
        self.pred_day = pred_day
        self.w = torch.nn.Linear(len(windows) * featdim, outdim)

    def forward(self, concept_tensor, times):
        window_onehots = []
        for w in self.windows:
            # bsz x max_nvisits
            time_mask = (times >= w) & (times >= 0)  # >= 0 in case visit before REFTIME
            time_mask = time_mask[:, :, None].expand_as(concept_tensor)
            window_tensor = concept_tensor * time_mask
            window_onehot = self._make_matrix_onehot(window_tensor)
            window_onehots.append(window_onehot)
        patient_onehot = torch.hstack(window_onehots)
        return self.w(patient_onehot)

    # return the top k pos/neg features by weight magnitude
    # and their corresponding time window
    def get_top_feat(self, k, tokenizer):
        n_windows = len(self.windows)
        top_pos_ids = torch.topk(self.w.weight, k).indices[0]
        top_pos_concepts, top_pos_times = (
            top_pos_ids // n_windows,
            top_pos_ids % n_windows,
        )

        top_neg_ids = torch.topk(-self.w.weight, k).indices[0]
        top_neg_concepts, top_neg_times = (
            top_neg_ids // n_windows,
            top_neg_ids % n_windows,
        )
        pos_concepts = tokenizer.ids_to_concepts(top_pos_concepts.cpu().tolist())
        neg_concepts = tokenizer.ids_to_concepts(top_neg_concepts.cpu().tolist())

        pos_concepts = [f"{c} --- {self.windows_day[top_pos_times[i]]} days" for i, c in enumerate(pos_concepts)]
        neg_concepts = [f"{c} --- {self.windows_day[top_neg_times[i]]} days" for i, c in enumerate(neg_concepts)]
        return pos_concepts, neg_concepts


class AdaptiveLinear(torch.nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()
        self.indim = indim
        self.w = torch.nn.Linear(indim, outdim)

    def forward(self, x, maxfeat):
        # pdb.set_trace()
        try:
            return x @ self.w.weight[:, self.indim - maxfeat :].T
        except:
            pdb.set_trace()


# todo: positional embeddings based on visit times, not seq position
class VisitTransformer(torch.nn.Module):
    def __init__(
        self,
        concept_dim,
        embedding_dim,
        max_nvisits,
        right_padded=True,
        linear_pooling=True,
        length_normalize=False,
        npools=5
    ):
        super().__init__()
        self.visit_embedder = VisitEmbedder(concept_dim, embedding_dim)
        self.right_padded = right_padded
        # Should we normalize each patient by the # of visits?
        self.length_normalize = length_normalize

        configuration = BertConfig(
            vocab_size=1,  # 0 gives err; we never pass in input_ids, only input_embs
            hidden_size=embedding_dim,
            max_position_embeddings=max_nvisits,
            num_hidden_layers=3,
            intermediate_size=500,
            num_attention_heads=2,
        )
        self.max_nvisits = max_nvisits  # 300
        self.emb_size = embedding_dim
        self.n_parallel_pools = npools  # hyperparamerer
        self.transformer = BertModel(configuration)

        # if self.conv_accumulator:
        # TODO: test that at the end, these are increasing over time
        self.linear_pooling = linear_pooling
        if linear_pooling:
            # We want to right pad so we do the weighted pool correctly over most recent ones
            assert self.right_padded
            # Hyperparam tune the 1 here TODO
            self.pooler = AdaptiveLinear(self.max_nvisits, self.n_parallel_pools)
        # self.cls = torch.nn.Linear(self.transformer.config.hidden_size, outdim)

    # Example shapes: 39 x 5 x 51,    39
    def forward(self, concept_tensor, num_visits):
        # nvisits x 5 (seq = max_visits) x 300 (embedding dim)
        emb = self.visit_embedder(concept_tensor)

        # We have a different local max_nvisits for the current batch
        max_nvisits = concept_tensor.shape[1]
        assert max_nvisits <= self.max_nvisits

        # only pay attention to the visits that occurred.
        # nvisits x max_nvisits
        inds = torch.arange(max_nvisits)[None, :].to(num_visits.device)
        if self.right_padded:
            # Reverses tensor order on the first dimension
            inds = torch.flip(inds, dims=[1])
            # 1 1 1 0 0
            # to
            # 0 0 1 1 1
        attention_mask = (inds < num_visits[:, None]).int()

        # N x seq x hdim
        # todo: mess with input position_ids---can we mimic the relative embeddings that way?
        # (by computing offset dates to prediction day)
        ctx_emb = self.transformer(inputs_embeds=emb, attention_mask=attention_mask).last_hidden_state

        masked_emb = ctx_emb * attention_mask[:, :, None]
        pooled_emb = masked_emb

        if self.linear_pooling:
            # 39 x 5 x 300 by 39 x 5 x 1, masks the embedding of future visits to 0
            # Pooler takes max_emb_dim to 1

            # We dont want to do the final .transpose(1, 2) because
            # we want the 1 dimension at the end
            # Adaptively adjust the linear layer to only look at the last max_nvisits weights

            pooled_emb = self.pooler(masked_emb.transpose(1, 2), max_nvisits)
            pooled_emb = pooled_emb.view(pooled_emb.shape[0], -1)
        else:
            # Returns batch x 300
            pooled_emb = masked_emb.sum(dim=1)

        if self.length_normalize:
            zmask = num_visits == 0
            num_visits[zmask] = 1
            # Being careful---if nv or nv2 = 0, causes nans even though you 0'd later.
            pooled_emb = pooled_emb / num_visits[:, None]
            # 0 out where we had 0 visits for now
            assert torch.sum(pooled_emb[zmask]) < eps
        else:
            pooled_emb = pooled_emb
        # todo: try predicting from cls embedding? you don't even add a cls token though.
        # Only return contextual embeddings
        # Note this pooled emb is unnormalized
        # (hyperparameter for normalization wherever it is used, by num_visits)

        return pooled_emb

        # N x hdim
        # for now, linear on top of avg pooling.
        # Adds an empty final dimension of size 1, sums
        # pooled_emb = (ctx_emb * attention_mask[:,:,None]).sum(dim=1) / num_visits[:,None]
        # return self.cls(pooled_emb)


class VisitClassifier(torch.nn.Module):
    def __init__(self, transformer, outdim=1):
        super().__init__()
        self.transformer = transformer
        featsize = self.transformer.transformer.config.hidden_size
        if self.transformer.linear_pooling:
            featsize *= self.transformer.n_parallel_pools
        self.cls = torch.nn.Linear(featsize, outdim)

    def forward(self, concepts, lengths):
        pooled_emb = self.transformer(concepts, lengths)
        return self.cls(pooled_emb)


class VisitEmbedder(torch.nn.Module):
    def __init__(self, concept_dim, embedding_dim, dropout_p=0):
        super().__init__()
        # Expands a tensor with indices from a 1-concept dim to an embedding dim
        self.concept_embeddings = torch.nn.Embedding(concept_dim, embedding_dim, padding_idx=0)
        self.concept_dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, visit_tensor):
        bsz, nvisits, max_visit_size = visit_tensor.shape
        # bsz * nvisits x max_visit_size
        flat_visit_view = visit_tensor.view((-1, max_visit_size))
        # bsz * nvisits x max_concept_len x concept_embedding_dim
        concept_embeddings = self.concept_embeddings(flat_visit_view)

        # visit embedding is sum of concept embeddings.
        # future: use a parameterized perm.-invariant function instead of just using sum.
        # e.g. see DeepSets
        tensor_view = concept_embeddings.view((bsz, nvisits, max_visit_size, -1))
        concept_embeddings = self.concept_dropout(tensor_view)
        visit_embeddings = concept_embeddings.sum(dim=2)
        return visit_embeddings
