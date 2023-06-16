import torch
from transformers import (
    Trainer,
    TrainingArguments,
    BertConfig,
    BertForMaskedLM,
    Trainer
)
import datasets

from omop_learn.omop import OMOPVisitDataset


datasets.config.MAX_IN_MEMORY_DATASET_SIZE_IN_BYTES *= 1000
datasets.set_caching_enabled(False)

dataset = OMOPVisitDataset.from_prebuilt("visit_pretrain")
hf_dataset = dataset.to_hf()['train']
hf_dataset = hf_dataset.rename_column("tok_concepts", "input_ids") # this name is important

hf_dataset = hf_dataset.map(lambda examples: {'lengths': [len(c) for c in examples['concepts']]},
                            batched=True, cache_file_name=str(dataset.data_dir / "tmp_cache.tmp"))

max_length = max(hf_dataset['lengths'])

splits = hf_dataset.train_test_split(shuffle=False) # default is 75/25 split
train_dataset, val_dataset = splits['train'], splits['test']

cfg = BertConfig(vocab_size=dataset.tokenizer.vocab_size,
                 hidden_size=300,
                 max_position_embeddings=max_length,
                 num_hidden_layers=12,
                 intermediate_size=500,
                 num_attention_heads=5,
)

model = BertForMaskedLM(cfg)

training_args = TrainingArguments(
    output_dir='./results_mlm_visits',
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500, # number of warmup steps for learning rate scheduler
    learning_rate=1e-5,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=50000,
    save_steps=100000,
    do_train=True,
    prediction_loss_only=True,
)


class MaskedCollator(object):
    def __init__(self, tokenizer, mlm_probability=0.15):
        super().__init__()
        self.tok = tokenizer
        self.mlm_probability = mlm_probability

    def __call__(self, examples):
        lengths = [len(e['input_ids']) for e in examples]
        input_ids = torch.full((len(examples), max(lengths)), self.tok.pad_token_idx)
        for i,l in enumerate(lengths):
            input_ids[i,:l] = torch.tensor(examples[i]['input_ids'])

        # compute attn masks
        lengths = torch.tensor(lengths)
        inds = torch.arange(max(lengths))[None, :]
        attention_mask = (inds < lengths[:, None]).int()

        # now do masking
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # 80% of the time, we replace masked input concepts with tokenizer.mask_token_idx ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tok.mask_token_idx

        # 10% of the time, we replace masked input concepts with random concept
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.tok.vocab_size, labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        position_ids = torch.zeros_like(input_ids) # hack for now; all same pos id so ordering (hopefully---need to check) isn't used

        return {'input_ids': input_ids, 'attention_mask': attention_mask,
                'labels' : labels, 'position_ids': position_ids}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_dataset,
    eval_dataset=val_dataset,
    data_collator=MaskedCollator(dataset.tokenizer),
)
trainer.train()
