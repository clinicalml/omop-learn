import torch
import json
from collections import Counter

from omop_learn.data.common import ConceptTokenizer
from omop_learn.utils.date_utils import to_unixtime


class OMOPDatasetTorch(torch.utils.data.Dataset):
    def __init__(
        self,
        omop_dataset,
        tokenize_on_load=False,
        max_num_visits=None,
        tokenizer=None,
    ):
        super().__init__()
        self.omop_dataset = omop_dataset
        self.items = {}
        self.visit_sequences = []  # patient x (# visits for patient) lists w/ concepts expressed
        self.time_sequences = []  # patient x (# visits for patient) times of visits
        self.visit_sizes = []  # patient x (# visits for patient) # concepts in each visit
        self.outcomes = []  # patient--outcome for each patient
        self.tok_visit_sequences = None
        self.tokenizer = tokenizer
        self.max_num_visits = max_num_visits  # if set, truncate to most recent
        self._load_json(omop_dataset.data_file, tokenize_on_load)


    def _load_json(self, path_to_json, tokenize_on_load):
        # read once to build concept set
        # (and load visits if tokenize_on_load=False)
        concept_set = set()
        concept_counts = Counter()
        concept_counts_by_year = Counter()
        years = set()
        max_num_visits = 0
        skipped = 0
        with open(path_to_json) as json_fh:
            for i, line in enumerate(json_fh.readlines()):
                example = self._process_line(line)
                max_num_visits = max(max_num_visits, len(example['visits']))

                for time, visit in zip(example['unix_times'], example['visits']):
                    for concept in visit:
                        concept_set.add(concept)

                if len(example['visits']) == 0:
                    skipped += 1
                    continue

                if i == 0:
                    for key, value in example.items():
                        self.items[key] = [value]
                else:
                    # correctly gives error when key is not found
                    # already; all items need to have exactly the same
                    # set of keys.
                    for key,value in example.items():
                        self.items[key].append(value)

        print(f"Skipped {skipped} patients for empty visit lists")
        if not self.max_num_visits:
            self.max_num_visits = max_num_visits

        if not self.tokenizer:
            self.tokenizer = ConceptTokenizer(concept_set)
            print("built tokenizer")

        # read again to build tokenized visits
        if tokenize_on_load:
            self.items['tok_visits'] = []
            with open(path_to_json, "r") as json_fh:
                for i, line in enumerate(json_fh.readlines()):
                    example = self._process_line(line)
                    tok_visit_list = []
                    for visit in example['visits']:
                        tok_visit = self.tokenizer.concepts_to_ids(visit)
                        tok_visit_list.append(tok_visit)
                    if len(tok_visit_list) > 0:
                        self.items['tok_visits'].append(tok_visit_list)

        self.outcomes = torch.LongTensor(self.items['y'])
        self.one_fraction = self.outcomes.sum() / len(self.outcomes)
        self.one_odds = self.one_fraction / (1 - self.one_fraction)

    def _process_line(self, line):
        example = json.loads(line)
        dates = example['dates']
        unix_times = to_unixtime(dates)
        example['unix_times'] = unix_times

        # make sure visits are sorted by date
        sorted_visits = [v for d,v in sorted(zip(example['unix_times'], example['visits']))]
        example['visits'] = sorted_visits
        example['unix_times'] = sorted(example['unix_times'])
        example['dates'] = sorted(example['dates'])

        return example

    def __getitem__(self, idx):
        example = {k : v[idx] for k,v in self.items.items()}
        times = torch.LongTensor(example['unix_times'])
        visits = example['visits'] if 'tok_visits' not in example else example['tok_visits']

        # trim before tokenizing
        visits = visits[-self.max_num_visits :]
        times = times[-self.max_num_visits :]

        if 'tok_visits' not in example:
            tok_visits = []
            for visit in example['visits']:
                tok_visits.append(self.tokenizer.concepts_to_ids(visit))
            visits = tok_visits

        example['visits'] = visits

        visit_sizes = torch.LongTensor([len(v) for v in visits])
        outcome = self.outcomes[idx]
        nvisits = len(visits)

        return example

    # pads a batch to largest # of visits / patient in the batch
    # and largest # of concepts / visit along concept dim.
    def collate(self, batch):
        # first group along dict keys
        batch_collated = {}
        for k in batch[0].keys():
            batch_collated[k] = [b[k] for b in batch]

        keys = list(batch_collated.keys())
        N = len(batch_collated['y'])

        # each patient is a list of visits
        max_num_visits = max([len(v) for v in batch_collated['visits']])
        max_num_concepts = max(l for p in range(N) for l in [len(v) for v in batch_collated['visits'][p]])

        concept_tensor = torch.full(
            (N, max_num_visits, max_num_concepts),
            self.tokenizer.pad_token_idx,
            dtype=torch.long,
        )

        times_tensor = torch.full((N, max_num_visits), -1, dtype=torch.long)
        batch = batch_collated

        lengths = torch.zeros(N)
        for i, visit_list in enumerate(batch['visits']):
            assert len(visit_list) == len(batch['unix_times'][i])
            num_visits = len(visit_list)  # visits of this patient we are including
            lengths[i] = num_visits
            for j, visit in enumerate(visit_list):
                visit_size = len(batch['visits'][i][j])
                assert(visit_size == len(visit))
                concept_tensor[i, j, : visit_size] = torch.Tensor(visit)
            times_tensor[i, :num_visits] = torch.Tensor(batch['unix_times'][i])
        batch.pop("unix_times")
        batch.pop("dates")
        batch["visits"] = concept_tensor
        batch["times"] = times_tensor
        batch["lengths"] = lengths
        for k,v in batch.items():
            if not isinstance(v, torch.Tensor):
                batch[k] = torch.tensor(v)
        return batch

    def __len__(self):
        return len(self.outcomes)

    def _build(self, path_to_json):
        print(f"Creating JSON data cache at {path_to_json}")
        # todo
