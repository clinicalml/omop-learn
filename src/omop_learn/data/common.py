import json


class ConceptTokenizer():
    def __init__(
        self,
        concept_set,
        bos_token = "[BOS]",
        eos_token = "[EOS]",
        sep_token = "[SEP]",
        pad_token = "[PAD]",
        unk_token = "[UNK]",
    ):
        self.concept_list = [bos_token, eos_token, sep_token, pad_token, unk_token] + sorted(list(set(concept_set)))
        self.concept_map = {concept: i for i, concept in enumerate(self.concept_list)}
        self.bos_token = bos_token
        self.eos_token = eos_token 
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token_idx = 0
        self.eos_token_idx = 1
        self.sep_token_idx = 2
        self.pad_token_idx = 3
        self.unk_token_idx = 4
        self.num_special_tokens = 5
        self.vocab_size = len(self.concept_list)

    def concepts_to_ids(self, concept_list):
        ret = []
        for concept in concept_list:
            concept = self.concept_map.get(concept, self.unk_token_idx)
            ret.append(concept)
        return ret

    def ids_to_concepts(self, id_list):
        ret = []
        for i in id_list:
            ret.append(self.concept_list[i])
        return ret

    def extend(self, new_concept_set):
        sorted_new = sorted(list(new_concept_set))
        filtered = filter(lambda c: c not in self.concept_map, sorted_new)
        self.concept_list.extend(filtered)

        # lazy---do a full rebuild
        self.concept_map = {concept: i for i, concept in enumerate(self.concept_list)}
        self.vocab_size = len(self.concept_list)


    def serialize(self, filename):
        with open(filename, 'w') as fh:
            json.dump(vars(self), fh)

    @classmethod
    def from_file(cls, path_to_json):
        with open(path_to_json, 'r') as fh:
            d = json.load(fh)
        concept_list = d['concept_list'][d['num_special_tokens']:]
        keep_keys = set(['bos_token', 'eos_token', 'sep_token', 'pad_token', 'unk_token'])

        # lazy; pop keys to match constructor signature
        for k in filter(lambda k: k not in keep_keys, list(d.keys())):
            d.pop(k)

        return cls(concept_list, **d)

    @classmethod
    def from_omop_dataset(cls, dataset):
        pass
