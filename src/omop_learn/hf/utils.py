def encode(examples, tokenizer=None):
    all_encoded_visits = []
    for i in range(len(examples['visits'])):
        visit_list = examples['visits'][i]
        encoded_visits = []
        for v in visit_list:
            encoded_visit = tokenizer.concepts_to_ids(v)
            encoded_visits.append(encoded_visit)
        all_encoded_visits.append(encoded_visits)
    return {'tok_visits': all_encoded_visits}


def visit_encode(examples, tokenizer=None):
    encoded_visits = []
    for v in examples["concepts"]:
        v_encoded = tokenizer.concepts_to_ids(v)
        encoded_visits.append(v_encoded)
    return {"tok_concepts": encoded_visits}

def collate(batch, tokenizer=None, max_num_concepts=None, max_num_visits=None):
    pass
