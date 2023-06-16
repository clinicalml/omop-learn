from dotenv import load_dotenv
import torch

from omop_learn.omop import OMOPDataset
from omop_learn.torch.models import VisitClassifier, VisitTransformer


load_dotenv()

config = Config({
    "path": os.getenv("DATABASE_PATH"),
    "cdm_schema": os.getenv("CDM_SCHEMA"),
    "aux_cdm_schema": os.getenv("AUX_CDM_SCHEMA"),
    "prefix_schema": os.getenv("USERNAME"),
    "datasets_home": os.getenv("OMOP_DATASETS_DIR")
})


# train directly with torch. easier if you want to do custom stuff,
# since you'll need to define your own training loop.

dataset = OMOPDataset.from_prebuilt("eol_cohort6")

# inspect parameters used to create this cohort
print(dataset.cohort.params)

torch_dataset = dataset.to_torch()

train_dataloader = torch.utils.data.DataLoader(torch_dataset, batch_size=16, shuffle=True,
                                               collate_fn=torch_dataset.collate)

tokenizer = torch_dataset.tokenizer

vtf = VisitTransformer(tokenizer.vocab_size, 300, torch_dataset.max_num_visits)
model = VisitClassifier(vtf, outdim=1)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.BCEWithLogitsLoss()

device='cuda'
model = model.to(device)

print("starting training loop")
num_epochs = 10

for e in range(num_epochs):
    running_loss = 0.0
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k,v in batch.items()}
        visits = batch['visits']
        output_logits = model(batch['visits'], batch['lengths']).view(-1)
        loss = criterion(output_logits, batch['y'].float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    print('[%d] loss: %.3f' % (e + 1, running_loss))
    
    # model.eval()
    # ... In case you want to evaluate the model somehow
