import os

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score

from omop_learn.omop import OMOPDataset
from omop_learn.sparse.models import OMOPLogisticRegression
from omop_learn.utils.config import Config


load_dotenv()

config = Config({
    "path": os.getenv("DATABASE_PATH"),
    "cdm_schema": os.getenv("CDM_SCHEMA"),
    "aux_cdm_schema": os.getenv("AUX_CDM_SCHEMA"),
    "prefix_schema": os.getenv("USERNAME"),
    "datasets_dir": os.getenv("OMOP_DATASETS_DIR"),
    "models_dir": os.getenv("OMOP_MODELS_DIR")
})


# Re-load a pre-built dataset
dataset = OMOPDataset.from_prebuilt(config.datasets_dir)

# Window the omop dataset and split it
window_days = [30, 180, 365, 730, 1500, 5000, 10000]
windowed_dataset = dataset.to_windowed(window_days)
windowed_dataset.split()

# Generate an OMOPLogisticRegression pipeline for training
model = OMOPLogisticRegression("eol", windowed_dataset)

# Tune on C parameter
Cs = [1, 1e-1, 1e-2, 1e-3, 1e-4]
scores = []
for C in Cs:
    # Gen and fit
    model.gen_pipeline(C)
    model.fit()
    # Eval on validation data
    pred = model._pipeline.predict_proba(windowed_dataset.val['X'])[:, 1]
    score = roc_auc_score(windowed_dataset.val['y'], pred)
    scores.append(score)
    print("C: %.4f, Val AUC: %.2f" % (C, score))

# Gen and fit on best C
best_C = Cs[np.argmax(scores)]
model.gen_pipeline(best_C)
model.fit()
# Eval on test data
pred = model._pipeline.predict_proba(windowed_dataset.test['X'])[:, 1]
score = roc_auc_score(windowed_dataset.test['y'], pred)
print("C: %.4f, Test AUC: %.2f" % (best_C, score))

# Serialize to model directory
model.save()