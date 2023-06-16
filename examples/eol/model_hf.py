import datasets

from omop_learn.omop import OMOPDataset


datasets.config.MAX_IN_MEMORY_DATASET_SIZE_IN_BYTES *= 1000

# train using the huggingface (hf) ecosystem.
# less code when you want to do standard stuff,
# since you're using built-in training + eval loops.

dataset = OMOPDataset.from_prebuilt("eol_cohort6")
hf_dataset = dataset.to_hf()
hf_dataset.set_format('torch')
