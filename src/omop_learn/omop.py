import pathlib
from pathlib import Path
from datasets import load_dataset

from omop_learn.utils.config import Config
from omop_learn.data.cohort import Cohort
from omop_learn.data.common import ConceptTokenizer
from omop_learn.data.feature import FeatureSet
from omop_learn.sparse.data import OMOPDatasetSparse, OMOPDatasetWindowed
from omop_learn.torch.data import OMOPDatasetTorch
from omop_learn.hf import utils as hf_utils


class OMOPDataset(object):
    def __init__(
        self,
        name=None,
        config=None,
        cohort=None,
        features=None,
        backend=None,
        data_dir=(Path.cwd() / "datasets/"),
        tokenizer=None,
        is_visit_dataset=False,
        num_workers=10,
        _already_built=False,
    ):
        dataset_dir = data_dir / name
        if dataset_dir.is_dir() and not _already_built:
            raise Exception("A dataset with that name already exists. Please choose a different name or back up the old one.")

        # make dset dir
        dataset_dir.mkdir(parents=True, exist_ok=True)

        self.data_dir = data_dir
        self.dataset_dir = dataset_dir
        self.config = config
        self.cohort = cohort

        if isinstance(features, FeatureSet):
            self.features = features
        else:
            self.features = FeatureSet(features)

        self.is_visit_dataset = is_visit_dataset
        self.tokenizer = tokenizer
        self.backend = backend
        self.num_workers = num_workers
        self.data_file = dataset_dir / 'data.json'
        
        if not self.data_file.is_file() and not _already_built:
            self.tokenizer = self.build()
        elif not self.data_file.is_file():
            raise FileNotFoundError

        # serialize cohort, config, features so we can load them later
        # write metadata after build so we can write tokenizer
        if not _already_built:
            self.serialize()

    def serialize(self, dataset_dir: pathlib.PosixPath = None, with_data: bool = True) -> None:
        
        """Serialize an OMOPDataset object.
        
        Parameters
        ----------
        dataset_dir : pathlib.PosixPath
            Parent directory to serialize information.
        with_data : bool
            Whether to serialize data objects as well as metadata.
        """
        
        # Allow for saving to locations other than where we initially specified.
        if dataset_dir is None:
            assert self.dataset_dir is not None
            dataset_dir = self.dataset_dir
            
        self.config.serialize(dataset_dir / "config.json")
        self.cohort.serialize(dataset_dir, with_data)
        self.features.serialize(dataset_dir / "features.json")
        assert(self.tokenizer is not None)
        self.tokenizer.serialize(dataset_dir / "tokens.json")

    def __len__(self):
        return len(self.cohort) # todo: this is wrong; should be len of data.json (they're not equal!)

    def build(self):
        # Call backend to build feature data
        return self.backend.build_features(
            self.config,
            self.cohort,
            self.features,
            self.tokenizer,
            self.dataset_dir,
            self.is_visit_dataset,
            self.num_workers
        )
    
    @classmethod
    def from_prebuilt(cls, name, data_dir=Path.cwd() / "datasets/"):
        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)

        dataset_dir = data_dir / name

        if not dataset_dir.exists():
            print(f"Couldn't find a directory at {data_dir / name}. Is your data_dir correct?")
            raise FileNotFoundError

        data_file = dataset_dir / "data.json"

        if not data_file.exists():
            print(f"Couldn't find the data file for dataset {name}. Are you sure it was generated with OMOPDataset?")
            raise FileNotFoundError

        config = Config.from_file(dataset_dir  / "config.json")
        cohort = Cohort.from_files(dataset_dir  / "cohort.csv", data_dir / name / "cohort_params.json")
        features = FeatureSet.from_file(dataset_dir / "features.json")
        tokenizer = ConceptTokenizer.from_file(dataset_dir / "tokens.json")

        init_args = {
            "name" : name,
            "config" : config,
            "cohort" : cohort,
            "features": features,
            "tokenizer": tokenizer,
            "data_dir": data_dir,
            "_already_built": True
        }

        return cls(**init_args)

    def to_torch(self, **kwargs):
        # todo: do train/test splits here and handle the correct tokenizing
        # i.e., return a pair of datasets instead of one if there's train_pct specified.
        # use the train_dataset tokenizer to build the

        # if you call w/ tokenizer=None, it rebuilds the tokenizer in the DatasetTorch
        # constructor. This should make train/test splitting easier.
        if 'tokenizer' not in kwargs:
            kwargs['tokenizer'] = self.tokenizer
        return OMOPDatasetTorch(self, **kwargs)

    def to_hf(self, **kwargs):
        # todo: less clean way to deal w/ separate train/test
        # tokenization here than in to_torch
        dataset = load_dataset('json', data_files = str(self.data_file),
                               cache_dir=self.data_dir / "hf_cache")
        if 'tokenizer' not in kwargs:
            kwargs['tokenizer'] = self.tokenizer
        dataset = dataset.map(hf_utils.encode, batched=True, fn_kwargs=kwargs)
        return dataset

    def to_sparse(self):
        return OMOPDatasetSparse(self, self.tokenizer)

    def to_windowed(self, window_days):
        return OMOPDatasetWindowed(self, window_days, self.tokenizer)


# almost identical to OMOPDataset, but data.json is
# one line / visit instead of one line / patient.
class OMOPVisitDataset(OMOPDataset):

    def __init__(self, **kwargs):
        assert('features' in kwargs)
        ntmp_features = list(filter(lambda f: not f.temporal, kwargs['features']))
        assert(len(ntmp_features) == 0)
        super().__init__(**kwargs, is_visit_dataset=True)

    def to_hf(self, **kwargs):
        # todo: less clean way to deal w/ separate train/test
        # tokenization here than in to_torch

        dataset = load_dataset('json', data_files = str(self.data_file),
                               cache_dir=self.data_dir / "hf_cache")
        if 'tokenizer' not in kwargs:
            kwargs['tokenizer'] = self.tokenizer
        dataset = dataset.map(hf_utils.visit_encode, batched=True, fn_kwargs=kwargs)
        return dataset
