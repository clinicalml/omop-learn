from pathlib import Path, PosixPath
import dill
import json

import scipy.sparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

from omop_learn.sparse.data import OMOPDatasetWindowed


def _sparse_ufunc(f):
    def wrapper(*a, **k):
        X = a[0]
        if not scipy.sparse.isspmatrix(X):
            raise ValueError
        X2 = X.copy()
        X2.data = f(X2.data, *(a[1:]), **k)
        return X2
    return wrapper

@_sparse_ufunc
def _tr_func(X, kwarg=1):
    """Binarize a feature matrix X.
    """
    return np.clip(X, 0, kwarg)

class OMOPLogisticRegression(object):
    """Train a logistic regression on a OMOPDatasetWindowed object.

    Attributes
    ----------

    """
    def __init__(
        self, 
        name: str, 
        windowed_dataset: OMOPDatasetWindowed, 
        _pipeline: Pipeline = None,
        model_root: PosixPath = None, 
        _already_built: bool = False
    ) -> None:
        
        """Initialize an OMOPLogisticRegression object.
        """
        
        self.name = name
        self.windowed_dataset = windowed_dataset
        self._already_built = _already_built

        if model_root:
            model_root = Path(model_root)
        else:
            model_root = Path(self.windowed_dataset.omop_dataset.config.models_dir)

        self.model_root = model_root

        self.model_dir = self.model_root / self.name
        # Preprocess step for data
        self._pre_process = FunctionTransformer(
            func=_tr_func, accept_sparse=True, validate=True, 
            kw_args={'kwarg': 1}
        )

        if self.model_dir.is_dir() and not self._already_built:
            print("A model with that name already exists. Loading model...")
            self._pipeline = self._load()
        else:
            # make model dir
            self.model_dir.mkdir(parents=True, exist_ok=True)
                
            # Where the current model pipeline will be saved
            self._pipeline = _pipeline

    def gen_pipeline(self, C: float = 0.01) -> None:
        lr = LogisticRegression(
            class_weight='balanced', C=C,
            penalty='l1', fit_intercept=True,
            solver='liblinear', random_state=0,
            verbose=0, max_iter = 200, tol=1e-1
        )

        # The classifier will transform each data point using func, 
        # which here takes a count vector to a binary vector
        # Then, it will use logistic regression to classify 
        # the transformed data
        self._pipeline = Pipeline([
            ('func', self._pre_process),
            ('lr', lr)
        ])
        
    def fit(self) -> None:

        """Fit a model pipeline.

        Uses data generated via OMOPDatasetWindowed split function.
        """
        
        assert self._pipeline is not None
        
        self._pipeline.fit(self.windowed_dataset.train['X'], self.windowed_dataset.train['y'])

    def _serialize(self):
        """Serialize an OMOPLogisticRegression object.

        What needs to be saved include the following:

        1. pre_process: Preprocessing step of the model pipeline.
        2. model: The learned logistic regression model.
        3. dataset.window_days: Window lengths to generate a windowed dataset.
        4. dataset.config: Database configuration parameters.
        5. dataset.features: Information needed to re-generate features.
        6. dataset.cohort.params: Cohort parameters to re-generate a cohort.
        7. dataset.tokenizer: Original concept to token map.
        """
        with open(self.model_dir / "model.pkl", "wb") as fh:
            dill.dump(self._pipeline, fh)
        
        with open(self.model_dir / 'window_days.json', 'w') as fh:
            json.dump(self.windowed_dataset.window_days, fh)
        
        self.windowed_dataset.omop_dataset.config.serialize(self.model_dir / "config.json")
        self.windowed_dataset.omop_dataset.features.serialize(self.model_dir / "features.json")
        self.windowed_dataset.omop_dataset.cohort.serialize(self.model_dir)
        self.windowed_dataset.omop_dataset.tokenizer.serialize(self.model_dir / "tokens.json")

    def _load(self) -> None:
        with open(self.model_dir / "model.pkl", "rb") as fh:
            return dill.load(fh)
        
    def save(self) -> None:
        """Saves information necessary to use the model in an inference setting."""
        self._serialize()
