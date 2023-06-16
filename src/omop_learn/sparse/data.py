import json
from typing import List, Tuple, Dict
from datetime import timedelta, datetime

from scipy.sparse import csr_matrix, coo_matrix, hstack
import pandas as pd
import numpy as np
import sparse
from sklearn.model_selection import train_test_split

from omop_learn.data.common import ConceptTokenizer
from omop_learn.utils.date_utils import to_unixtime, from_unixtime


class OMOPDatasetSparse(object):
  
    """Generate a 3D sparse representation of an OMOPDataset.

    Attributes
    ----------
    omop_dataset : OMOPDataset
        An OMOP dataset object.
    tokenizer : ConceptTokenizer
        Object used to map concepts to matrix indices.
    feature_tensor : sparse.COO matrix
        3D sparse matrix of persons x times x concepts.
    outcomes : list of int
        List of outcomes by person.
    times_map : dict of [int, int]
        Dictionary mapping unix times to integer IDs.
    """
    
    def __init__(
        self, 
        omop_dataset: 'OMOPDataset', 
        tokenizer: ConceptTokenizer = None
    ) -> None:

        """Initialize an OMOPDatasetSparse object.

        Parameters
        ----------
        omop_dataset : OMOPDataset
            An OMOP dataset object.
        tokenizer : ConceptTokenizer
            Object used to map concepts to matrix indices.
        """

        self.omop_dataset = omop_dataset
        self.tokenizer = tokenizer
        self.feature_tensor = None
        self.outcomes = None
        self.times_map = None
        assert tokenizer is not None
        output = self._gen_sparse_data(omop_dataset.data_file)
        self.feature_tensor = output[0]
        self.outcomes = output[1]
        self.times_map = output[2]

    def _gen_sparse_data(
        self, 
        path_to_json : str
    ) -> Tuple[sparse.COO, List[int], Dict[int, int]]:

        """Load and convert data from json to a 3D sparse matrix.

        Parameters
        ----------
        path_to_json : str
            Path to the json file of data.

        Returns
        ------
        output : tuple of (sparse.COO, list of int, dict of [int int])
             3D sparse matrix of person x time x concept, a list of 
             outcomes, and a dictionary of unix times to integer ids.
        """
        
        # Elements we'll build iteratively
        concepts = []
        times = []
        persons = []
        outcomes = []
        times_set = set()
        
        # Process the json data file by line
        # A line constitutes an entire person worth of data
        with open(path_to_json, "r") as json_fh:
            for person_id, line in enumerate(json_fh.readlines()):

                # This is a person
                example = self._process_line(line)
                outcomes.append(example['y'])

                # These are the visits, which can have many concepts each
                for i, v in enumerate(example['unix_times']):

                    # This is the number of concepts in this visit
                    visit_concept_num = len(example['visits'][i])

                    # Extend lists by the number of concepts in this visit
                    concepts.extend(example['visits'][i])
                    times.extend([v]*visit_concept_num)
                    persons.extend([person_id]*visit_concept_num)

                    # Make a time set for use in mapping later
                    times_set.add(v)

        # Now make a dict of our times. Save the map for use in windowed methods.
        times_list = sorted(list(times_set))
        times_map = {time: i for i, time in enumerate(times_list)}
        del times_list
        
        # Equivalent to ConceptTokenizer concepts_to_ids
        times_mapped = []
        for time in times:
            time = times_map.get(time)
            times_mapped.append(time)

        # Build 3D sparse matrix representation of the data
        # persons x times x concepts
        concepts_mapped = self.tokenizer.concepts_to_ids(concepts)
        feature_matrix = sparse.COO(
          [persons, times_mapped, concepts_mapped], 1, 
          shape=(len(set(persons)), 
                 len(times_map), 
                 len(self.tokenizer.concept_map))
        )
        
        return feature_matrix, outcomes, times_map
      
    def _process_line(self, line):
        example = json.loads(line)
        dates = example['dates']
        unix_times = to_unixtime(dates)
        example['unix_times'] = unix_times

        # make sure visits are sorted by date
        sorted_visits = [v for d, v in sorted(zip(example['unix_times'], example['visits']))]
        example['visits'] = sorted_visits
        example['unix_times'] = sorted(example['unix_times'])
        example['dates'] = sorted(example['dates'])

        return example    
    
    def __getitem__(self, idx) -> Tuple[sparse.COO, int]:
        """Gets the feature-outcome pair located at idx.
        
        Parameters
        ----------
        idx : int
            The index of the dataset you'd like to access.
        
        Returns
        -------
        example : tuple of (sparse.COO, int)
            The feature-outcome pair at idx.
        """
        
        assert self.feature_tensor is not None
        assert self.outcomes is not None
        
        return self.feature_tensor[idx], self.outcomes[idx]
    
    def __len__(self):
        """Number of rows in the OMOPDatasetSparse object."""
        return len(self.outcomes)

class OMOPDatasetWindowed(OMOPDatasetSparse):
  
    """Generate a windowed representation of an OMOPDataset.

    Attributes
    ----------
    omop_dataset : OMOPDataset
        An OMOP dataset object.
    window_days : list of int
        A list of the length of each window, in days.
    tokenizer : ConceptTokenizer
        Object used to map concepts to matrix indices.
    feature_tensor : sparse.COO
        3D sparse matrix of persons x times x concepts.
    matrix_windowed : scipy.sparse.coo_matrix
        2D sparse matrix. Concepts aggregated by person.
    outcomes : list of int
        List of outcomes by person.
    times_map : dict of [int, int]
        Dictionary mapping unix times to integer IDs.    
    feature_names_windowed : list of str
        List of feature names, corresponding to windowed matrix columns.
    """

    def __init__(
        self, 
        omop_dataset: 'OMOPDataset', 
        window_days: List[int], 
        tokenizer: ConceptTokenizer = None
    ) -> None:

        """Initialize an OMOPDatasetWindowed object.

        Parameters
        ----------
        omop_dataset : OMOPDataset
            An OMOP dataset object.
        window_days : List[int]
            A list of window lengths in days.
        tokenizer : ConceptTokenizer
            Object used to map concepts to matrix indices.
        """
        
        super().__init__(omop_dataset, tokenizer)
        self.window_days = window_days
        self.matrix_windowed = None
        output = self._gen_windowed_data()
        self.matrix_windowed = output[0]
        self.feature_names_windowed = output[1]
        
        # In case data is split
        self.train = {'X': None, 'y': None}
        self.val = {'X': None, 'y': None}
        self.test = {'X': None, 'y': None}

    def _gen_windowed_data(self) -> Tuple[coo_matrix, List[str]]:

        """Aggregate concepts by person.

        Returns
        ------
        output : tuple of (coo_matrix, list of str)
             Feature count matrix by person and a list of feature names.
        """
        
        # Generated in OMOPDatasetSparse
        assert self.feature_tensor is not None
        assert self.times_map is not None
        assert self.omop_dataset is not None
                
        # Build time windows
        all_times = pd.to_datetime(np.array(from_unixtime(self.times_map.keys())))
        window_end = datetime.strptime(
          self.omop_dataset.cohort.params['training_end_date'], 
          '%Y-%m-%d')
        windowed_time_ixs = dict()
        for days in self.window_days:
            windowed_time_ixs[days] = self._gen_window_ixs(days, all_times, window_end)

        # Create matrix time slices, aggregate on the time dimension
        feature_matrix_slices = []
        feature_names = []
        for interval in sorted(windowed_time_ixs):
            feature_matrix_slices.append(
                self.feature_tensor[
                    :, windowed_time_ixs[interval][0]:windowed_time_ixs[interval][1], :
                ]
            )
            feature_names += [
                '{} - {} days'.format(name, interval)
                for name in self.tokenizer.concept_map.keys()
            ]
        feature_matrix_counts = hstack(
            [m.sum(axis=1).tocsr() for m in feature_matrix_slices]
        )

        return feature_matrix_counts, feature_names
    
    def _gen_window_ixs(
        self, 
        window_days: int, 
        all_times: pd.DatetimeIndex, 
        window_end: datetime) -> Tuple[int, int]:
        """Generate start and end matrix time dimension indices.

        Parameters
        ----------
        window_days : int
            The number of days. Calculates the start date of the window 
            based on the training end date specified in cohort creation.
        all_times : pd.DatetimeIndex
            A pandas datetime index representing all of the datetimes
            represented in the feature matrix.
        window_end : datetime
            A datetime representing the end of the time window.
        Returns
        -------
        tuple_ixs : tuple of int
            A tuple of integers specifying the (start, end) of the matrix 
            time dimension indices.
        """
        window_start = window_end - timedelta(days=window_days)
        return all_times.searchsorted(window_start), \
        all_times.searchsorted(window_end)

    def split(self, test_size: float = 0.2, random_state: int = 1) -> None:
      
      """Split the data into train, validation, and test datasets.
      
      Validation and test datasets are split roughly 50/50 of the 
      test_size parameter.
      test_size parameter.
      
      Parameters
      ----------
      test_size : float
          A number between 0 and 1. The proportion of data reserved for 
          validation + testing.
      random_state : int
          Used to control the stochastic split of the data.
      """
      
      assert self.matrix_windowed is not None
      assert self.outcomes is not None
      
      self.train['X'], X_test, self.train['y'], y_test = train_test_split(
          self.matrix_windowed, self.outcomes, test_size=test_size, random_state=random_state
      )
      
      val_size = int(X_test.shape[0] * 0.5)
      
      self.val['X'] = X_test[:val_size]
      self.val['y'] = y_test[:val_size]
      self.test['X'] = X_test[val_size:]
      self.test['y'] = y_test[val_size:]
      
      # Get rid of matrix_windowed and outcomes, since they're redundance
      # Also, this breaks __getitem__? So setting to None is reasonable.
      self.matrix_windowed = None
      self.outcomes = None
    
    def __getitem__(self, idx) -> Tuple[csr_matrix, int]:
        """Gets the feature-outcome pair located at idx.
        
        Parameters
        ----------
        idx : int
            The index of the dataset you'd like to access.
        
        Returns
        -------
        example : tuple of (scipy.sparse.csr_matrix, int)
            The feature-outcome pair at idx.
        """
        
        assert self.matrix_windowed is not None
        assert self.outcomes is not None
        
        return self.matrix_windowed.tocsr()[idx], self.outcomes[idx]
