import sys
sys.path.append('..')

import time
import sparse
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

import config 

class Feature():

    def __init__(
        self,
        feature_sql_file,
        feature_sql_params,
        temporal=True
    ):
        self.is_temporal = temporal
        
        self.params = feature_sql_params
        with open(feature_sql_file, 'r') as f:
            raw_sql = f.read()
            
        self._feature_sql_file = feature_sql_file
        self._sql_raw = raw_sql


    def __str__(self):
        return "Temporal feature extracted from {}".format(
            self._feature_sql_file
        )
    
    
class FeatureSet():
    
    def __init__(
        self,
        db,
        dtcols=(
            'feature_start_date',
            'person_start_date',
            'person_end_date'
            ),
        id_col = 'person_id',
        time_col = 'feature_start_date',
        feature_col = 'concept_name',
        ntmp_val_col = 'ntmp_val',
        unique_id_col = 'example_id'
        ):
        
        self._db = db
        self._dtcols = dtcols
        
        self.id_col = id_col
        self.time_col = time_col
        self.feature_col = feature_col
        self.ntmp_val_col = ntmp_val_col
        self.unique_id_col = unique_id_col
        
        self._temporal_features = []
        self._nontemporal_features = []
        
        self._temporal_feature_names = []
        self._temporal_feature_names_set = set()
        
        self._nontemporal_feature_names = []
        self._nontemporal_feature_names_set = set()
        
        self._spm_arr = []
        self._ntmp_spm = None
        self.id_map = None
        self.id_map_rev = None
        self.concept_map = None
        self.concept_map_rev = None
        self.ntmp_concept_map = None
        self.ntmp_concept_map_rev = None
        self.time_map = None
        self.time_map_rev = None

    def add(self, feature):
        if feature.is_temporal:
            self._temporal_features.append(feature)
        else:
            self._nontemporal_features.append(feature)

    def add_default_features(self, default_features, schema_name=None, cohort_name=None, temporal=True):
        fns = [
            './sql/Features/{}.sql'.format(f)
            for f in default_features
        ]
        for fn in fns:
            feature = Feature(
                fn,
                {
                    'cdm_schema':config.OMOP_CDM_SCHEMA,
                    'cohort_table':'{}.{}'.format(
                        schema_name,
                        cohort_name
                    )
                },
                temporal=temporal
            )
            self.add(feature)

            
    def get_feature_names(self):
        return self._temporal_feature_names +  self._nontemporal_feature_names

    def get_num_features(self):
        return len (
            self._temporal_feature_names +  self._nontemporal_feature_names
        )

    def build(self, cohort, cache_file='/tmp/store.csv', nontemporal_cache_file='/tmp/store_ntmp.csv', from_cached=False):
        joined_sql = "{} order by {} asc".format(
            " union all ".join(
                    f._sql_raw.format(
                        cdm_schema=config.OMOP_CDM_SCHEMA,
                        cohort_table='{}.{}'.format(
                            cohort._schema_name,
                            cohort._cohort_table_name
                        )
                    )
                for f in self._temporal_features
            ),
            ",".join([self.unique_id_col,      ## Order by unique_id
                      self.time_col, self.feature_col])    
        )
        if not from_cached:
            copy_sql = """
                copy 
                    ({query})
                to 
                    stdout 
                with 
                    csv {head}
            """.format(
                query=joined_sql,
                head="HEADER"
            )
            t = time.time()
            conn = self._db.engine.raw_connection()
            cur = conn.cursor()
            store = open(cache_file,'wb')
            cur.copy_expert(copy_sql, store)
            store.seek(0)
            print('Data loaded to buffer in {0:.2f} seconds'.format(
                time.time()-t
            ))
            
        t = time.time()
        store = open(cache_file,'rb')
    
        self.concepts = set()
        self.times = set()
        self.seen_ids=set()
        chunksize = int(2e6) 
        for chunk in pd.read_csv(store, chunksize=chunksize):
            self.concepts = self.concepts.union(set(chunk[self.feature_col].unique()))
            self.times = self.times.union(set(chunk[self.time_col].unique()))
            self.seen_ids = self.seen_ids.union(set(chunk[self.unique_id_col].unique()))
        self.times = sorted(list(self.times))
        self.concepts = sorted(list(self.concepts))
        self.seen_ids = sorted(list(self.seen_ids))
        print('Got Unique Concepts and Timestamps in {0:.2f} seconds'.format(
            time.time()-t
        ))
        
        t = time.time()
        store.seek(0)
        self.ids = cohort._cohort[self.unique_id_col].unique()
        self.id_map = {i:example_id for i,example_id in enumerate(self.seen_ids)}
        self.id_map_rev = {example_id:i for i,example_id in enumerate(self.seen_ids)}
        self.concept_map = {i:concept_name for i,concept_name in enumerate(self.concepts)}
        self.concept_map_rev = {concept_name:i for i,concept_name in enumerate(self.concepts)}
        self.time_map = {i:t for i,t in enumerate(self.times)}
        self.time_map_rev = {t:i for i,t in enumerate(self.times)}

        print('Created Index Mappings in {0:.2f} seconds'.format(
            time.time()-t
        ))

        t = time.time()
        last = None
        spm_stored = None
        spm_arr = []
        self.recorded_ids = set()
        for chunk_num, chunk in enumerate(pd.read_csv(store, chunksize=chunksize)):
            first = chunk.iloc[0][self.unique_id_col]
            vals = chunk[self.unique_id_col].unique()
            indices = np.searchsorted(chunk[self.unique_id_col], vals)
            self.recorded_ids = self.recorded_ids.union(set(vals))
            
            chunk.loc[:, self.feature_col] = chunk[self.feature_col].apply(self.concept_map_rev.get)
            chunk.loc[:, self.time_col] = chunk[self.time_col].apply(self.time_map_rev.get)

            df_split = [
                chunk.iloc[indices[i]:indices[i+1]]
                for i in range(len(indices) - 1)
            ] +  [chunk.iloc[indices[-1]:]]

            def gen_sparr(sub_df):
                sparr = coo_matrix(
                    (
                        np.ones(len(sub_df[self.feature_col])),
                        (sub_df[self.feature_col], sub_df[self.time_col])
                    ),
                    shape=(len(self.concepts), len(self.times))
                )
                return sparr
            spm_local = [gen_sparr(s) for s in df_split]
            if first == last:
                spm_local[0] += spm_stored
            else:
                if spm_stored is not None:
                    spm_arr.append(spm_stored)
            spm_arr += spm_local[:-1]
            spm_stored = spm_local[-1]
            # last = chunk.iloc[-1][sep_col]
            last = chunk.iloc[-1][self.unique_id_col]
        spm_arr.append(spm_stored)
        print(len(spm_arr))
        self._spm_arr = sparse.stack([sparse.COO.from_scipy_sparse(m) for m in spm_arr], 2)
        print('Generated Sparse Representation of Data in {0:.2f} seconds'.format(
            time.time() - t
        ))

        # Build nontemporal feature matrix

        if len(self._nontemporal_features) > 0:
            joined_sql = "{} order by {} asc".format(
                " union all ".join(
                        f._sql_raw.format(
                            cdm_schema=config.OMOP_CDM_SCHEMA,
                            cohort_table='{}.{}'.format(
                                cohort._schema_name,
                                cohort._cohort_table_name
                            )
                        )
                    for f in self._nontemporal_features
                ),
                ",".join([self.unique_id_col,      ## Order by unique_id
                          self.feature_col])    
            )
            if not from_cached:
                copy_sql = """
                    copy 
                        ({query})
                    to 
                        stdout 
                    with 
                        csv {head}
                """.format(
                    query=joined_sql,
                    head="HEADER"
                )
                t = time.time()
                conn = self._db.engine.raw_connection()
                cur = conn.cursor()
                store = open(nontemporal_cache_file,'wb')
                cur.copy_expert(copy_sql, store)
                store.seek(0)
                print('Nontemporal data loaded to buffer in {0:.2f} seconds'.format(
                    time.time()-t
                ))
                
            t = time.time()
            store = open(nontemporal_cache_file,'rb')
        
            self.ntmp_concepts = set()
            self.ntmp_seen_ids = set()
            chunksize = int(2e6) 
            for chunk in pd.read_csv(store, chunksize=chunksize):
                self.ntmp_seen_ids = self.ntmp_seen_ids.union(set(chunk[self.unique_id_col].unique()))
                self.ntmp_concepts = self.ntmp_concepts.union(set(chunk[self.feature_col].unique()))
            self.ntmp_concepts = sorted(list(self.ntmp_concepts))
            print('Got Unique Nontemporal Concepts in {0:.2f} seconds'.format(
                time.time()-t
            ))
            
            t = time.time()
            store.seek(0)
            self.ntmp_id_map = {i:example_id for i,example_id in enumerate(self.ntmp_seen_ids)}
            self.ntmp_id_map_rev = {example_id:i for i,example_id in enumerate(self.ntmp_seen_ids)}
            self.ntmp_concept_map = {i:concept_name for i,concept_name in enumerate(self.ntmp_concepts)}
            self.ntmp_concept_map_rev = {concept_name:i for i,concept_name in enumerate(self.ntmp_concepts)}

            print('Created Nontemporal Index Mappings in {0:.2f} seconds'.format(
                time.time()-t
            ))

            t = time.time()
            ntmp_data = []
            ntmp_unique_id = []
            ntmp_feature_id = []
            for chunk_num, chunk in enumerate(pd.read_csv(store, chunksize=chunksize)):
                chunk.loc[:, self.feature_col] = chunk[self.feature_col].apply(self.ntmp_concept_map_rev.get)
                ntmp_data.append(chunk[self.ntmp_val_col])
                ntmp_unique_id.append(chunk[self.unique_id_col])
                ntmp_feature_id.append(chunk[self.feature_col])
            ntmp_data = np.concatenate(ntmp_data)
            ntmp_unique_id = np.concatenate(ntmp_unique_id)
            ntmp_feature_id = np.concatenate(ntmp_feature_id)
            sparr = coo_matrix(
                (
                    ntmp_data,
                    (ntmp_unique_id, ntmp_feature_id)
                ),
                shape=(len(self.ids), len(self.ntmp_concepts))
            )
            self._ntmp_spm = sparr
            print('Generated Sparse Representation of Nontemporal Data in {0:.2f} seconds'.format(
                time.time() - t
            ))

    def get_sparr_rep(self):
        '''
        Returns the temporal sparse 3d matrix built by FeatureSet.build().
        '''
        return self._spm_arr

    def get_nontemporal_sparr_rep(self):
        '''
        Returns the nontemporal sparse 2d matrix built by FeatureSet.build().
        '''
        return self._ntmp_spm
        
def postprocess_feature_matrix(cohort, featureSet, training_end_date_col='training_end_date', include_nontemporal=False):
    '''
    Filter out feature names with no matching concepts, time indices beyond the training end date,
    and people without any temporal features.

    If there are nontemporal features, applies the same filter to the nontemporal feature matrix
    and returns it as output.
    '''

    feature_matrix_3d = featureSet.get_sparr_rep()
    outcomes = cohort._cohort.set_index('example_id').loc[
        sorted(featureSet.seen_ids)
    ]['y']
    good_feature_ix = [
        i for i in sorted(featureSet.concept_map)
        if '- No matching concept' not in featureSet.concept_map[i]
    ]
    good_feature_names = [
        featureSet.concept_map[i] for i in sorted(featureSet.concept_map)
        if '- No matching concept' not in featureSet.concept_map[i]
    ]
    good_time_ixs = [
        i for i in sorted(featureSet.time_map)
        if featureSet.time_map[i] <= cohort._cohort_generation_kwargs[training_end_date_col]
    ]
    feature_matrix_3d = feature_matrix_3d[good_feature_ix, :, :]
    feature_matrix_3d = feature_matrix_3d[:, good_time_ixs, :]
    feature_matrix_3d_transpose = feature_matrix_3d.transpose((2,1,0))
    total_events_per_person = feature_matrix_3d_transpose.sum(axis=-1).sum(axis=-1)

    # Retain only people for whom we have some *temporal* feature
    people_with_data_ix = np.where(total_events_per_person.todense() > 0)[0].tolist()
    feature_matrix_3d_transpose = feature_matrix_3d_transpose[people_with_data_ix, :, :]
    outcomes_filt = outcomes.loc[[featureSet.id_map[i] for i in people_with_data_ix]]
    remap = {
        'id':people_with_data_ix,
        'time':good_time_ixs,
        'concept':good_feature_ix
    }
    nontemporal_feature_matrix = featureSet.get_nontemporal_sparr_rep()
    if include_nontemporal and nontemporal_feature_matrix is not None:
        ntmp_people_with_data_ix = [featureSet.ntmp_id_map_rev[featureSet.id_map[i]] for i in people_with_data_ix]
        nontemporal_feature_matrix = coo_matrix(csr_matrix(nontemporal_feature_matrix)[ntmp_people_with_data_ix, :])
        nontemporal_feature_names = list(featureSet.ntmp_concept_map.values())
        return outcomes_filt, feature_matrix_3d_transpose, nontemporal_feature_matrix, remap, \
                good_feature_names, nontemporal_feature_names
    else:
        return outcomes_filt, feature_matrix_3d_transpose, remap, good_feature_names
