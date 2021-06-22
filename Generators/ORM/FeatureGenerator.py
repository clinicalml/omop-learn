import sys
sys.path.append('..')

import time
import sparse
import pandas as pd
import numpy as np
import csv
import psutil
import sparse

from sqlalchemy import union_all
from sqlalchemy import String
from sqlalchemy.sql.expression import select, join, text, case, cast
from Generators.ORM.CohortGenerator import CohortTable
from ORMTables.cdm_6_0 import ConditionOccurrence, ProcedureOccurrence, DrugExposure, VisitOccurrence, Provider, Concept
from tqdm import tqdm

import config 

def mb_used():
    return psutil.Process().memory_info().rss / 1024 ** 2

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
        unique_id_col = 'example_id'
        ):
        
        self._db = db
        self._dtcols = dtcols
        
        self.id_col = id_col
        self.time_col = time_col
        self.feature_col = feature_col
        self.unique_id_col = unique_id_col
        
        self._temporal_features = []
        self._nontemporal_features = []
        
        self._temporal_feature_names = []
        self._temporal_feature_names_set = set()
        
        self._nontemporal_feature_names = []
        self._nontemporal_feature_names_set = set()
        
        self._spm_arr = []
        self.id_map = None
        self.id_map_rev = None
        self.concept_map = None
        self.concept_map_rev = None
        self.time_map = None
        self.time_map_rev = None
        
        # SqlAlchemy feature definitions
        condition_features = select([
            CohortTable.example_id, ConditionOccurrence.person_id, 
            (cast(ConditionOccurrence.condition_concept_id, String) + 
             ' - condition - ' + 
             case([(Concept.concept_name == None, 'no match')], else_ = Concept.concept_name)).label('concept_name'), 
            ConditionOccurrence.condition_start_datetime.label(dtcols[0]), 
            CohortTable.start_date.label(dtcols[1]), 
            CohortTable.end_date.label(dtcols[2])
        ])\
        .select_from(
            join(ConditionOccurrence, CohortTable, ConditionOccurrence.person_id == CohortTable.person_id)\
            .join(Concept, Concept.concept_id == ConditionOccurrence.condition_concept_id)
        )

        procedure_features = select([
            CohortTable.example_id, ProcedureOccurrence.person_id, 
            (cast(ProcedureOccurrence.procedure_concept_id, String) + 
             ' - procedure - ' + 
             case([(Concept.concept_name == None, 'no match')], else_ = Concept.concept_name)).label('concept_name'), 
            ProcedureOccurrence.procedure_datetime.label(dtcols[0]), 
            CohortTable.start_date.label(dtcols[1]), 
            CohortTable.end_date.label(dtcols[2])
        ])\
        .select_from(
            join(ProcedureOccurrence, CohortTable, ProcedureOccurrence.person_id == CohortTable.person_id)\
            .join(Concept, Concept.concept_id == ProcedureOccurrence.procedure_concept_id)
        )

        drug_features = select([
            CohortTable.example_id, DrugExposure.person_id,
            (cast(DrugExposure.drug_concept_id, String) + 
             ' - drug - ' + 
             case([(Concept.concept_name == None, 'no match')], else_ = Concept.concept_name)).label('concept_name'), 
            DrugExposure.drug_exposure_start_datetime.label(dtcols[0]), 
            CohortTable.start_date.label(dtcols[1]), 
            CohortTable.end_date.label(dtcols[2])
        ])\
        .select_from(
            join(DrugExposure, CohortTable, DrugExposure.person_id == CohortTable.person_id)\
            .join(Concept, Concept.concept_id == DrugExposure.drug_concept_id)
        )

        specialty_features = select([
            CohortTable.example_id, VisitOccurrence.person_id,
            (cast(Provider.specialty_concept_id, String) + 
             ' - specialty - ' + 
             case([(Concept.concept_name == None, 'no match')], else_ = Concept.concept_name)).label('concept_name'), 
            VisitOccurrence.visit_start_date.label(dtcols[0]), 
            CohortTable.start_date.label(dtcols[1]), 
            CohortTable.end_date.label(dtcols[2])   
        ])\
        .select_from(
            join(VisitOccurrence, CohortTable, VisitOccurrence.person_id == CohortTable.person_id)\
            .join(Provider, VisitOccurrence.provider_id == Provider.provider_id)\
            .join(Concept, Concept.concept_id == Provider.specialty_concept_id)
        )

        self.feature_dict = {
            'Conditions': {'sql': condition_features, 'is_temporal': True},
            'Procedures': {'sql': procedure_features, 'is_temporal': True},
            'Drugs':      {'sql': drug_features, 'is_temporal': True}, 
            'Specialty': {'sql': specialty_features, 'is_temporal': True}
        }

    def add(self, feature):
        if feature['is_temporal']:
            self._temporal_features.append(feature['sql'])
        else:
            self._nontemporal_features.append(feature['sql'])

    def add_default_features(self, default_features):
        for domain in default_features:
            self.add(self.feature_dict[domain])
            
    def get_feature_names(self):
        return self._temporal_feature_names + self._nontemporal_feature_names

    def get_num_features(self):
        return len (
            self._temporal_feature_names + self._nontemporal_feature_names
        )

    def build(self, cohort, cache_file='/tmp/store.csv', from_cached=False):
        if not from_cached:
            with self._db.session.session_manager() as session:
                t = time.time()
                store = open(cache_file, 'w', encoding='utf-8')
                outcsv = csv.writer(store)
                union_stmt = session.query(union_all(*self._temporal_features).alias('u')).subquery('union_stmt')
                result = session.query(union_stmt)\
                .order_by(union_stmt.c[self.unique_id_col], union_stmt.c[self.time_col], union_stmt.c[self.feature_col])\
                .yield_per(int(1e3))
                outcsv.writerow(x['name'] for x in result.column_descriptions) # Write header first
                for row in result:
                    outcsv.writerow(row)
                store.seek(0)
                store.close()
                print('Data loaded to buffer in {0:,.2f} seconds'.format(time.time()-t))
            
        t = time.time()
        store = open(cache_file,'rb')
    
        self.concepts = set()
        self.times = set()
        self.seen_ids=set()
        chunksize = int(1e5) 
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
        coords = [
            [], # concepts
            [], # times
            []  # person IDs
        ]
        store.seek(0)
        csv_store = tqdm(pd.read_csv(store, usecols=[self.unique_id_col, self.feature_col, self.time_col], chunksize=chunksize))
        for chunk in csv_store:
            coords[0].extend(chunk[self.feature_col].apply(self.concept_map_rev.get).tolist())
            coords[1].extend(chunk[self.time_col].apply(self.time_map_rev.get).tolist())
            coords[2].extend(chunk[self.unique_id_col].apply(self.id_map_rev.get).tolist())
            csv_store.set_description('{:,.2f} MB consumed'.format(mb_used()))

        self._spm_arr = sparse.COO(coords, 1, shape=(len(self.concepts), len(self.times), len(self.seen_ids)))
        print('Generated Sparse Representation of Data in {0:.2f} seconds'.format(
            time.time() - t
        ))

    def get_sparr_rep(self):
        return self._spm_arr
        


def postprocess_feature_matrix(cohort, featureSet, training_end_date_col='training_end_date'):
    feature_matrix_3d = featureSet.get_sparr_rep()
#     outcomes = cohort._cohort.set_index('person_id').loc[
#         sorted(featureSet.seen_ids)
#     ]['y']
    outcomes = cohort._cohort.set_index('example_id').loc[
        sorted(featureSet.seen_ids)
    ]['y']
#     outcomes = cohort._cohort['y']
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
        if featureSet.time_map[i] <= str(getattr(cohort, training_end_date_col))
    ]
    feature_matrix_3d = feature_matrix_3d[good_feature_ix, :, :]
    feature_matrix_3d = feature_matrix_3d[:, good_time_ixs, :]
    feature_matrix_3d_transpose = feature_matrix_3d.transpose((2,1,0))
    total_events_per_person = feature_matrix_3d_transpose.sum(axis=-1).sum(axis=-1)
    people_with_data_ix = np.where(total_events_per_person.todense() > 0)[0].tolist()
    feature_matrix_3d_transpose = feature_matrix_3d_transpose[people_with_data_ix, :, :]
#     outcomes_filt = outcomes.loc[[featureSet.id_map[i] for i in people_with_data_ix]]
    outcomes_filt = outcomes.loc[[featureSet.id_map[i] for i in people_with_data_ix]]
    remap = {
        'id':people_with_data_ix,
        'time':good_time_ixs,
        'concept':good_feature_ix
    }
    return outcomes_filt, feature_matrix_3d_transpose, remap, good_feature_names
