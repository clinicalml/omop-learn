from collections import defaultdict
import io
import json

from itertools import repeat
from multiprocessing import Pool
import numpy as np
import pandas as pd
from tqdm import tqdm
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

from omop_learn.backends.backend import OMOPDatasetBackend
from omop_learn.data.common import ConceptTokenizer


def preprocess_person_dict(person_dict):
    dates = sorted(person_dict['visits'].keys())
    new_visits = []
    person_dict['dates'] = []
    for date in dates:
        visit = person_dict['visits'][date]
        if len(visit) > 0:
            new_visits.append(visit)
            person_dict['dates'].append(date)
    person_dict['visits'] = new_visits
    return person_dict


def preprocess_visit_list(person_dict):
    dates = sorted(person_dict['visits'].keys())
    visit_list = []
    for date in dates:
        concept_list = person_dict['visits'][date]
        visit_dict = {'concepts': concept_list, 'date': date}
        visit_list.append(visit_dict)
    return visit_list


def load_person_data(person_dict, cdm_schema, tmp_features, ntmp_features, is_visit_dataset=False):
    '''
    Run query to load single person's features. Defined at module level 
    for use with multiprocessing.Pool().
    Args:
        person_dict: Dictionary for single person_id, extracted from 
                     Cohort object
        cdm_schema: String for CDM schema
        tmp_features: List of temporal feature objects
        ntmp_features: List of non-temporal feature objects
    Returns:
        person_dict with temporal and non-temporal features
    '''
    concept_set = set()
    person_id = person_dict['person_id']
    end_date = person_dict['end_date']
    tmp_combined = []
    ntmp_combined = []

    with engine.connect() as conn:
        for f in tmp_features:
            cur = conn.execute(f.raw_sql.format(
                cdm_schema=cdm_schema,
                person_id=person_id,
            ))
            tmp_combined += cur.fetchall()
        for f in ntmp_features:
            cur = conn.execute(f.raw_sql.format(
                cdm_schema=cdm_schema,
                person_id=person_id,
                end_date=end_date,
            ))
            ntmp_combined += cur.fetchall()
    
    visit_dict = defaultdict(list)
    for concept_name, visit_date in tmp_combined:
        if visit_date <= end_date:
            visit_dict[visit_date].append(concept_name) 
            concept_set.add(concept_name)
    person_dict["visits"] = visit_dict
    for concept_val, concept_name in ntmp_combined:
        person_dict[concept_name] = concept_val
    if is_visit_dataset:
        ret_col = preprocess_visit_list(person_dict)
    else:
        ret_col = preprocess_person_dict(person_dict)
    return ret_col, concept_set 


class PostgresBackend(OMOPDatasetBackend):

    def __init__(self, config, connect_args=None, echo=False):
        global engine
        if connect_args is not None:
            self.engine = sqlalchemy.create_engine(
                config.path,
                echo=echo,
                connect_args=connect_args
            )
            engine = self.engine
        else:
            self.engine = sqlalchemy.create_engine(
                config.path,
                echo=echo
            )   
            engine = self.engine         
        self.meta = sqlalchemy.MetaData(
            bind=self.engine
        )
        self.cdmMeta = sqlalchemy.MetaData(
            bind=self.engine, 
            schema="cdm"
        )
        self.selfMeta = sqlalchemy.MetaData(
            bind=self.engine, 
            schema=config.prefix_schema
        )

    @contextmanager
    def _session_scope(self):
        Session = sessionmaker(bind=self.engine)
        session = Session()

        try:
            yield session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()
    
    def execute_query(self, sql):
        '''
        Run sql and dump results to a Pandas Dataframe.
        Args:
            sql: A SQL command to fetch desired results
        Returns:
            pandas.DataFrame with query results
        '''
        return pd.read_sql(sql, self.engine)

    def fast_query(self, sql):
        '''
        Run sql and dump results to a Pandas Dataframe.
        Runs faster by using an intermediate string buffer.
        Args:
            sql: A SQL command to fetch desired results
        Returns:
            pandas.DataFrame with query results
        '''
        copy_sql = """
            copy
                ({query})
            to
                stdout
            with
                csv {head}
        """.format(
            query=sql,
            head="HEADER"
        )
        conn = self.engine.raw_connection()
        cur = conn.cursor()
        store = io.StringIO()
        cur.copy_expert(copy_sql, store)
        store.seek(0)
        df = pd.read_csv(store, engine='python')
        return df

    def execute_queries(self, *sqls):
        '''
        Run each command in sqls.
        Args:
            sqls: Any number of SQL command strings
        Returns:
            None
        '''
        with self._session_scope() as session:
            for sql in sqls:
                session.execute(sql)
            session.commit()
            print('Executed {} SQLs'.format(
                len(sqls)
            ))

    def build_table(self, schema_name, table_name, sql):
        '''
        (Re)builds table_name using sql.
        Note that the table will be deleted if it already exists.

        Args:
            table_name: The name of the table to be rebuilt
            sql: A SQL command string to create the table table_name
        Returns:
            None
        '''
        with self._session_scope() as session:

            drop_sql = f"drop table if exists {schema_name}.{table_name}"
            session.execute(sqlalchemy.text(drop_sql))

            session.execute(sqlalchemy.text(sql))
            session.commit()

    def get_all_tables(self, schema=None):
        '''
        Return a pandas.Series of all tables in schema

        Args:
            schema: The schema to look for tables in
        Returns:
            pandas.Series of all tables in schema
        '''
        sql = """
            select table_name
            from information_schema.tables
            {};
        """.format(
            '' if schema is None
            else "where table_schema = '{}'".format(schema)
        )
        return self.query(sql)['table_name']

    def create_schema(self, schema):
        '''
        Create schema in database if not exists
        
        Args:
            schema: Schema to create
        Returns:
            None
        '''
        self.execute_queries('create schema if not exists {}'.format(schema))

    def reset_schema(self, schema):
        '''
        Drop schema in database if exists
        
        Args:
            schema: Schema to drop
        Returns:
            None
        '''
        self.execute_queries('drop schema if exists {} cascade'.format(schema))

    # https://docs.sqlalchemy.org/en/14/core/pooling.html#using-connection-pools-with-multiprocessing-or-os-fork
    def init_worker(self):
        self.engine.pool = self.engine.pool.recreate()
    
    def build_features(
        self,
        config,
        cohort,
        features,
        tokenizer,
        dataset_dir,
        is_visit_dataset,
        num_workers
    ):
        concept_set = set()
        json_path = dataset_dir / "data.json"
        person_dicts = cohort.cohort.to_dict('records')
        tmp_features = list(filter(lambda f: f.temporal, features))
        ntmp_features = list(filter(lambda f: not f.temporal, features))

        print("Generating temporal and non-temporal features...")
        with Pool(processes=num_workers, initializer=self.init_worker) as p:
            mp_result_col = p.starmap(
                load_person_data, tqdm(
                    zip(person_dicts, repeat(config.cdm_schema), repeat(tmp_features), repeat(ntmp_features), repeat(is_visit_dataset)),
                    total=len(person_dicts),
                )
            )
        
        print("Writing features to data.json...")
        with open(json_path, "w") as fh:
            for result in tqdm(mp_result_col):
                if is_visit_dataset:
                    # Result is a list of visits
                    for cur_visit in result[0]:
                        fh.write(json.dumps(cur_visit, default=str) + "\n")
                else:
                    # Result is a single patient dict
                    cur_patient = result[0]
                    if len(cur_patient["visits"]) > 0:
                        fh.write(json.dumps(cur_patient, default=str) + "\n")

        if tokenizer is None:
            for result in mp_result_col:
                concept_set.update(result[1])
            tokenizer = ConceptTokenizer(concept_set)
        
        return tokenizer

    def _count_df_rows(self, pd_io_parser: pd.io.parsers.TextFileReader) -> int:
        '''
        Count the number of non-NA rows in a pandas df.

        Args:
            pd_io_parser: pd.DataFrame as returned by pandas read_csv
        
        Returns:
            num_rows: A count of the number of rows without NAs
        '''
        num_rows = sum([x.dropna().shape[0] for x in pd_io_parser])
        return num_rows
