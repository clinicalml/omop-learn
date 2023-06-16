from collections import defaultdict
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.Catalog import dropGlobalTempView

from omop_learn.backends.backend import OMOPDatasetBackend
from omop_learn.data.common import ConceptTokenizer


class SparkBackend(OMOPDatasetBackend):  # noqa

    def __init__(self, config, connect_args=None):
        conf = SparkConf()
        for key, value in connect_args.items(): conf.set(key, value) 
        self.session = SparkSession.builder.appName(config.prefix_schema).config(conf).getOrCreate()
    
    def execute_query(self, sql):
        '''
        Run sql and dump results to a Spark DataFrame.

        Args:
            sql: A SQL command to fetch desired results
        Returns:
            pyspark.sql.DataFrame with query results
        '''
        return self.session.sql(sql)

    def execute_queries(self, *sqls):
        '''
        Run each command in sqls.
        Args:
            sqls: Any number of SQL command strings
        Returns:
            None
        '''
        for sql in sqls:
            self.session.sql(sql)
        print('Executed {} SQLs'.format(len(sqls)))

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
        table_df = self.session.sql(sql)
        table_df.createOrReplaceGlobalTempView(table_name)

    def get_all_tables(self, schema="global_temp"):
        '''
        Args:
            schema: The schema to look for tables in
        Returns:
            pandas.Series of all tables in schema
        '''
        return self.session.tableNames(schema)

    def reset_schema(self, schema):
        dropGlobalTempView(schema)

    def build_features(self, features, dataset_dir, cache_name):      
        """Execute feature query and output to a csv file.
        
        Args:
            db (omop.utils.dbutils.Database): Database class.
            features (list of omop.data.feature.Feature): List of feature objects.
            cache_name (str): Name of the output csv file.
        
        Returns:
            csv_cache_file (pathlib.PosixPath): Path object to the output csv file.
        """
        # write feature queries to disk, one line per concept
        csv_cache_file = dataset_dir /  cache_name

        query = "{} order by {} asc".format(
                " union all ".join(
                    f.raw_sql.format(
                        cdm_schema=self.config.cdm_schema,
                        cohort_table='{}.{}'.format(
                            self.cohort.params['schema_name'],
                            self.cohort.params['cohort_table_name'],
                        ),
                    )
                    for f in features
                ),
                ",".join(["example_id"
                ])
            )
        features_df = self.sql(query)
        features_df.toPandas().to_csv(csv_cache_file)
        return csv_cache_file

    def _count_df_rows(self, pd_io_parser: pd.io.parsers.TextFileReader) -> int:
        """Count the number of non-NA rows in a pandas df.

        Parameters
        ----------
        pd_io_parser : pandas.io.parsers.TextFileReader
            A df as returned by pandas read_csv.
        
        Returns
        -------
        num_rows : int
            A count of the number of rows without NAs.
        """
        
        num_rows = sum([x.dropna().shape[0] for x in pd_io_parser])
        return num_rows

    def _csv_to_json(self, cohort, is_visit_dataset, csv_filename):

        store = open(csv_filename,'rb')

        concept_set = set()
        chunksize = int(2e6)

        cur_patient = {'visits': defaultdict(list)}

        json_path = self.dataset_dir / "data.json"
        unique_id_col = "example_id"
        time_col = "feature_start_date"
        end_col = "person_end_date"
        feature_col = "concept_name"

        i = 0
        cur_uniq_id = 0
        max_num_visits = 0

        # matching on example_ids and collating both sets of features together
        df = pd.read_csv(store, chunksize=chunksize)
        num_rows = self._count_df_rows(df)
        store.seek(0)
        df = pd.read_csv(store, chunksize=chunksize) # reset iterator
        with open(json_path, 'w', buffering=1) as fh:
            for chunk in tqdm(df):
                chunk.dropna(inplace=True)
                chunk.index = np.arange(i, len(chunk)+i)
                for _ in range(len(chunk)):
                    idx = chunk[unique_id_col][i]
                    feat = chunk[feature_col][i]
                    rowtime = chunk[time_col][i]
                    end_date = chunk[end_col][i]

                    # the rows are sorted by unique_id, so we'll never see cur_uniq_id person later.
                    # the row we're on now is already the next person's data.
                    # so write the old info out and reset for the next one.
                    if idx != cur_uniq_id or i == num_rows-1:
                        outcome = cohort.cohort['y'][cur_uniq_id].item()
                        cur_patient['y'] = outcome
                        cur_patient['cohort_id'] = cur_uniq_id
                        cur_patient['person_id'] = cohort.cohort['person_id'][cur_uniq_id].item()

                        num_visits = len(cur_patient['visits'].keys())

                        # add the last row before writing out
                        if i == num_rows - 1 and rowtime <= end_date:
                            cur_patient['visits'][rowtime].append(feat)

                        if num_visits > 0 and is_visit_dataset:
                            self._write_visit_line(fh, cur_patient)
                        elif num_visits > 0:
                            self._write_patient_line(fh, cur_patient)

                        max_num_visits = max(num_visits, max_num_visits)

                        # now step forward
                        cur_uniq_id = idx.item() # .item() for serialize
                        cur_patient = {'visits': defaultdict(list)}

                    # only keep visits before training end date for this patient
                    # todo: use datetimes instead of lexicographical ordering
                    if rowtime <= end_date:
                        cur_patient['visits'][rowtime].append(feat)
                        concept_set.add(feat)

                    i += 1 # dframe indices still increase across chunks

        if tokenizer is None:
            tokenizer = ConceptTokenizer(concept_set)

        return json_path, tokenizer
    
    def _write_json_patient_line(self, fh, cur_patient):
        dates = sorted(cur_patient['visits'].keys())
        new_visits = []
        cur_patient['dates'] = []
        for date in dates:
            visit = cur_patient['visits'][date]
            if len(visit) > 0:
                new_visits.append(visit)
                cur_patient['dates'].append(date)
        cur_patient['visits'] = new_visits
        if len(cur_patient['visits']) > 0:
            fh.write(json.dumps(cur_patient)+"\n")

    def _write_json_visit_line(self, fh, cur_patient):
        dates = sorted(cur_patient['visits'].keys())
        for date in dates:
            concept_list = cur_patient['visits'][date]
            out_dict = {'concepts': concept_list, 'date': date}
            fh.write(json.dumps(out_dict)+"\n")

    def _append_to_json(self, json_path, csv_cache_file):
        chunksize = int(2e4)

        new_json_path = self.dataset_dir / "newdata.json"

        with open(json_path, 'r') as fh, open(csv_cache_file, 'rb') as fh2, \
             open(new_json_path, 'w', buffering=1) as ofh:

            i = 0
            cur_uniq_id = 0
            unique_id_col = 'example_id'
            val_col = 'ntmp_val'
            name_col = 'concept_name'

            patient_dict = json.loads(fh.readline().rstrip())
            cur_uniq_id = patient_dict['cohort_id']

            df = pd.read_csv(fh2, chunksize=chunksize)
            num_rows = self._count_df_rows(df)
            fh2.seek(0)
            df = pd.read_csv(fh2, chunksize=chunksize)
            for chunk in df:
                done = False
                chunk.dropna(inplace=True)
                chunk.index = np.arange(i, len(chunk)+i)              
                for _ in range(len(chunk)):
                    idx = chunk[unique_id_col][i]
                    if idx > cur_uniq_id and i < num_rows-1:
                        ofh.write(f"{json.dumps(patient_dict)}\n")

                        # read in new patient and update cur_uniq_id
                        line = fh.readline()
                        if not line:
                            done = True
                            break
                        patient_dict = json.loads(line.rstrip())
                        cur_uniq_id = patient_dict['cohort_id']

                    if idx == cur_uniq_id:
                        feat_name = chunk[name_col][i]
                        patient_dict[feat_name] = chunk[val_col][i].item()
                        if i == num_rows-1:
                            ofh.write(f"{json.dumps(patient_dict)}\n")
                    i += 1
                if done:
                    break

        # move new data file to overwrite data.json
        new_json_path.rename(json_path)
        return json_path
