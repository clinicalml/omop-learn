import time
from abc import ABC, abstractmethod


class OMOPDatasetBackend(ABC):
    
    @abstractmethod
    def __init__(self, config, connect_args):
        pass

    @abstractmethod
    def execute_query(self, sql):
        pass

    @abstractmethod
    def execute_queries(self, *sqls):
        pass

    @abstractmethod
    def build_table(self, schema_name, table_name, sql):
        pass

    @abstractmethod
    def reset_schema(self, schema):
        pass

    @abstractmethod
    def build_features(self, cohort, features, tokenizer, dataset_dir, is_visit_dataset):
        pass
 
    def build_cohort_table_from_sql_file(
        self, sql_path, table_name, params, schema_name, replace=False
    ):
        if replace or table_name not in self.get_all_tables(schema_name).values:
            t = time.time()
            if replace:
                print('Regenerating Table (replace=True)')
            else:
                print('Table not found in schema {}, regenerating'.format(schema_name))

            with open(sql_path, 'r') as f:
                cohort_generation_sql_raw = f.read()
            if params is not None:
                cohort_generation_sql = cohort_generation_sql_raw.format(**params)
            else:
                cohort_generation_sql = cohort_generation_sql_raw

            self.build_table('{}.{}'.format(schema_name, table_name), cohort_generation_sql)

            print('Regenerated Cohort in {} seconds'.format(time.time() - t))
        else:
            print('Table already exists, set replace=True to rebuild')
