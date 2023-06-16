import pandas as pd
import pathlib
from pathlib import Path
import json


class Cohort(object):
    def __init__(self, sql_template, params, cohort):
        self.cohort = cohort
        self.sql_template = sql_template
        self.params = params

    def __len__(self):
        return len(self.cohort)

    @classmethod
    def from_sql_file(cls, sql_file, backend, params):
        if isinstance(sql_file, str):
            try:
                sql_file = open(sql_file, "r")
            except:
                print(f"Error opening sql file {sql_file}")
                raise
        sql_template = sql_file.read()
        sql_script = sql_template.format(**params)
        assert "cohort_table_name" in params, "Cohort table must have a name"
        tblname = params["cohort_table_name"]
        schema_name = params["schema_name"]
        backend.build_table(schema_name, tblname, sql_script)
        cohort = backend.execute_query(f"select * from {schema_name}.{tblname}")
        return cls(sql_template, params, cohort) # todo: add config here too?

    @classmethod
    def from_prebuilt(cls, backend, params):
        tblname = params["cohort_table_name"]
        schema_name = params["schema_name"]
        cohort = backend.execute_query(f"select * from {schema_name}.{tblname}")
        return cls(None, params, cohort)

    @classmethod
    def from_files(cls, cohort_csv_path, cohort_params_path):
        cohort = pd.read_csv(cohort_csv_path)

        # todo: do this checking elsewhere nicely
        if isinstance(cohort_params_path, str):
            fh = open(cohort_params_path, 'r')
        elif isinstance(cohort_params_path, Path):
            fh = cohort_params_path.open('r')

        cohort_params = json.load(fh)
        sql_template = cohort_params.pop("sql_template")
        return cls(sql_template, cohort_params, cohort)

    def serialize(self, dirpath: pathlib.PosixPath = None, with_data: bool = True) -> None:
        # save cohort dataframe to one file
        if with_data:
            self.cohort.to_csv(dirpath / "cohort.csv")

        # save params / sql to other file
        # clone params and add raw sql
        params = {k: v for k,v in self.params.items()}
        params['sql_template'] = self.sql_template
        json.dump(params, (dirpath / "cohort_params.json").open('w'))
