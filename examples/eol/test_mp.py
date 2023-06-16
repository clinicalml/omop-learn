import sqlalchemy
from multiprocessing import Pool
import multiprocessing as mp
import os
from dotenv import load_dotenv
from itertools import repeat
from collections import defaultdict
from tqdm import tqdm

# from omop_learn.backends.bigquery import BigQueryBackend
from omop_learn.backends.postgres import PostgresBackend
from omop_learn.data.cohort import Cohort
from omop_learn.data.feature import Feature
from omop_learn.utils.config import Config


load_dotenv()

config = Config({
    "path": os.getenv("DATABASE_PATH"),
    "cdm_schema": os.getenv("CDM_SCHEMA"),
    "aux_cdm_schema": os.getenv("AUX_CDM_SCHEMA"),
    "prefix_schema": "buendia",
    "datasets_dir": os.getenv("OMOP_DATASETS_DIR"),
    "models_dir": os.getenv("OMOP_MODELS_DIR")
})

# Set up database, reset schemas as needed
backend = PostgresBackend(config, connect_args = {"host": "/var/run/postgresql/"})

cohort_params = {
    "cohort_table_name": "eol_cohort_test",
    "schema_name": config.prefix_schema,
    "cdm_schema": config.cdm_schema,
    "aux_data_schema": config.aux_cdm_schema,
    "training_start_date": "2009-01-01",
    "training_end_date": "2009-12-31",
    "gap": "3 month",
    "outcome_window": "6 month",
}
sql_dir = "examples/eol/postgres_sql"
sql_file = open(f"{sql_dir}/gen_EOL_cohort.sql", 'r')
cohort = Cohort.from_sql_file(sql_file, backend, params=cohort_params)

feature_paths = [f"{sql_dir}/drugs.sql"]
feature_names = ["drugs"]
features = [Feature(n, p) for n, p in zip(feature_names, feature_paths)]

ntmp_feature_paths = [f"{sql_dir}/age.sql", f"{sql_dir}/gender.sql"]
ntmp_feature_names = ["age", "gender"]
features.extend([Feature(n,p,temporal=False) for n, p in zip(ntmp_feature_names, ntmp_feature_paths)])


engine = sqlalchemy.create_engine("postgresql://localhost/omop_v7", connect_args = {"host": "/var/run/postgresql/"})

def load_person_data(person_dict, cdm_schema, tmp_features, ntmp_features):
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
        visit_dict[visit_date].append(concept_name)        
    person_dict["visits"] = list(visit_dict.values())
    person_dict["dates"] = list(visit_dict.keys())
    for concept_val, concept_name in ntmp_combined:
        person_dict[concept_name] = concept_val
    return person_dict

# https://docs.sqlalchemy.org/en/14/core/pooling.html#using-connection-pools-with-multiprocessing-or-os-fork
def init_worker():
    engine.pool = engine.pool.recreate()

person_dicts = cohort.cohort.to_dict('records')
tmp_features = list(filter(lambda f: f.temporal, features))
ntmp_features = list(filter(lambda f: not f.temporal, features))

with Pool(10, initializer=init_worker) as p:
    result = p.starmap(
        load_person_data, tqdm(
            zip(person_dicts, repeat(config.cdm_schema), repeat(tmp_features), repeat(ntmp_features)),
            total=len(person_dicts),
        )
    )