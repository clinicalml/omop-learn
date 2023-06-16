import os
from dotenv import load_dotenv

from omop_learn.backends.bigquery import BigQueryBackend
from omop_learn.data.cohort import Cohort
from omop_learn.data.feature import Feature
from omop_learn.utils.config import Config
from omop_learn.omop import OMOPDataset


load_dotenv("bigquery.env")

config = Config({
    "project_name": os.getenv("PROJECT_NAME"),
    "cdm_schema": os.getenv("CDM_SCHEMA"),
    "aux_cdm_schema": os.getenv("AUX_CDM_SCHEMA"),
    "prefix_schema": os.getenv("PREFIX_SCHEMA"),
    "datasets_dir": os.getenv("OMOP_DATASETS_DIR"),
    "models_dir": os.getenv("OMOP_MODELS_DIR")
})

# Set up database, reset schemas as needed
backend = BigQueryBackend(config)
backend.reset_schema(config.prefix_schema)
backend.create_schema(config.prefix_schema)

cohort_params = {
    "cohort_table_name": "eol_cohort",
    "schema_name": config.prefix_schema,
    "cdm_schema": config.cdm_schema,
    "aux_data_schema": config.aux_cdm_schema,
    "training_start_date": "2009-01-01",
    "training_end_date": "2009-12-31",
    "gap": "3 month",
    "outcome_window": "6 month",
}
sql_dir = "examples/eol/bigquery_sql"
sql_file = open(f"{sql_dir}/gen_EOL_cohort.sql", 'r')
cohort = Cohort.from_sql_file(sql_file, backend, params=cohort_params)

feature_paths = [f"{sql_dir}/drugs.sql"]
feature_names = ["drugs"]
features = [Feature(n, p) for n, p in zip(feature_names, feature_paths)]

ntmp_feature_paths = [f"{sql_dir}/age.sql", f"{sql_dir}/gender.sql"]
ntmp_feature_names = ["age", "gender"]
features.extend([Feature(n,p,temporal=False) for n, p in zip(ntmp_feature_names, ntmp_feature_paths)])

init_args = {
    "config" : config,
    "name" : "bigquery_eol_cohort",
    "cohort" : cohort,
    "features": features,
    "backend": backend,
    "is_visit_dataset": False,
    "num_workers": 10
}

dataset = OMOPDataset(**init_args)
