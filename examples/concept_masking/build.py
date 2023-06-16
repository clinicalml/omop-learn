from omop_learn.backends.postgres import PostgresBackend
from omop_learn.data.cohort import Cohort
from omop_learn.data.feature import Feature
from omop_learn.utils.config import Config
from omop_learn.omop import OMOPVisitDataset


config = Config({
    "path": "postgresql://localhost/omop_v7",
    "cdm_schema": "cdm",
    "aux_cdm_schema": "cdm_aux",
    "prefix_schema": "hjl"
})

backend = PostgresBackend(config)

cohort_params = {
    "cohort_table_name": "visit_pretrain",
    "schema_name": config.prefix_schema,
    "aux_data_schema": config.aux_cdm_schema,
    "training_start_date": "2000-01-01",
    "training_end_date": "2019-01-01",
}
sql_file = open("sql/cohort.sql", 'r')
cohort = Cohort.from_sql_file(sql_file, config, backend, params=cohort_params)
print(f"Cohort length {len(cohort)}")

feature_paths = ["sql/drugs.sql", "sql/conditions.sql", "sql/procedures.sql"]
feature_names = ["drugs", "conditions", "procedures"]
features = [Feature(n, p) for n, p in zip(feature_names, feature_paths)]

init_args = {
    "name" : "visit_pretrain2",
    "config" : config,
    "cohort" : cohort,
    "features": features,
    "backend": backend,
}

dataset = OMOPVisitDataset(**init_args)
