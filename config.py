
# Database Setup
DB_NAME = 'localhost/omop_v3'
PG_USERNAME = 'omop_admins'
PG_PASSWORD = '***'

# Schemas
OMOP_CDM_SCHEMA = 'cdm' # schema holding standard OMOP tables
CDM_AUX_SCHEMA = 'cdm_aux' # schema to hold auxilliary tables not tied to a particular schema
CDM_VERSION = 'v5.3.1'

# SQL Paths
SQL_PATH_COHORTS = 'sql/Cohorts' # path to SQL scripts that generate cohorts
SQL_PATH_FEATURES = 'sql/Features' # path to SQL scripts that generate features

# Cache
DEFAULT_SAVE_LOC = '/tmp/' # where to save temp files
