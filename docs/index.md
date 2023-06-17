# omop-learn


## Introductory Tutorial

`omop-learn` allows [OMOP-standard (CDM v5.3 and v6)](https://github.com/OHDSI/CommonDataModel/wiki) medical data like claims and EHR information to be processed efficiently for predictive tasks. The library allows users to precisely define cohorts of interest, patient-level time series features, and target variables of interest. Relevant data is automatically extracted and surfaced in formats suitable for most machine learning algorithms, and the (often extreme) sparsity of patient-level data is fully taken into account to provide maximum performance. 

The library provides several benefits for modeling, both in terms of ease of use and performance:
* All that needs to be specified are cohort and outcome definitions, which can often be done using simple SQL queries.
* Our fast data ingestion and transformation pipelines allow for easy and efficient tuning of algorithms. We have seen significant improvements in out-of-sample performance of predictors after hyperparameter tuning that would take days with simple SQL queries but minutes with `omop-learn`.
* We modularize the data extraction and modeling processes, allowing users to use new models as they become available with very little modification to the code. Tools ranging from simple regression to deep neural net models can easily be substituted in a plug-and-play manner.

`omop-learn` serves as a modern python alternative to the [PatientLevelPrediction R library](https://github.com/OHDSI/PatientLevelPrediction). We allow seamless integration of many Python-based machine learning and data science libraries by supporting generic `sklearn`-style classifiers. Our new data storage paradigm also allows for more on-the-fly feature engineering as compared to previous libraries. 

In this tutorial, we walk through the process of using `omop-learn` for an end-of-life prediction task for synthetic Medicare patients with clear applications to improving palliative care. The code used can also be found in the [example notebook](https://github.com/clinicalml/omop-learn/blob/master/examples/eol/sard_eol.ipynb), and can be run on your own data as you explore `omop-learn`. The control flow diagram below also links to relevant sections of the library documentation.
<center>
<div class="mxgraph" style="max-width:100%;border:1px solid transparent;" data-mxgraph="{&quot;highlight&quot;:&quot;#006633&quot;,&quot;lightbox&quot;:false,&quot;nav&quot;:false,&quot;resize&quot;:true,&quot;xml&quot;:&quot;&lt;mxfile host=\&quot;www.draw.io\&quot; modified=\&quot;2020-01-27T20:09:03.888Z\&quot; agent=\&quot;Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36\&quot; etag=\&quot;8otv9sNdF-oivO5T6t3e\&quot; version=\&quot;12.5.8\&quot; type=\&quot;device\&quot;&gt;&lt;diagram id=\&quot;C5RBs43oDa-KdzZeNtuy\&quot; name=\&quot;Page-1\&quot;&gt;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&lt;/diagram&gt;&lt;/mxfile&gt;&quot;}"></div>
<script type="text/javascript" src="https://www.draw.io/js/viewer.min.js"></script>
</center>
### 1. Defining a Predictive Task 
To formally specify our task, we require a set of rules to decide who is included in a group representing the population of interest, patient-level features for each of the members of this group, and an outcome or result per patient. Furthermore, each of these parameters must be specified with respect to a timeframe.
![Diagram of a Predictive Task Specification](https://i.imgur.com/P03wz6X.png)
We define our end-of-life task as follows:

> For each patient who is over the age of 70 at prediction time, and is enrolled in an insurance plan for which we have claims data available for 95% of the days of calendar year 2009, and is alive as of March 31, 2010: predict if the patient will die during the interval of time between April 1, 2010 and September 31, 2010 using the drugs prescribed, procedures performed, and conditions diagnosed during the year 2009.

`omop-learn` splits the conversion of this natural language specification of a task to code into two natural steps. First, we define a **cohort** of patients, each of which has an outcome. Second, we generate **features** for each of these patients. These two steps are kept independent of each other, allowing different cohorts or feature sets to very quickly be tested and evaluated. We explain how cohorts and features are initialized through the example of the end-of-life problem.

#### 1.1 Data Backend Initialization
`omop-learn` supports a collection of data [backend engines](https://github.com/clinicalml/omop-learn/tree/master/src/omop_learn/backends) depending on where the source OMOP tables are stored: PostgreSQL, Google BigQuery, and Apache Spark. The `PostgresBackend`, `BigQueryBackend`, and `SparkBackend` classes inherit from `OMOPDatasetBackend` which defines the set of methods to interface with the data storage as well as the feature creation.

Configuration parameters used to initialize the backend are surfaced through python `.env` files, for example [`bigquery.env`](https://github.com/clinicalml/omop-learn/blob/master/bigquery.env). For this example, the `.env` file stores the name of the project in Google BigQuery, schemas to read data from and write the cohort to, as well as local directories to store feature data and trained models. The backend can then simply be created as:

```python
load_dotenv("bigquery.env")

config = Config({
    "project_name": os.getenv("PROJECT_NAME"),
    "cdm_schema": os.getenv("CDM_SCHEMA"),
    "aux_cdm_schema": os.getenv("AUX_CDM_SCHEMA"),
    "prefix_schema": os.getenv("PREFIX_SCHEMA"),
    "datasets_dir": os.getenv("OMOP_DATASETS_DIR"),
    "models_dir": os.getenv("OMOP_MODELS_DIR")
})

# Set up database
backend = BigQueryBackend(config)
```

#### 1.2 <a name="define_cohort"></a> Cohort Initialization
OMOP's [`PERSON`](https://github.com/OHDSI/CommonDataModel/wiki/PERSON) table is the starting point for cohort creation, and is filtered via SQL query. Note that these SQL queries can be written with variable parameters which can be adjusted for different analyses. These parameters are implemented as [Python templates](https://www.python.org/dev/peps/pep-3101/). In this example, we leave dates as parameters to show how cohort creation can be flexible.

We first want to establish when patients were enrolled in insurance plans which we have access to. We do so using OMOP's `OBSERVATION_PERIOD` table. Our SQL logic finds the number of days within our data collection period (all of 2009, in this case) that a patient was enrolled in a particular plan:
```sql
death_training_elig_counts as (
        select
            o.person_id,
            o.observation_period_start_date as start,
            o.observation_period_end_date as finish,
            greatest(
                date_diff(
                    least(o.observation_period_end_date, date '{training_end_date}'), 
                    greatest(o.observation_period_start_date, date '{training_start_date}'),
                    day
                ), 0
            ) as num_days
        from {cdm_schema}.observation_period o
        inner join eligible_people p
        on o.person_id = p.person_id
    )
```
Note that the dates are left as template strings that can be filled later on. Next, we want to filter for patients who are enrolled for 95% of the days in our data collection period. Note that we must be careful to include patients who used multiple different insurance plans over the course of the year by aggregating the intermediate table `death_training_elig_counts` which is specified above. Thus, we first aggregate and then collect the `person_id` field for patients with sufficient coverage over the data collection period:
```sql
death_trainingwindow_elig_perc as (
        select
            person_id
        from
            death_training_elig_counts
        group by
            person_id
        having
            sum(num_days) >= 0.95 * extract(day from (date '{training_end_date}' - date '{training_start_date}'))
    )
```
The next step is to find outcomes. 
```sql
death_dates as (
        select
            p.person_id,
            a.death_date
        from
            {cdm_schema}.person p
        inner join
            {cdm_schema}.death a
        on
            p.person_id = a.person_id
    )
```
Then, we select for patients over the age of 70 at prediction time:
```sql
eligible_people as (
        select p.person_id
        from {cdm_schema}.person p
        where extract(
            year from date '{training_end_date}'
        ) - p.year_of_birth > 70
    )
```
Finally, we can create the cohort:
```sql
select
        row_number() over (order by te.person_id) - 1 as example_id,
        te.person_id,
        date '{training_start_date}' as start_date,
        date '{training_end_date}' as end_date,
        d.death_date as outcome_date,

        cast(coalesce(
            (d.death_date between
                date '{training_end_date}'
                 + interval {gap}
                and
                date '{training_end_date}'
                 + interval {gap}
                 + interval {outcome_window}
            ), false
        ) as int) as y
    from
        death_testwindow_elig_perc te
        left join death_dates d on d.person_id = te.person_id
    where
        (
            d.death_date is null
            or d.death_date >= (date '{training_end_date}' + interval {gap})
        )
```
The full cohort creation SQL query can be found [here](https://github.com/clinicalml/omop-pkg/blob/main/examples/eol/bigquery_sql/gen_EOL_cohort.sql).

Note the following key fields in the resulting table:

Field | Meaning
------------ | -------------
`example_id` | A unique identifier for each example in the dataset. While in the case of end-of-life each patient will occur as a positive example at most once, this is not the case for all possible prediction tasks, and thus this field offers more flexibility than using the patient ID alone.
`y` | A column indicating the outcome of interest. Currently, `omop-learn` supports binary outcomes.
`person_id` | A column indicating the ID of the patient.
`start_date` and `end_date` | Columns indicating the beginning and end of the time periods to be used for data collection for this patient. This will be used downstream for feature generation. 

We are now ready to build a cohort. We construct a [`Cohort`](https://github.com/clinicalml/omop-learn/blob/master/src/omop_learn/data/cohort.py) object by passing the path to a defining SQL script, the relevant data backend, and the set of cohort params.
```python
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
```

#### <a name="define_features"></a> 1.3 Feature Initialization
With a cohort now fully in place, we are ready to associate features with each patient in the cohort. These features will be used downstream to predict outcomes. 

The OMOP Standardized Clinical Data tables offer several natural features for a patient, including histories of condition occurrence, procedures, and drugs administered. `omop-learn` includes SQL scripts to collect time-series of these common features automatically for any cohort, allowing a user to quickly set up a feature set. We supply paths to the feature SQLs as well as the name of each feature in constructing features using the [`Feature`](https://github.com/clinicalml/omop-learn/blob/master/src/omop_learn/data/feature.py) object: 
```python
sql_dir = "examples/eol/bigquery_sql"

feature_paths = [f"{sql_dir}/drugs.sql"]
feature_names = ["drugs"]
features = [Feature(n, p) for n, p in zip(feature_names, feature_paths)]

ntmp_feature_paths = [f"{sql_dir}/age.sql", f"{sql_dir}/gender.sql"]
ntmp_feature_names = ["age", "gender"]
features.extend([Feature(n, p, temporal=False) for n, p in zip(ntmp_feature_names, ntmp_feature_paths)])
```
By default, the package assumes that added features are temporal in nature, i.e. that observations are collected at a time interval for a patient. `omop-learn` also supports nontemporal features which are assumed to be static in nature for a given time period, such as age and gender. This is specified by setting the flag `temporal=False` in the construction of the `Feature` object.

Finally, we create an `OMOPDataset` object to trigger creation of the features via the backend. Here we pass in initialization arguments which include the `Config` object used to specify backend parameters, the backend itself (e.g. `BigQueryBackend`), the previously created `Cohort` object, and the list of features storing `Feature` objects:
```python
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
```
Note that feature extraction outputs the set of features to local disk in the directory specified by `data_dir` into the initialization arguments (if left blank, this defaults to the directory supplied in the `.env` file). This directory outputs features in a `data.json` file, which defaults to storing a patient's features in a single json line. Here the features are stored as a list of lists in the json key `visits`, in which the outer list stores features for a given date, and the inner list stores the concepts that appeared on that day. The corresponding dates can be extracted from the patient line using the json key `dates`. Note also that `person_id` and static features such as `age` and `gender` are saved down into the json. The argument `is_visit_dataset=True` configures an alternative feature representation in which a single line of `data.json` represents a visit, rather than a patient.

Feature extraction is written with python's `multiprocessing` library for enhanced performance; the `num_workers` argument can be used to configure the number of parallel processes. Additional customized features can also be created by adding additional files to the [feature SQL directory](https://github.com/clinicalml/omop-learn/tree/master/examples/eol/bigquery_sql). The added queries should output rows in the same format as the existing SQL scripts.

### <a name="preprocess"></a> 2. Ingesting Feature Data
Once features are created using the `OMOPDataset` object, `omop-learn` uses sparse tensors in COO format to aggregate features for use in models, with indices accessed via bi-directional hash maps. These are interfaced through the `OMOPDatasetSparse` class.

For temporal features, this tensor can be accessed through the variable `OMOPDatasetSparse.feature_tensor`. This object has three axes corresponding to patients, timestamps, and OMOP concepts respectively.

In our EOL example, we will filter on both time and concept axes. We filter on the concept axis exactly as above, removing OMOP's catch-all "no matching concept" buckets since they don't correspond to any real medical feature. We create features by collecting counts of how many times each OMOP code has been applied to a patient over the last `d` days for several values of `d`, and then concatenating these variables together into a feature vector. Thus for each backwards-looking window `d` we must create a seperate time filter. This filtering is executed using the `OMOPDatasetWindowed` class by calling `to_windowed()`. 
```python
# Re-load a pre-built dataset
dataset = OMOPDataset.from_prebuilt(config.datasets_dir)

# Window the omop dataset and split it
window_days = [30, 180, 365, 730, 1500, 5000, 10000]
windowed_dataset = dataset.to_windowed(window_days)
windowed_dataset.split()
```
The `to_windowed()` function takes in the raw sparse tensor of features, filters several times to collect data from the past `d` days for each `d` in `window_lengths`, then sums along the time axis to find the total count of the number of times each code was assigned to a patient over the last `d` days. These count matrices are then concatenated to build a final feature set of windowed count features. Note that unlike a pure SQL implementation of this kind of feature, `omop-learn` can quickly rerun the analysis for a different set of windows; this ability to tune parameters allows use of a validation set to determine optimal values and thus significantly increase model performance. Note that we can also easily split the windowed data into train, validation, and test sets by calling the method `split()` on the windowed dataset in evaluating model performance.


## Files

We review the subdirectories in the source package for [`omop-learn`](https://github.com/clinicalml/omop-learn/tree/master/src/omop_learn).

### backends

The set of backends interfaces with the data storage and the compute engine to run feature extraction. We support PostgreSQL, Google BigQuery, and Apache Spark. The set of defining methods are inherited from `OMOPDatasetBackend`. Note that backend feature creation leverages python's `multiprocessing` library to extract features, parallelized by OMOP `person_id`.

### data

Data methods include the `Cohort`, `Feature`, and `ConceptTokenizer` classes. Cohort and features can be initialized using the previously reviewed code snippets. 

The `ConceptTokenizer` class offers a compact representation for storing the set of relevant OMOP concepts by providing a mapping from concept index to name. This class also includes a set of special tokens, including beginning of sequence, end of sequence, separator, pad, and unknown, for use with language modeling applications.


### hf

Utilities for interfacing with [Hugging Face libraries](https://huggingface.co/) are provided. This includes a mapping from the `OMOPDataset` object to dataset objects ingestible by Hugging Face models.


### models

The files `transformer.py` and `visit_transformer.py` provide modeling methods used to create the SARD architecture [Kodialam et al. 2021]. The methods in `transformer.py` define transformer blocks and multi-head attention in the standard way. The methods in `visit_transformer.py` define a transformer-based architecture over visits consisting of OMOP concepts.


### sparse

The classes in `sparse` allow for end-to-end modeling over the created feature representation using sparse tensors in COO format. `data.py` defines the previously reviewed `OMOPDatasetSparse` and `OMOPDatasetWindowed` classes which aggregate features over multiple time windows. `models.py` defines a wrapper over the `sklearn` `LogisticRegression` object, which integrates tightly with the `OMOPDatasetWindowed` class to define an end-to-end modeling pipeline.

### torch

The classes in `data.py` define a wrapper around the `OMOPDataset` object for use with pytorch tensors. Similar to the classes in `hf`, this allows for quick modeling with `torch` code. `models.py` gives some example models that can ingest `OMOPDatasetTorch` objects, including an alternate implementation for the `VisitTransformer`.

### utils

A variety of `utils` are provided which support both data ingestion and modeling. `config.py` defines a simple configuration object for use in constructing the backend, while methods in `date_utils.py` are used for conversion between unix timestamps and datetime objects.

`embedding_utils.py` defines a gensim word embedding model used in the end-of-life example notebook.
