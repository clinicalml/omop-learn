# omop-learn


## Introductory Tutorial

omop-learn allows [OMOP-standard (CDM v6)](https://github.com/OHDSI/CommonDataModel/wiki) medical data like Claims and EHR information to be processed efficiently for predictive tasks. The library allows users to precisely define cohorts of interest, patient-level time series features, and target variables of interest. Relevant data is automatically extracted and surfaced in formats suitable for most machine learning algorithms, and the (often extreme) sparsity of patient-level data is fully taken into account to provide maximum performance. 


The library provides several benefits for modelling, both in terms of ease of use and performance:
* All that needs to be specified are cohort and outcome definitions, which can often be done using simple SQL queries.
* Our fast data ingestion and transformation pipelines allow for easy and efficient tuning of algorithms -- we have seen significant improvements in out-of-sample performance of predictors after hyperparameter tuning that would take days with simple SQL queries but minutes with Prediction Library
* We modularize the data extraction and modelling processes, allowing users to use new models as they become available with very little modification to the code. Tools ranging from simple regressions to deep neural net models can easily be substituted in and out in a plug-and-play manner.

omop-learn serves as a modern python alternative to the [PatientLevelPrediction R library](https://github.com/OHDSI/PatientLevelPrediction). We allow seamless integration of many Python based machine learning and data science libraries by supporting generic sklearn-stye classifiers. Our new data storage paradigm also allows for more on-the-fly feature engineering as compared to previous libraries. 

In this tutorial, we walk through the process of using omop-learn for an end-of-life prediction task for Medicare patients with clear applications to improving palliative care. The code used can also be found in the [example notebook](https://github.com/clinicalml/omop-learn/blob/master/PL2%20Test%20Driver.ipynb), and can be run on your own data as you explore omop-learn. The control flow diagram below also links to relevant sections of the library documentation.
<center>
<div class="mxgraph" style="max-width:100%;border:1px solid transparent;" data-mxgraph="{&quot;highlight&quot;:&quot;#006633&quot;,&quot;lightbox&quot;:false,&quot;nav&quot;:false,&quot;resize&quot;:true,&quot;xml&quot;:&quot;&lt;mxfile host=\&quot;www.draw.io\&quot; modified=\&quot;2020-01-27T20:09:03.888Z\&quot; agent=\&quot;Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36\&quot; etag=\&quot;8otv9sNdF-oivO5T6t3e\&quot; version=\&quot;12.5.8\&quot; type=\&quot;device\&quot;&gt;&lt;diagram id=\&quot;C5RBs43oDa-KdzZeNtuy\&quot; name=\&quot;Page-1\&quot;&gt;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&lt;/diagram&gt;&lt;/mxfile&gt;&quot;}"></div>
<script type="text/javascript" src="https://www.draw.io/js/viewer.min.js"></script>
</center>
### 1. Defining a Predictive Task 
To formally specify our task, we require a set of rules to decide who is included in a group representing the population of interest, patient-level features for each of the members of this group, and an outcome or result per patient. Furthermore, each of these parameters must be specified with respect to a timeframe.
![Diagram of a Predictive Task Specification](https://i.imgur.com/P03wz6X.png)
We define our end-of-life task as follows:

> For each patient who is over the age of 70 at prediction time, and is enrolled in an insurance plan for which we have claims data available for 95% of the days of calendar year 2016, and is alive as of March 31, 2017: predict if the patient will die during the interval of time between April 1, 2017 and September 31, 2017 using data including the drugs prescribed, procedures performed, conditions diagnosed and the medical specialties of the clinicians who cared for the patient during 2016.

omop-learn splits the conversion of this natural language specification of a task to code into two natural steps. First, we define a **cohort** of patients, each of which has an outcome. Second, we generate **features** for each of these patients -- these two steps are kept independent of each other in omop-learn, allowing different cohorts or feature sets to very quickly be tested and evaluated. We explain how cohorts and features are initialized through the example of the end-of-life problem.
#### 1.1 <a name="define_cohort"></a> Cohort Initialization
OMOP's [`PERSON`](https://github.com/OHDSI/CommonDataModel/wiki/PERSON) table is the starting point for cohort creation, and is filtered via SQL query. Note that these SQL queries can be written with variable parameters which can be adjusted for different analyses. These parameters are implemented as [Python templates](https://www.python.org/dev/peps/pep-3101/). In this example, we leave dates as parameters to show how cohort creation can be flexible.

We first want to establish when patients were enrolled in insurance plans which we have access to. We do so using OMOP's `OBSERVATION_PERIOD` table. Our SQL logic finds the number of days within our data collection period (all of 2016, in this case) that a patient was enrolled in a particular plan:
```sql
death_training_elig_counts as (
        select
            person_id,
            observation_period_start_date as start,
            observation_period_end_date as finish,
            greatest(
                least (
                    observation_period_end_date,
                    date '{ training_end_date }'
                ) - greatest(
                    observation_period_start_date,
                    date '{ training_start_date }'
                ), 0
            ) as num_days
        from cdm.observation_period
    )
```
Note that the dates are left as template strings that can be filled later on. Next, we want to filter for patients who are enrolled for 95% of the days in our data collection period - note that we must be careful to include patients who used multiple different insurance plans over the course of the year by aggregating the intermediate table `death_training_elig_counts` which is specified above. Thus, we first aggregate and then collect the `person_id` field for patients with sufficient coverage over the data collection period:
```sql
    death_trainingwindow_elig_perc as (
        select
            person_id
        from
            death_training_elig_counts
        group by
            person_id
        having
            sum(num_days) >= 0.95 * (date '{ training_end_date }' - date '{ training_start_date }')
    )
```
The next step is to find outcomes. 
```sql
death_dates as (
        select
            p.person_id,
            p.death_datetime
        from
            cdm.person p
    )
```
Then, we select for patients over the age of 70 at prediction time:
```sql
eligible_people as (
        select p.person_id
        from cdm.person p
        where extract(
            year from date '{training_end_date}'
        ) - p.year_of_birth > 70
    ),
```
Finally, we can create the cohort:
```sql
    select
        row_number() over (order by p.person_id) - 1 as example_id,
        p.person_id,
        date '{ training_start_date }' as start_date,
        date '{ training_end_date }' as end_date,
        d.death_datetime as outcome_date,
        coalesce(
            (d.death_datetime between
                date '{ training_end_date }'
                 + interval '{ gap }'
                and
                date '{ training_end_date }'
                 + interval '{ gap }'
                 + interval '{ outcome_window }'
            ), false
        )::int as y
    from
        eligible_people p
        inner join death_testwindow_elig_perc te on te.person_id = p.person_id
        left join death_dates d on d.person_id = p.person_id
    where
        (
            d.death_datetime is null
            or d.death_datetime >= (date '{ training_end_date }' + interval '{ gap }')
        )
```
The full cohort creation SQL query can be found [here](https://github.com/clinicalml/omop-learn/blob/master/sql/Cohorts/gen_EOL_cohort.sql).

Note the following key fields in the resulting table:

Field | Meaning
------------ | -------------
`example_id` | A unique identifier for each example in the dataset. While in the case of end-of-life each patient will occur as a positive example at most once, this is not the case for all possible prediction tasks, and thus this field offers more flexibility than using the patient ID alone.
`y` | A column indicating the outcome of interest. Currently, omop-learn supports binary 0/1 outcomes
`person_id` | A column indicating the ID of the patient
`start_date` and `end_date` | Columns indicating the beginning and end of the time periods to be used for data collection for this patient. This will be used downstream for feature generation. 

We are now ready to build a cohort. We use the `CohortGenerator` class to pass in a cohort name, a path to a SQL script, and relevant parameters in Python:
```sql
cohort_name = '__eol_cohort'
cohort_script_path = config.SQL_PATH_COHORTS + '/gen_EOL_cohort.sql'
params = {'schema_name'           : schema_name,
          'aux_data_schema'       : config.CDM_AUX_SCHEMA,
          'training_start_date'   : '2016-01-01',
          'training_end_date'     : '2017-01-01',
          'gap'                   : '3 months',
          'outcome_window'        : '6 months'
         }

cohort = CohortGenerator.Cohort(
    schema_name=schema_name,
    cohort_table_name=cohort_name,
    cohort_generation_script=cohort_script_path,
    cohort_generation_kwargs=params
)
```
Note that this does *not* run the SQL queries -- the `CohortGenerator` object currently just stores *how* to set up the cohort in any system, allowing for more portability. Thus, our next step is to materialize the actual cohort to a table in a specific database `db` by calling `cohort.build(db)`.
#### <a name="define_features"></a> 1.2 Feature Initialization
With a cohort now fully in place, we are ready to associate features with each patient in the cohort. These features will be used downstream to predict outcomes. 

The OMOP Standardized Clinical Data tables offer several natural features for a patient, including histories of condition occurrence, procedures, etc. omop-learn includes SQL scripts to collect time-series of these common features automatically for any cohort, allowing a user to very quickly set up a feature set. To do so, we first initialize a `FeatureGenerator` object with a database indicating where feature data is to be found. Similar to the `CohortGenerator`, this does not actually create a feature set -- that is only done once all parameters are specified. We next select the pre-defined features of choice, and finally select a cohort for which data is to be collected:
```sql
featureSet = FeatureGenerator.FeatureSet(db)
featureSet.add_default_features(
    ['drugs','conditions','procedures'],
    schema_name,
    cohort_name
)
```
By default, the package assumes that added features are temporal in nature. omop-learn also supports nontemporal features, such as age and gender.
```sql
featureSet.add_default_features(
    ['age','gender'],
    schema_name,
    cohort_name,
    temporal=False
)
```
Finally, we call the build() function, which executes the relevant feature queries to create the feature set.
```sql
featureSet.build(cohort, cache_file='eol_feature_matrix', from_cached=False)
```
Since collecting the data is often the most time-consuming part of the setup process, we cache intermediate results in the `cache_name` file (if `from_cached` is False) and can later use this data instead of executing the relevant queries (if `from_cached` is True).

Additional customized features can also be created by advanced users, by adding files to the [feature SQL directory](https://github.com/clinicalml/omop-learn/tree/master/sql/Features). The added queries should output rows in the same format as the existing SQL scripts.

### <a name="preprocess"></a> 2. Ingesting Feature Data
Once we have called `build` on a `FeatureSet` object, omop-learn will begin collecting all the relevant data from the OMOP database. To efficiently store and manipulate this information, we use sparse tensors in COO format, with indices accessed via bi-directional hash maps. 

For the temporal features, this tensor can be accessed by calling `featureSet.get_sparr_rep()`. This object has three axes corresponding to patients, timestamps, and OMOP concepts respectively. The axes can be manipulated as outlined in the table below:

Axis | Index Maps | Description| Example Utilization
-----|-----|-------|-----------------
Patient | `featureSet.id_map` and `featureSet.id_map_rev` | Each index corresponds to a patient in the cohort. The data for the patient with id `a` will be at index `featureSet.id_map_rev[a]` and likewise index `b` corresponds to the patient with id `fearureSet.id_map[b]`.| To get data for patients whose patient ID's are in the list `filtered_ids`, we would find the relevant indices by running `filtered_indices = [featureSet.id_map_rev[id] for id in filtered_ids]`, then indexing into Patient axis of the sparse tensor with `filtered_tensor = featureSet.get_sparr_rep()[filtered_indices, :, :]`.
Time | `featureSet.time_map` and `featureSet.concept_map_rev` | Each index corresponds to a unique timestamp. At present, the data we use comes in at daily ticks, so each index corresponds to a day on which an OMOP code was assigned to a patient. The index corresponding to timestamp `t` is `featureSet.time_map_rev[t]` and the timestamp corresponding to index `i` is `featureSet.time_map[i]`. | To get data from April 2016 onwards only, we can filter the indices of the time axis by running `time_indices_filtered = [i for i in featureSet.time_map if featureSet.time_map[i] > pd.to_datetime('2017-4-1')]`, then index into the sparse tensor along the time axis : `filtered_tensor = featureSet.get_sparr_rep()[:, time_indices_filtered, :]`.
OMOP Concept | `featureSet.concept_map` and `featureSet.concept_map_rev` | Each index corresponds to a unique OMOP concept. The index corresponding to concept `c` is `featureSet.concept_map_rev[c]` and the timestamp corresponding to index `i` is `featureSet.concept_map[i]`. | To get data for all codes where an OMOP concept is matched, we would want to exclude codes that map to "no matching concept". We can get the indices corresponding to the non-excluded codes with `feature_indices_filtered = [i for i in featureSet.concept_map if '- No matching concept' not in featureSet.concept_map[i]]`, then indexing in with `filtered_tensor = featureSet.get_sparr_rep()[:, :, feature_indices_filtered]`.

In our EOL example, we will filter on both time and concept axes. We filter on the concept axis exactly as above, removing OMOP's catch-all "no matching concept" buckets since they don't correspond to any real medical feature. We create features by collecting counts of how many times each OMOP code has been applied to a patient over the last `T` days for several values of `T`, and then concatenating these varibales together into a feature vector. Thus for each backwards-looking window `T` we must create a seperate time filter -- this advanced filtering is already pre-coded into omop-learn and can be called as follows:
```sql
feature_matrix_counts, feature_names = data_utils.window_data(
    window_lengths = [30, 180, 365, 730],
    feature_matrix = feature_matrix_3d,
    all_feature_names = good_feature_names,
    cohort = cohort,
    featureSet = featureSet
)
```
This function takes in the raw sparse tensor of features, filters several times to collect data from the past `d` days for each `d` in `window_lengths`, then sums along the time axis to find the total count of the number of times each code was assigned to a patient over the last `d` days. These count matrices are then concatenated to each other to build a final feature set of windowed count features. Note that unlike a pure SQL implementation of this kind of feature, omop-learn can quickly rerun the analysis for a different set of windows -- this ability to tune the parameters allows us to use a validation set to determine optimal values and thus significantly increase model performance.  

This feature matrix can then be used with any sklearn modelling pipeline -- see the example notebook [End of Life Linear Model Example](https://github.com/clinicalml/omop-learn/blob/master/End%20of%20Life%20Linear%20Model%20Example.ipynb) for an example pipeline involving some pre-processing followed by a heavily regularized logistic regression. 

For the nontemporal features, the 2d sparse feature matrix can be access by calling `featureSet.get_nontemporal_sparr_rep()`. This object has two axes corresponding to patients and OMOP concepts respectively. For an example involving nontemporal features, see the [End of Life Linear Model Example (With Nontemporal Features)](https://github.com/clinicalml/omop-learn/blob/master/End%20of%20Life%20Linear%20Model%20Example%20(With%20Nontemporal%20Features).ipynb).

## Code Documentation

Code documentation can be accessed [here](/omop-learn/sphinx/)

## Files
### config.py
This file contains global constants and the parameters needed to connect to a postgres database in which OMOP data is stored. The password field has been reset and must be entered to run the code


### Utils

#### dbutils.py

dbutils.py provides tools for interacting with a postgres database into which a set of OMOP compliant tables have been loaded. The Database object can be instantiated using a standard postgres connection string, and can then be used (via 'query', 'execute' and 'fast_query') to run arbitrary SQL code and return results in Pandas dataframes.

#### PopulateAux.py

PopulateAux.py allows for the definition of custom tables that do not exist in the OMOP framework, but are required over multiple models by the user. These can be instantiated and kept in an auxiliary schema, and used persistently as needed.


### Generators

This directory contains the implementation of classes to store and instantiate Cohorts of patients and sets of Features that can be used for prediction tasks.

#### CohortGenerator.py

Cohorts are defined by giving the schema in which the cohort table will be materialized, a unique cohort name, and a SQL script that uses OMOP standard tables (and/or user defined auxiliary tables) to generate the cohort itself.


An example script can be found in /sql/Cohorts. As in that script, cohort definitions should give at minimum a unique example ID, a person ID corresponding to the patient's unique identifier in the rest of the OMOP database, and an outcome column (here denoted by 'y') indicating the outcome of interest for this particular patient. 

#### FeatureGenerator.py

The FeatureGenerator file defines two objects: Features and FeatureSets. Features are defined by a SQL script and a set of keyword arguments that can be used to modify the SQL script just before it is run through Python's 'format' functionality. Several SQL scripts are already pre-implemented and can be seen in /sql/Features. At present, PredictionLibrary supports time-series of binary features. Thus, feature SQL scripts should generate tables with at least the following columns:
- A person ID to join with the cohort and identify which patient this feature is associated with
- A feature name, which often will be generated by joining with OMOP's concept table to get a human-readable description of a OMOP concept
- A timestamp value

FeatureSet objects simply collect a list of Feature objects. When the 'build' function is called, the FeatureSet will run all SQL associated with each Feature and insert the resulting rows into a highly data-efficient three-dimensional sparse tensor representation, with the three axes of this tensor representing distinct patients, distinct features, and distinct timestamps respectively. The tensor can then be accessed directly and manipulated as needed for any chosen modelling approach. 


### End of Life Linear Model Example.ipynb and End of Life Deep Model Example.ipynb

These notebooks walk through all the functionality of the library through the example of building a relatively simple yet performant end-of-life prediction model from OMOP data loaded from IBC. Use these files as a tutorial and as a way to see the correct way to call the functions in the library.
