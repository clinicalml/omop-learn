select 
    b.example_id,
    a.person_id,
    a.drug_concept_id || ' - drug - ' || c.concept_name as concept_name,
    a.drug_exposure_start_datetime as feature_start_date,
    b.start_date as person_start_date,
    b.end_date as person_end_date
from 
    parquet.`datasets/parquet/drug_exposure.parquet` a
inner join
    cohort b
on 
    a.person_id = b.person_id
inner join
    parquet.`datasets/parquet/concept.parquet` c
on 
    c.concept_id = a.drug_concept_id
