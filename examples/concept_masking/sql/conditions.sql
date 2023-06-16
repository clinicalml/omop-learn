select 
    b.example_id,
    a.person_id,
    a.condition_concept_id || ' - condition - ' || coalesce (
        c.concept_name, 'no match'
    ) as concept_name,
    a.condition_start_datetime as feature_start_date,
    b.start_date as person_start_date,
    b.end_date as person_end_date
from 
    {cdm_schema}.condition_occurrence a
inner join
    {cohort_table} b
on 
    a.person_id = b.person_id
left join
    {cdm_schema}.concept c
on 
    c.concept_id = a.condition_concept_id
