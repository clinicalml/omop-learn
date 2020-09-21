select 
    b.example_id,
    a.person_id,
    p.specialty_concept_id || ' - specialty - ' || coalesce (
        c.concept_name, 'no match'
    ) as concept_name,
    date '1900-01-01' - b.end_date + date (a.visit_start_date) as feature_start_date,       -- dummy date to right-align features 
    b.start_date as person_start_date,
    b.end_date as person_end_date
from 
    {cdm_schema}.visit_occurrence a
inner join
    {cohort_table} b
on 
    a.person_id = b.person_id
inner join
    {cdm_schema}.provider p
on
     a.provider_id = p.provider_id
left join
    {cdm_schema}.concept c
on 
    c.concept_id = p.specialty_concept_id
