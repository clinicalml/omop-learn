select 
    a.condition_concept_id || ' - condition - ' || c.concept_name as concept_name,
    a.condition_start_date as feature_start_date
from 
    {cdm_schema}.condition_occurrence a
left join
    {cdm_schema}.concept c
on 
    c.concept_id = a.condition_concept_id
where
    a.person_id = {person_id}