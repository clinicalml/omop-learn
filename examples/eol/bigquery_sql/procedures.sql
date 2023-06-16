select 
    a.procedure_concept_id || ' - procedure - ' || c.concept_name as concept_name,
    a.procedure_dat as feature_start_date
from 
    {cdm_schema}.procedure_occurrence a
left join
    {cdm_schema}.concept c
on 
    c.concept_id = a.procedure_concept_id
where
    a.person_id = {person_id}