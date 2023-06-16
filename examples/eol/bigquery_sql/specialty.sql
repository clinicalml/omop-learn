select 
    p.specialty_concept_id || ' - specialty - ' || c.concept_name as concept_name,
    a.visit_start_date as feature_start_date
from 
    {cdm_schema}.visit_occurrence a
inner join
    {cdm_schema}.provider p
on
     a.provider_id = p.provider_id
left join
    {cdm_schema}.concept c
on 
    c.concept_id = p.specialty_concept_id