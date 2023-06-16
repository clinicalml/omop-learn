select
    a.drug_concept_id || ' - drug - ' || c.concept_name as concept_name,
    a.drug_exposure_start_date as feature_start_date
from
    {cdm_schema}.drug_exposure a
left join
    {cdm_schema}.concept c
on
    c.concept_id = a.drug_concept_id
where
    a.person_id = {person_id}