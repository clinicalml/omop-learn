select
    case
        when a.gender_concept_id = 8507 then 1
        when a.gender_concept_id = 8532 then 0
        else 1
    end as ntmp_val,
    case
        when a.gender_concept_id in (8507, 8532) then 'Gender M(1)/F(0)'
        else 'Gender not recorded'
    end as concept_name
from
    {cdm_schema}.person a
where
    a.person_id = {person_id}
