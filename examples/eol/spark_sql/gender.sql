select
    b.example_id,
    b.person_id,
    case
        when b.gender_concept_id = 8507 then 1
        when b.gender_concept_id = 8532 then 0
        else 1
    end as ntmp_val,
    case
        when b.gender_concept_id in (8507, 8532) then 'Gender M(1)/F(0)'
        else 'Gender not recorded'
    end as concept_name,
    b.start_date as person_start_date,
    b.end_date as person_end_date
from
    {cohort_table} b
