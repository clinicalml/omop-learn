select 
    b.example_id,
    a.person_id,
    1 as ntmp_val,
    a.race_concept_id || ' - race - ' || coalesce (
        c.concept_name, 'no match'
    ) as concept_name,
    b.start_date as person_start_date,
    b.end_date as person_end_date
from 
    {cdm_schema}.person a
inner join
    {cohort_table} b
on 
    a.person_id = b.person_id
left join
    {cdm_schema}.concept c
on 
    c.concept_id = a.race_concept_id
