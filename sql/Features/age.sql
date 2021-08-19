select 
    b.example_id,
    a.person_id,
    case
        when a.year_of_birth is not null then date_part('year', b.end_date) - a.year_of_birth
        else 1
    end as ntmp_val,
    case 
        when a.year_of_birth is not null then 'Age at end_date'
        else 'Missing year of birth' 
    end as concept_name,
    b.start_date as person_start_date,
    b.end_date as person_end_date
from
    {cohort_table} b
left join
    {cdm_schema}.person a
on 
    a.person_id = b.person_id
