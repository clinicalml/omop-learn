select
    b.example_id,
    b.person_id,
    case
        when b.year_of_birth is not null then b.year_of_birth
        else 1
    end as ntmp_val,
    case
        when b.year_of_birth is not null then 'Age at end_date'
        else 'Missing year of birth'
    end as concept_name,
    b.start_date as person_start_date,
    b.end_date as person_end_date
from
    {cohort_table} b
