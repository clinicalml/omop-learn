select 
    case
        when a.year_of_birth is not null then extract(year from date '{end_date}') - a.year_of_birth
        else 1
    end as ntmp_val,
    case 
        when a.year_of_birth is not null then 'Age at end_date'
        else 'Missing year of birth' 
    end as concept_name
from
    {cdm_schema}.person a
where
    a.person_id = {person_id}
