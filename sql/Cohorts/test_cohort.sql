/*
    Construct the cohort table used in an example End-of-Life prediction Model

    Inclusion criteria:
    - Enrolled in 95% of months of training
    - Enrolled in 95% of days during outcome window, or expired during outcome window
    - Patient over the age of 70 at prediction time
*/

create table {schema_name}.{cohort_table_name} as

with
    death_dates as (
        select
            p.person_id,
            p.death_datetime,
            p.year_of_birth
        from
            cdm.person p
    ),
    
    -- First process positive examples...
    positive_people as (
        select
            person_id,
            case  -- if uniform, sample from [gap, gap+outcome_window], otherwise constant interval
                when {positive_pred_unif}
                    then death_datetime - (random() * interval '{outcome_window}' 
                                            + interval '{gap}') 
                else death_datetime - interval '{positive_pred_delta}'  
            end as end_date,
            death_datetime as outcome_date,
            year_of_birth,
            1 as y
        from death_dates 
        where death_datetime is not null
    ),
    
    -- ...checking for age...
    positive_people_age as (
        select
            person_id,
            end_date,
            outcome_date,
            y
        from positive_people
        where extract(
            year from end_date
        ) - year_of_birth > 70
    ),
    
    -- ...then check eligibility of positive samples...
    positive_training_elig_counts as (
        select
            o.person_id,
            o.observation_period_start_date as start,
            o.observation_period_end_date as finish,
            greatest(
                least (
                    o.observation_period_end_date,
                    date (p.end_date)
                ) - greatest(
                    o.observation_period_start_date,
                    date (p.end_date - interval '{eligibility_period}')
                ), 0
            ) as num_days
        from cdm.observation_period o
        inner join positive_people_age p
        on o.person_id = p.person_id
    ),
    positive_training_elig_perc as (
        select
            person_id
        from
            positive_training_elig_counts
        group by
            person_id
        having
            sum(num_days) >= 0.95 * extract(
                epoch from (
                    interval '{eligibility_period}' --epoch returns the number of seconds in eligibility_period
                )
            )/(24*60*60) --convert seconds to days
    ),
    
    -- ...to get final cohort of positive samples.
    eligible_positive_people as (
        select
            p.person_id,
            p.end_date,
            p.outcome_date,
            p.y
        from positive_people p
        join positive_training_elig_perc pt
        on p.person_id = pt.person_id
    ),
    
    -- Then process negative examples...
    negative_people as (
        select
            person_id,
            date '{negative_pred_date}' as end_date,
            null::timestamp as outcome_date,
            year_of_birth,
            0 as y
        from death_dates
        where death_datetime is null
    ),
    
    -- ...checking for age...
    negative_people_age as (
        select
            person_id,
            end_date,
            outcome_date,
            y
        from negative_people
        where extract(
            year from end_date
        ) - year_of_birth > 70
    ),
    
    -- ...and check eligibility before prediction time...
    negative_training_elig_counts as (
        select
            o.person_id,
            o.observation_period_start_date as start,
            o.observation_period_end_date as finish,
            greatest(
                least (
                    o.observation_period_end_date,
                    date (p.end_date)
                ) - greatest(
                    o.observation_period_start_date,
                    date (p.end_date - interval '{eligibility_period}')
                ), 0
            ) as num_days,
            p.end_date
        from cdm.observation_period o
        inner join negative_people_age p
        on o.person_id = p.person_id
    ),
    negative_training_elig_perc as (
        select
            person_id,
            end_date
        from
            negative_training_elig_counts
        group by
            person_id, end_date
        having
            sum(num_days) >= 0.95 * extract(
                epoch from (
                    interval '{eligibility_period}' --epoch returns the number of seconds in eligibility_period
                )
            )/(24*60*60) --convert seconds to days
    ),
    
    -- ...as well as eligibility in prediction window...
    negative_test_elig_counts as (
        select
            p.person_id,
            p.observation_period_start_date as start,
            p.observation_period_end_date as finish,
            greatest(
                    least (
                        p.observation_period_end_date,
                        date (
                            tr.end_date
                            + interval '{gap}'
                            + interval '{outcome_window}'
                        )
                    ) - greatest(
                        p.observation_period_start_date,
                        date(tr.end_date)
                    ), 0
            ) as num_days
        from cdm.observation_period p
        inner join 
            negative_training_elig_perc tr
        on 
            tr.person_id = p.person_id
    ), 
    negative_test_elig_perc as (
        select
            person_id
        from
            negative_test_elig_counts
        group by
            person_id
        having
            sum(num_days) >= 0.95 * extract(
                epoch from (
                    interval '{gap}' 
                    + interval '{outcome_window}' --epoch returns the number of seconds in gap + outcome_window
                )
            )/(24*60*60) --convert seconds to days
    ),
    
    -- ...to get final cohort of negative samples.
    eligible_negative_people as (
        select
            p.person_id,
            p.end_date,
            p.outcome_date,
            p.y
        from negative_people p
        join negative_test_elig_perc nt
        on p.person_id = nt.person_id
    )
    
    select
        row_number() over (order by person_id) - 1 as example_id,
        null::timestamp as start_date,         -- required by omop-learn
        *
    from (
        select * from eligible_positive_people
        union all
        select * from eligible_negative_people
    ) tmp;

