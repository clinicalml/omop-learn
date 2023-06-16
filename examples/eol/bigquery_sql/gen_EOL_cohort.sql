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
            a.death_date
        from
            {cdm_schema}.person p
        inner join
            {cdm_schema}.death a
        on
            p.person_id = a.person_id
    ),
    eligible_people as (
        select p.person_id
        from {cdm_schema}.person p
        where extract(
            year from date '{training_end_date}'
        ) - p.year_of_birth > 70
    ),
    death_training_elig_counts as (
        select
            o.person_id,
            o.observation_period_start_date as start,
            o.observation_period_end_date as finish,
            greatest(
                date_diff(
                    least(o.observation_period_end_date, date '{training_end_date}'), 
                    greatest(o.observation_period_start_date, date '{training_start_date}'),
                    day
                ), 0
            ) as num_days
        from {cdm_schema}.observation_period o
        inner join eligible_people p
        on o.person_id = p.person_id
    ),
    death_trainingwindow_elig_perc as (
        select
            person_id
        from
            death_training_elig_counts
        group by
            person_id
        having
            sum(num_days) >= 0.95 * extract(day from (date '{training_end_date}' - date '{training_start_date}'))
    ),
    death_testperiod_elig_counts as (
        select
            p.person_id,
            p.observation_period_start_date as start,
            p.observation_period_end_date as finish,
            greatest(
                date_diff(
                    least(
                        p.observation_period_end_date, 
                        date '{training_end_date}' + interval {gap} + interval {outcome_window}
                    ), 
                    greatest(p.observation_period_start_date, date '{training_end_date}'),
                    day 
                ), 0
            ) as num_days
        from {cdm_schema}.observation_period p
        inner join
            death_trainingwindow_elig_perc tr
        on
            tr.person_id = p.person_id
    ),
    death_testwindow_elig_perc as (
        select
            dtec.person_id
        from
            death_testperiod_elig_counts dtec
        join
            death_dates d
        on
            dtec.person_id = d.person_id
        group by
            dtec.person_id, d.death_date
        having
            (d.death_date >= date '{training_end_date}' + interval {gap} and
             d.death_date <= date '{training_end_date}' + interval {gap} + interval {outcome_window})
        or
            sum(num_days) >= 0.95 * date_diff(
                date '1900-01-01' + interval {gap} + interval {outcome_window},
                date '1900-01-01', day 
            ) -- Use dummy date 1900-01-01 in diff to get num_days in {gap} + {outcome_window}
    )

    select
        row_number() over (order by te.person_id) - 1 as example_id,
        te.person_id,
        date '{training_start_date}' as start_date,
        date '{training_end_date}' as end_date,
        d.death_date as outcome_date,

        cast(coalesce(
            (d.death_date between
                date '{training_end_date}'
                 + interval {gap}
                and
                date '{training_end_date}'
                 + interval {gap}
                 + interval {outcome_window}
            ), false
        ) as int) as y
    from
        death_testwindow_elig_perc te
        left join death_dates d on d.person_id = te.person_id
    where
        (
            d.death_date is null
            or d.death_date >= (date '{training_end_date}' + interval {gap})
        )
    ;