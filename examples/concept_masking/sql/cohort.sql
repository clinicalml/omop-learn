/*
    Construct the cohort table used for the visit embedder pretraining.

    Inclusion criteria:
    - Enrolled in at least 5% of months of training
    - Patient over the age of 40 at the end of training period
*/

create table {schema_name}.{cohort_table_name} as

with
    death_dates as (
        select
            p.person_id,
            p.death_datetime
        from
            cdm.person p
    ),
    eligible_people as (
        select p.person_id
        from cdm.person p
        where extract(
            year from date '{training_end_date}'
        ) - p.year_of_birth > 40
    ),
    training_elig_counts as (
        select
            o.person_id,
            o.observation_period_start_date as start,
            o.observation_period_end_date as finish,
            greatest(
                least (
                    o.observation_period_end_date,
                    date '{training_end_date}'
                ) - greatest(
                    o.observation_period_start_date,
                    date '{training_start_date}'
                ), 0
            ) as num_days
        from cdm.observation_period o
        inner join eligible_people p
        on o.person_id = p.person_id
    ),
    trainingwindow_elig_perc as (
        select
            person_id
        from
            training_elig_counts
        group by
            person_id
        having
            sum(num_days) >= 0.05 * (date '{training_end_date}' - date '{training_start_date}')
    )

    select
        row_number() over (order by te.person_id) - 1 as example_id,
        te.person_id,
        date '{training_start_date}' as start_date,
        date '{training_end_date}' as end_date,
        0 as y /* dummy value for y */
    from
        trainingwindow_elig_perc te
    LIMIT 100;
