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
            p.death_date
        from
            {cdm_schema}.death p
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
                least (
                    o.observation_period_end_date,
                    date '{training_end_date}'
                ) - greatest(
                    o.observation_period_start_date,
                    date '{training_start_date}'
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
            sum(num_days) >= 0.95 * (date '{training_end_date}' - date '{training_start_date}')
    ),
    death_testperiod_elig_counts as (
        select
            p.person_id,
            p.observation_period_start_date as start,
            p.observation_period_end_date as finish,
            greatest(
                    least (
                        p.observation_period_end_date,
                        date (
                            date '{training_end_date}'
                            + interval '{gap}'
                            + interval '{outcome_window}'
                        )
                    ) - greatest(
                        p.observation_period_start_date,
                        date '{training_end_date}'
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
            (d.death_date >= date '{training_end_date}' + interval '{gap}' and
             d.death_date <= date '{training_end_date}' + interval '{gap}' + interval '{outcome_window}')
        or
            sum(num_days) >= 0.95 * extract(
                epoch from (
                    interval '{gap}'
                    + interval '{outcome_window}' --epoch returns the number of seconds in gap + outcome_window
                )
            )/(24*60*60) --convert seconds to days
    )

    select
        row_number() over (order by te.person_id) - 1 as example_id,
        te.person_id,
        date '{training_start_date}' as start_date,
        date '{training_end_date}' as end_date,
        d.death_date as outcome_date,

        coalesce(
            (d.death_date between
                date '{training_end_date}'
                 + interval '{gap}'
                and
                date '{training_end_date}'
                 + interval '{gap}'
                 + interval '{outcome_window}'
            ), false
        )::int as y
    from
        death_testwindow_elig_perc te
        left join death_dates d on d.person_id = te.person_id
    where
        (
            d.death_date is null
            or d.death_date >= (date '{training_end_date}' + interval '{gap}')
        )
    ;

