/*
End-of-life cohort query with ALL samples with outcome = 1 (patient expired) and MATCHED distributions for outcome = 0 and 1.
    
Inclusion criteria:
- Enrolled in 95% of months of training
- Enrolled in 95% of days during outcome window, or expired during outcome window
- Patient over the age of 70 at prediction time

Additionally,
- Includes every sample with outcome = 1 (prediction time set to between 'gap' and 'gap'+'outcome_window' before death). 
- For each sample with outcome = 1, includes outcome_ratio number of samples with outcome = 0 at the same prediction date. 
- This has the effect of making the distributions of samples with outcome = 0 and 1 equal.
*/

create table {schema_name}.{cohort_table_name} as

with
    death_dates as (
        select
            p.person_id,
            p.death_datetime,
            p.year_of_birth
        from
            {cdm_schema}.person p
    ),

    -- First process examples with outcome = 1...
    outcome1_people as (
        select
            person_id,
            case  -- if uniform, sample from [gap, gap+outcome_window], otherwise constant interval
                when {outcome1_pred_unif}
                    then (death_datetime - (random() * interval '{outcome_window}' 
                                            + interval '{gap}'))::date
                else death_datetime - interval '{outcome1_pred_delta}'    
            end as end_date,
            death_datetime as outcome_date,
            year_of_birth,
            1 as y
        from death_dates 
        where death_datetime is not null
    ),

    -- ...checking for age...
    outcome1_people_age as (
        select
            person_id,
            end_date,
            outcome_date,
            y
        from outcome1_people
        where extract(
            year from end_date
        ) - year_of_birth > 70
    ),

    -- ...then check eligibility of samples with outcome = 1...
    outcome1_training_elig_counts as (
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
        from {cdm_schema}.observation_period o
        inner join outcome1_people_age p
        on o.person_id = p.person_id
    ),
    outcome1_training_elig_perc as (
        select
            person_id
        from
            outcome1_training_elig_counts
        group by
            person_id
        having
            sum(num_days) >= 0.95 * extract(
                epoch from (
                    interval '{eligibility_period}' --epoch returns the number of seconds in eligibility_period
                )
            )/(24*60*60) --convert seconds to days
    ),

    -- ...to get final cohort of samples with outcome = 1.
    eligible_outcome1_people as (
        select
            p.person_id,
            date(p.end_date) as end_date,
            p.outcome_date,
            p.y
        from outcome1_people p
        join outcome1_training_elig_perc pt
        on p.person_id = pt.person_id
    ),

    -- Then process examples with outcome = 0--
    outcome0_people as(
        select
            person_id,
            null::timestamp as outcome_date,
            year_of_birth,
            0 as y
        from death_dates
        where death_datetime is null
    ),
    outcome0_people_with_dates as (
        select 
            o.person_id,
            o.observation_period_start_date as start,
            o.observation_period_end_date as finish,
            p.outcome_date,
            generate_series(
                date_trunc('month', o.observation_period_start_date), 
                o.observation_period_end_date, '1 month'
            )::date as possible_end_dates,
            p.year_of_birth,
            p.y
        from {cdm_schema}.observation_period o
        inner join outcome0_people p
        on o.person_id = p.person_id
    ),


    -- Age inclusion criteria --
    outcome0_people_age as (
        select
            person_id,
            start,
            finish,
            year_of_birth,
            possible_end_dates,
            y
        from outcome0_people_with_dates
        where extract(
            year from possible_end_dates
        ) - year_of_birth > 70
    ),

    -- ...and check eligibility before prediction time...
    outcome0_training_elig_counts as (
        select 
            o.person_id,
            o.observation_period_start_date as start,
            o.observation_period_end_date as finish,
            p.possible_end_dates,
            greatest(
                least (
                    o.observation_period_end_date,
                    date (p.possible_end_dates)
                ) - greatest(
                    o.observation_period_start_date,
                    date (p.possible_end_dates - interval '{eligibility_period}')
                ), 0
            ) as num_days
        from {cdm_schema}.observation_period o
        inner join outcome0_people_age p
        on o.person_id = p.person_id
    ),   


    -- New inclusion criteria: 100% in window --
    outcome0_training_elig_perc as (
        select
            person_id,
            possible_end_dates
        from
            outcome0_training_elig_counts
        group by
            person_id, possible_end_dates
        having
            sum(num_days) >= 0.95 * extract(        
                epoch from (
                    interval '{eligibility_period}' --epoch returns the number of seconds in eligibility_period
                )
            )/(24*60*60) --convert seconds to days
    ),

    -- ...as well as eligibility in prediction window...
    outcome0_test_elig_counts as (
        select
            p.person_id,
            p.observation_period_start_date as start,
            p.observation_period_end_date as finish,
            tr.possible_end_dates,
            greatest(
                    least (
                        p.observation_period_end_date,
                        date (
                            tr.possible_end_dates
                            + interval '{gap}'
                            + interval '{outcome_window}'
                        )
                    ) - greatest(
                        p.observation_period_start_date,
                        date(tr.possible_end_dates)
                    ), 0
            ) as num_days
        from {cdm_schema}.observation_period p
        inner join 
            outcome0_training_elig_perc tr
        on 
            tr.person_id = p.person_id
    ), 
    outcome0_test_elig_perc as (
        select
            person_id,
            possible_end_dates
        from
            outcome0_test_elig_counts
        group by
            person_id,
            possible_end_dates
        having
            sum(num_days) >= 0.95 * extract(
                epoch from (
                    interval '{gap}' 
                    + interval '{outcome_window}' --epoch returns the number of seconds in gap + outcome_window
                )
            )/(24*60*60) --convert seconds to days
    ),

    -- ...to get final cohort of samples with outcome = 0.
    eligible_outcome0_people as (
        select
            p.person_id,
            nt.possible_end_dates as end_date,
            p.outcome_date,
            p.y,
            row_number() over (partition by nt.possible_end_dates order by random()) as rk
        from outcome0_people_with_dates p
        join outcome0_test_elig_perc nt
        on p.person_id = nt.person_id
        and p.possible_end_dates = nt.possible_end_dates
    ),

    -- Compute how many samples wit outcome = 0 we need at each date...
    outcome1_date_counts as (
        select
            date_trunc('month', end_date) as rounded_end_date,
            count(*) as num_outcome1s
        from eligible_outcome1_people
        group by date_trunc('month', end_date)
    ),

    -- ...index samples with outcome = 1...
    outcome1_date_indexed as (
        select
            person_id,
            date_trunc('month', end_date) as rounded_end_date,
            end_date,
            row_number() over (partition by date_trunc('month', end_date) order by random()) as outcome1_rk
        from eligible_outcome1_people
    ),

    -- ...collect the samples wit outcome = 0 that we want...
    outcome0s_matching_outcome1_dates as (
        select
            n.person_id,
            n.end_date,
            n.outcome_date,
            n.y,
            n.rk
        from eligible_outcome0_people n
        join outcome1_date_counts p
        on n.end_date = p.rounded_end_date
            and n.rk <= cast('{outcome_ratio}' as int) * p.num_outcome1s
    ), -- match this ratio of samples with outcome = 0 for each sample with outcome = 1

    -- ...then assign the dates with outcome = 1 to the corresponding samples with outcome = 0.
    outcome0s_matched as (
        select
            n.person_id, 
            p.end_date,     -- use end date for sample with outcome = 1
            n.outcome_date,
            n.y
        from outcome0s_matching_outcome1_dates n
        join outcome1_date_indexed p
        on p.rounded_end_date = n.end_date
            and n.rk >= cast('{outcome_ratio}' as int) * p.outcome1_rk
            and n.rk < cast('{outcome_ratio}' as int) * (p.outcome1_rk+1)
    )

    select
        row_number() over (order by person_id) - 1 as example_id,
        null::timestamp as start_date,            -- required by omop-learn
        *
    from (
        select * from eligible_outcome1_people
        union all
        select * from outcome0s_matched 
    ) tmp
    where end_date <= date '{max_prediction_date}'    -- only use samples up to this point
    ;