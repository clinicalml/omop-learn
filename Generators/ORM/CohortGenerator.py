import pandas as pd
from config import omop_schema, user_schema
from sqlalchemy import Column, BigInteger, Integer, String

from ORMTables.cdm_6_0 import Base, ConditionOccurrence, ProcedureOccurrence, DrugExposure, DeviceExposure, ObservationPeriod

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from sqlalchemy import func, and_, or_, union
from sqlalchemy.sql.expression import select, literal, case

class CohortTable(Base):
    __tablename__ = 'omop_learn_cohort'
    __table_args__ = {
      'schema': user_schema
    }

    training_start_date   = None
    training_end_date     = None
    gap_months            = None
    outcome_months        = None
    min_enroll_proportion = 0.95
    inclusion_concept_ids = None
    exclusion_concept_ids = None
    target_concept_ids    = None
    
    # For backward compatibility with original FeatureGenerator class
    _cohort            = None
    _schema_name       = user_schema
    _cohort_table_name = 'omop_learn_cohort'

    _dtype = {}
    _dtype['example_id'] = int
    _dtype['person_id'] = int
    _dtype['person_source_value'] = str
    _dtype['start_date'] = str
    _dtype['end_date'] = str

    # Table columns
    example_id   = Column(Integer, primary_key=True, nullable=True)
    person_id    = Column(BigInteger)
    start_date   = Column(String(length=20))
    end_date     = Column(String(length=20))
    outcome_date = Column(String(length=20))
    y            = Column(Integer)
    
    domain_table_dict = {
        'Condition': {'table': ConditionOccurrence, 'concept_column': 'condition_concept_id', 'date_column': 'condition_start_date'}, 
        'Procedure': {'table': ProcedureOccurrence, 'concept_column': 'procedure_concept_id', 'date_column': 'procedure_date'},
        'Drug': {'table': DrugExposure, 'concept_column': 'drug_concept_id', 'date_column': 'drug_exposure_start_date'}, 
        'Device': {'table': DeviceExposure, 'concept_column': 'device_concept_id', 'date_column': 'device_exposure_start_date'}
    }

    def __repr__(self):
        return (
          "<Cohort(example_id='%s', person_id='%s', start_date='%s', end_date='%s', outcome_date='%s', y='%s')>" % (
              self.example_id, self.person_id, self.start_date, self.end_date, self.outcome_date, self.y
          )
        )

    def build(self, db, replace=False):
        if replace:
            with db.session.session_manager() as session:
                try:
                    self.__table__.drop(session.get_bind()) # Drop the table if it already exists, otherwise we wouldn't be calling build
                except:
                    pass

                self.__table__.create(session.get_bind())

                # Step 1: Calculate prediction end date
                prediction_end_date = self.training_end_date + relativedelta(months=self.gap_months+self.outcome_months) - timedelta(1)

                # Step 2: Identify person IDs that occur during our training timeframe
                least_end_date = case(
                    [(ObservationPeriod.observation_period_end_date < self.training_end_date, ObservationPeriod.observation_period_end_date)], 
                    else_ = self.training_end_date
                )
                greatest_start_date = case(
                    [(ObservationPeriod.observation_period_start_date > self.training_start_date, ObservationPeriod.observation_period_start_date)], 
                    else_ = self.training_start_date
                )
                num_days_expr = case(
                    [((least_end_date - greatest_start_date) > timedelta(0), least_end_date - greatest_start_date)],
                    else_ = timedelta(0)
                )

                training_elig_counts = session\
                .query(
                    ObservationPeriod.person_id, 
                    num_days_expr.label('num_days')
                )\
                .filter(ObservationPeriod.person_id > 0)\
                .subquery('training_elig_counts')

                training_window_elig_percent = session\
                .query(
                    training_elig_counts.c.person_id
                )\
                .group_by(training_elig_counts.c.person_id)\
                .having(func.sum(training_elig_counts.c.num_days) >= (self.min_enroll_proportion * (self.training_end_date - self.training_start_date)))\
                .subquery('training_window_elig_percent')

                least_end_date = case(
                    [(ObservationPeriod.observation_period_end_date < prediction_end_date, ObservationPeriod.observation_period_end_date)], 
                    else_ = prediction_end_date
                )
                greatest_start_date = case(
                    [(ObservationPeriod.observation_period_start_date > self.training_end_date, ObservationPeriod.observation_period_start_date)], 
                    else_ = self.training_end_date
                )
                num_days_expr = case(
                    [((least_end_date - greatest_start_date) > timedelta(0), least_end_date - greatest_start_date)],
                    else_ = timedelta(0)
                )   

                test_period_elig_counts = session\
                .query(ObservationPeriod.person_id, num_days_expr.label('num_days'))\
                .join(training_window_elig_percent, training_window_elig_percent.c.person_id == ObservationPeriod.person_id)\
                .filter(ObservationPeriod.person_id > 0)\
                .subquery('test_period_elig_counts')

                test_window_elig_percent = session\
                .query(test_period_elig_counts.c.person_id)\
                .group_by(test_period_elig_counts.c.person_id)\
                .having(func.sum(test_period_elig_counts.c.num_days) >= (self.min_enroll_proportion * (prediction_end_date - self.training_end_date)))\
                .subquery('test_window_elig_percent')

                # Step 3: Iteratively find PEID labels based on inclusion criteria
                if self.inclusion_concept_ids is not None:
                    selectables = []
                    for domain in self.domain_table_dict:
                        table          = self.domain_table_dict[domain]['table']
                        concept_column = getattr(table, self.domain_table_dict[domain]['concept_column'])
                        date_column    = getattr(table, self.domain_table_dict[domain]['date_column'])
                        selectables    += [select([table.person_id]).where(and_(concept_column.in_(self.inclusion_concept_ids), date_column.between(self.training_end_date, prediction_end_date)))]

                    person_ids_incl = session.query(union(*selectables).alias('union')).subquery('person_ids_incl')
                    del selectables
                    
                    elig_inclu = session\
                    .query(test_window_elig_percent)\
                    .join(person_ids_incl, test_window_elig_percent.c.person_id == person_ids_incl.c.person_id)\
                    .subquery('elig_inclu')
                else:
                    elig_inclu = test_window_elig_percent
                
                # Step 4: Iteratively find PEID labels based on exclusion criteria
                if self.exclusion_concept_ids is not None:
                    selectables = []
                    for domain in self.domain_table_dict:
                        table          = self.domain_table_dict[domain]['table']
                        concept_column = getattr(table, self.domain_table_dict[domain]['concept_column'])
                        date_column    = getattr(table, self.domain_table_dict[domain]['date_column'])
                        selectables    += [select([table.person_id]).where(and_(concept_column.in_(self.exclusion_concept_ids), date_column.between(self.training_end_date, prediction_end_date)))]

                    person_ids_excl = session.query(union(*selectables).alias('union')).subquery('person_ids_excl')
                    del selectables
                    
                    person_ids_elig = session\
                    .query(elig_inclu)\
                    .join(person_ids_excl, test_window_elig_percent.c.person_id == person_ids_excl.c.person_id, isouter=True)\
                    .filter(person_ids_excl.c.person_id == None)\
                    .subquery('person_ids_elig')
                else:
                    person_ids_elig = elig_inclu

                # Step 5: Iteratively find PEID labels based on target concept IDs
                selectables = []
                for domain in self.domain_table_dict:
                    for target in self.target_concept_ids:
                        table          = self.domain_table_dict[domain]['table']
                        concept_column = getattr(table, self.domain_table_dict[domain]['concept_column'])
                        date_column    = getattr(table, self.domain_table_dict[domain]['date_column'])
                        selectables    += [select([table.person_id, date_column.label('date'), literal(target).label('y')]).where(and_(concept_column.in_(self.target_concept_ids[target]), date_column.between(self.training_end_date, prediction_end_date)))]

                union_stmt = session.query(union(*selectables).alias('union')).subquery('union_stmt')
                del selectables

                person_ids_pos = session\
                .query(union_stmt.c.person_id, func.min(union_stmt.c.date).label('outcome_date'), union_stmt.c.y)\
                .group_by(union_stmt.c.person_id, union_stmt.c.y)\
                .subquery('person_ids_pos')
                del union_stmt

                # Step 6: Define our final table of person_id, start_date, end_date, and y
                # IMPORTANT: MIT's feature creation process relies on this table being sorted in ascending order by 
                # person id and example id.
                labeler = case([(person_ids_pos.c.y == None, 0)], else_ = person_ids_pos.c.y).label('y')
                cohort = session\
                .query(person_ids_elig.c.person_id, 
                       literal(self.training_start_date).label('start_date'), 
                       literal(self.training_end_date).label('end_date'), 
                       person_ids_pos.c.outcome_date, 
                       labeler)\
                .join(person_ids_pos, person_ids_elig.c.person_id == person_ids_pos.c.person_id, isouter=True)\
                .order_by(person_ids_elig.c.person_id)\
                .subquery('cohort')

                # Step 7: Insert the cohort to our current table
                sel = select([cohort])
                ins = self.__table__.insert().from_select(['person_id', 'start_date', 'end_date', 'outcome_date', 'y'], sel)
                session.execute(ins)
                
                
        # Save a representation of our cohort into a local pandas data frame for use in feature generation code
        self._cohort = pd.read_sql("SELECT * FROM " + user_schema + ".omop_learn_cohort", db.engine)
        for date_col in ['start_date', 'end_date']:
            self._cohort[date_col] = pd.to_datetime(self._cohort[date_col])
        self._cohort = self._cohort.astype(
            {k:v for k,v in self._dtype.items() if k in self._cohort.columns}
        )
