import pandas as pd
from config import omop_schema, user_schema
from sqlalchemy import Column, BigInteger, Integer, String

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from sqlalchemy import func, and_, or_
from sqlalchemy.sql.expression import select, literal, case
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class EolCohortTable(Base):
    """End of Life (EOL) OMOP cohort.

    SqlAlchemy ORM class definition that can be used to define an EOL cohort.

    Keyword arguments
    ----------
    training_start_date -- datetime.datetime, The earliest date from which to look for eligible people for the cohort.
    training_end_date -- datetime.datetime, The latest date from which to look for eligible people for the cohort.
    gap_months -- int, The number of months between training_end_date and the start of the prediction date range.
    outcome_months -- int, The number of months between the start and end of the prediction date range.
    min_enroll_proportion -- float, A number between 0 and 1. The minimum proportion of days a member needs to be enrolled during training and prediction date ranges.
    """
    __tablename__ = 'eol_cohort'
    __table_args__ = {
      'schema': user_schema
    }

    training_start_date   = None
    training_end_date     = None
    gap_months            = None
    outcome_months        = None
    min_enroll_proportion = 0.95
    
    # For backward compatibility with original FeatureGenerator class
    _cohort            = None
    _schema_name       = user_schema
    _cohort_table_name = 'eol_cohort'

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
    
    def __repr__(self):
        return (
          "<EolCohort(example_id='%s', person_id='%s', start_date='%s', end_date='%s', outcome_date='%s', y='%s')>" % (
              self.example_id, self.person_id, self.start_date, self.end_date, self.outcome_date, self.y
          )
        )

    def build(self, db, replace=False):
        """Build an End of Life (EOL) OMOP cohort.
        
        Contains the logic to build an EOL cohort, including:
            1. Find members > 70 years old with data from the training and prediction windows, 
            2. Mark members who have a death date during the prediction window as 1, otherwise 0.
            
        Parameters
        ----------
        db: One of the database classes defined in Utils.ORM, such as PostgresDatabase.
        
        Notes
        -----
        Does not return anything. However, populates a table, eol_cohort, in a schema based on the value set in config.user_schema. Also stores a pandas dataframe of the results in the _cohort class variable.
        """
        if replace:
            with db.session.session_manager() as session:
                try:
                    self.__table__.drop(session.get_bind()) # Drop the table if it already exists, otherwise we wouldn't be calling build
                except:
                    pass

                self.__table__.create(session.get_bind())
                
                # Step 1: Add table references from db parameter (db contains an 'inspector', as referenced in InspectOMOP docs).
                Person            = db.inspector.tables['person']
                ObservationPeriod = db.inspector.tables['observation_period']
                
                # Step 1: Get death dates of members
                death_dates = session.query(
                    Person.person_id,
                    Person.death_datetime
                )\
                .subquery('death_dates')
                
                # Step 2: Find members greater than 70 years of age
                eligible_people = session.query(
                    Person.person_id
                )\
                .filter(
                    (self.training_end_date.year - Person.year_of_birth) > 70
                )\
                .subquery('eligible_people')
                
                # Step 3: Calculate prediction window
                prediction_start_date = self.training_end_date + relativedelta(months=self.gap_months)
                prediction_end_date = self.training_end_date + relativedelta(months=self.gap_months+self.outcome_months)

                # Step 4: Identify person IDs that occur during our training timeframe
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
                .join(eligible_people, ObservationPeriod.person_id == eligible_people.c.person_id)\
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
                .join(death_dates, test_period_elig_counts.c.person_id == death_dates.c.person_id)\
                .group_by(test_period_elig_counts.c.person_id, death_dates.c.death_datetime)\
                .having(
                    or_(
                        and_(death_dates.c.death_datetime >= prediction_start_date, death_dates.c.death_datetime <= prediction_end_date)
                        , func.sum(test_period_elig_counts.c.num_days) >= 
                        (self.min_enroll_proportion * (prediction_end_date - self.training_end_date))
                    )
                )\
                .subquery('test_window_elig_percent')
                
                # Step 5: Define our final table of person_id, start_date, end_date, and y
                # IMPORTANT: MIT's feature creation process relies on this table being sorted in ascending order by 
                # person id and example id. example id is auto populated based on our table definition.
                labeler = case(
                    [(death_dates.c.death_datetime.between(prediction_start_date, prediction_end_date), 1)], else_ = 0
                ).label('y')
                cohort = session\
                .query(test_window_elig_percent.c.person_id,
                       literal(self.training_start_date).label('start_date'),
                       literal(self.training_end_date).label('end_date'),
                       death_dates.c.death_datetime.label('outcome_date'),
                       labeler)\
                .join(death_dates, death_dates.c.person_id == test_window_elig_percent.c.person_id, isouter=True)\
                .filter(or_(death_dates.c.death_datetime == None, death_dates.c.death_datetime >= prediction_start_date))\
                .order_by(test_window_elig_percent.c.person_id)\
                .subquery('cohort')

                # Step 6: Insert the cohort to our current table
                sel = select([cohort])
                ins = self.__table__.insert().from_select(['person_id', 'start_date', 'end_date', 'outcome_date', 'y'], sel)
                session.execute(ins)
                
        # Save a representation of our cohort into a local pandas data frame for use in feature generation code
        self._cohort = pd.read_sql("SELECT * FROM " + user_schema + "." + self._cohort_table_name, db.engine)
        for date_col in ['start_date', 'end_date']:
            self._cohort[date_col] = pd.to_datetime(self._cohort[date_col])
        self._cohort = self._cohort.astype(
            {k:v for k,v in self._dtype.items() if k in self._cohort.columns}
        )
