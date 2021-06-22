import pandas as pd
from config import omop_schema, user_schema
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, BigInteger, Integer, Float, Date, DateTime, String, ForeignKey, Boolean, Text

Base = declarative_base()

"""
Standardized clinical data (COMPLETE)
"""
class Person(Base):
    __tablename__ = 'person'
    __table_args__ = {
        'schema': omop_schema
    }
    person_id                   = Column(BigInteger, primary_key=True, nullable=False)
    gender_concept_id           = Column(Integer)
    year_of_birth               = Column(Integer)
    month_of_birth              = Column(Integer)
    day_of_birth                = Column(Integer)
    birth_datetime              = Column(DateTime)
    death_datetime              = Column(DateTime)
    race_concept_id             = Column(Integer)
    ethnicity_concept_id        = Column(Integer)
    location_id                 = Column(BigInteger)
    provider_id                 = Column(BigInteger)
    care_site_id                = Column(BigInteger)
    person_source_value         = Column(String(length=50))
    gender_source_value         = Column(String(length=50))
    gender_source_concept_id    = Column(Integer)
    race_source_value           = Column(String(length=50))
    race_source_concept_id      = Column(Integer)
    ethnicity_source_value      = Column(String(length=50))
    ethnicity_source_concept_id = Column(Integer)

    def __repr__(self):
        return (
            "<Person(person_id='%s', gender_concept_id='%s', birth_datetime='%s', death_datetime='%s')>" % (
                self.person_id, self.gender_concept_id, self.birth_datetime, self.death_datetime
            )
        )

class ObservationPeriod(Base):
    __tablename__ = 'observation_period'
    __table_args__ = {
      'schema': omop_schema
    }

    observation_period_id         = Column(BigInteger, primary_key=True, nullable=False)
    person_id                     = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'person', 'person_id'])), nullable=False)
    observation_period_start_date = Column(Date, nullable=False)
    observation_period_end_date   = Column(Date, nullable=False)
    period_type_concept_id        = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)

    def __repr__(self):
        return (
            "<ObservationPeriod(observation_period_id='%s', person_id='%s', observation_period_start_date='%s', observation_period_end_date='%s')>" % (
                self.observation_period_id, self.person_id, self.observation_period_start_date, self.observation_period_end_date
            )
        )

class VisitOccurrence(Base):
    __tablename__ = 'visit_occurrence'
    __table_args__ = {
      'schema': omop_schema
    }

    visit_occurrence_id           = Column(BigInteger, primary_key=True, nullable=False)
    person_id                     = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'person', 'person_id'])), nullable=False)
    visit_concept_id              = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    visit_start_date              = Column(DateTime, nullable=False)
    visit_start_datetime          = Column(Date)
    visit_end_date                = Column(DateTime)
    visit_end_datetime            = Column(Date)
    visit_type_concept_id         = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    provider_id                   = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'provider', 'provider_id'])))
    care_site_id                  = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'care_site', 'care_site_id'])))
    visit_source_value            = Column(String(length=50))
    visit_source_concept_id       = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    admitted_from_concept_id      = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    admitted_from_source_value	  = Column(String(length=50))
    discharge_to_concept_id       = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    discharge_to_source_value     = Column(String(length=50))
    preceding_visit_occurrence_id	= Column(BigInteger, ForeignKey('.'.join([omop_schema, 'visit_occurrence', 'visit_occurrence_id'])))

    def __repr__(self):
        return (
            "<VisitOccurrence(visit_occurrence_id='%s', person_id='%s', visit_concept_id='%s')>" % (
                self.visit_occurrence_id, self.person_id, self.visit_concept_id
            )
        )

class VisitDetail(Base):
    __tablename__ = 'visit_detail'
    __table_args__ = {
      'schema': omop_schema
    }  
    visit_detail_id               = Column(BigInteger, primary_key=True, nullable=False)
    person_id                     = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'person', 'person_id'])), nullable=False)
    visit_detail_concept_id       = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    visit_detail_start_date       = Column(DateTime, nullable=False)
    visit_detail_start_datetime   = Column(Date)
    visit_detail_end_date         = Column(DateTime)
    visit_detail_end_datetime     = Column(Date)
    visit_detail_type_concept_id  = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    provider_id                   = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'provider', 'provider_id'])))
    care_site_id                  = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'care_site', 'care_site_id'])))
    visit_detail_source_value     = Column(String(length=50))
    visit_detail_source_concept_id = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    admitting_source_value        = Column(String(length=50))
    admitting_source_concept_id   = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    discharge_to_concept_id       = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    preceding_visit_detail_id     = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'visit_detail', 'visit_detail_id'])))
    visit_detail_parent_id        = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'visit_detail', 'visit_detail_id'])))
    visit_occurrence_id           = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'visit_occurrence', 'visit_occurrence_id'])), nullable=False)
    
    def __repr__(self):
        return (
            "<VisitDetail(visit_detail_id='%s', person_id='%s', visit_detail_concept_id='%s')>" % (
                self.visit_detail_id, self.person_id, self.visit_detail_concept_id
            ) 
        )
    
    
class ConditionOccurrence(Base):
    __tablename__ = 'condition_occurrence'
    __table_args__ = {
      'schema': omop_schema
    }
    condition_occurrence_id       = Column(BigInteger, primary_key=True, nullable=False)
    person_id                     = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'person', 'person_id'])), nullable=False)
    condition_concept_id          = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    condition_start_date          = Column(DateTime, nullable=False)
    condition_start_datetime      = Column(Date)
    condition_end_date            = Column(DateTime)
    condition_end_datetime        = Column(Date)
    condition_type_concept_id     = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    condition_status_concept_id   = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    stop_reason                   = Column(String(length=20))
    provider_id                   = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'provider', 'provider_id'])))
    visit_occurrence_id           = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'visit_occurrence', 'visit_occurrence_id'])))
    visit_detail_id               = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'visit_detail', 'visit_detail_id'])))
    condition_source_value        = Column(String(length=50))
    condition_source_concept_id   = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    condition_status_source_value = Column(String(length=50))

    def __repr__(self):
        return (
            "<ConditionOccurrence(condition_occurrence_id='%s', person_id='%s', condition_concept_id='%s')>" % (
                self.condition_occurrence_id, self.person_id, self.condition_concept_id
            )
        )


class DrugExposure(Base):
    __tablename__ = 'drug_exposure'
    __table_args__ = {
      'schema': omop_schema
    }
    drug_exposure_id              = Column(BigInteger, primary_key=True, nullable=False)
    person_id                     = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'person', 'person_id'])), nullable=False)
    drug_concept_id               = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    drug_exposure_start_date      = Column(Date, nullable=False)
    drug_exposure_start_datetime  = Column(DateTime)
    drug_exposure_end_date        = Column(Date, nullable=False)
    drug_exposure_end_datetime    = Column(DateTime)
    verbatim_end_date             = Column(Date)
    drug_type_concept_id          = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    stop_reason                   = Column(String(length=20))
    refills                       = Column(Integer)
    quantity                      = Column(Float)
    days_supply                   = Column(Integer)
    sig                           = Column(Text)
    route_concept_id              = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])))
    lot_number                    = Column(String(length=50))
    provider_id                   = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'provider', 'provider_id'])))
    visit_occurrence_id           = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'visit_occurrence', 'visit_occurrence_id'])))
    visit_detail_id               = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'visit_detail', 'visit_detail_id'])))
    drug_source_value             = Column(String(length=50))
    drug_source_concept_id        = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    route_source_value            = Column(String(length=50))
    dose_unit_source_value        = Column(String(length=50))

    def __repr__(self):
        return (
            "<DrugExposure(drug_exposure_id='%s', person_id='%s', drug_concept_id='%s')>" % (
                self.drug_exposure_id, self.person_id, self.drug_concept_id
            )
        )


class ProcedureOccurrence(Base):
    __tablename__ = 'procedure_occurrence'
    __table_args__ = {
      'schema': omop_schema
    }
    procedure_occurrence_id       = Column(BigInteger, primary_key=True, nullable=False)
    person_id                     = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'person', 'person_id'])), nullable=False)
    procedure_concept_id          = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    procedure_date                = Column(Date, nullable=False)
    procedure_datetime            = Column(DateTime)
    procedure_type_concept_id     = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    modifier_concept_id           = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    quantity                      = Column(Integer)
    provider_id                   = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'provider', 'provider_id'])))
    visit_occurrence_id           = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'visit_occurrence', 'visit_occurrence_id'])))
    visit_detail_id               = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'visit_detail', 'visit_detail_id'])))
    procedure_source_value        = Column(String(length=50))
    procedure_source_concept_id	  = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    modifier_source_value         = Column(String(length=50))

    def __repr__(self):
        return (
            "<ProcedureOccurrence(procedure_occurrence_id='%s', person_id='%s', procedure_concept_id='%s')>" % (
                self.procedure_occurrence_id, self.person_id, self.procedure_concept_id
            )
        )


class DeviceExposure(Base):
    __tablename__ = 'device_exposure'
    __table_args__ = {
      'schema': omop_schema
    }  
    device_exposure_id            = Column(BigInteger, primary_key=True, nullable=False)
    person_id                     = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'person', 'person_id'])), nullable=False)
    device_concept_id             = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    device_exposure_start_date    = Column(Date, nullable=False)
    device_exposure_start_datetime = Column(DateTime)
    device_exposure_end_date      = Column(Date)
    device_exposure_end_datetime  = Column(DateTime)
    device_type_concept_id        = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    unique_device_id              = Column(String(length=50))
    quantity                      = Column(Integer)
    provider_id                   = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'provider', 'provider_id'])))
    visit_occurrence_id           = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'visit_occurrence', 'visit_occurrence_id'])))
    visit_detail_id               = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'visit_detail', 'visit_detail_id'])))
    device_source_value           = Column(String(length=50))
    device_source_concept_id      = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)

    def __repr__(self):
        return (
            "<DeviceExposure(device_exposure_id='%s', person_id='%s', device_concept_id='%s')>" % (
                self.device_exposure_id, self.person_id, self.device_concept_id
            )
        )

    
    
class Measurement(Base):
    __tablename__ = 'measurement'
    __table_args__ = {
      'schema': omop_schema
    }
    measurement_id                = Column(BigInteger, primary_key=True, nullable=False)
    person_id                     = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'person', 'person_id'])), nullable=False)
    measurement_concept_id        = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    measurement_date              = Column(Date, nullable=False)
    measurement_datetime          = Column(DateTime)
    measurement_time              = Column(String(length=10))
    measurement_type_concept_id   = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    operator_concept_id           = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    value_as_number               = Column(Float)
    value_as_concept_id           = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])))
    unit_concept_id               = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])))
    range_low                     = Column(Float)
    range_high                    = Column(Float)
    provider_id                   = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'provider', 'provider_id'])))
    visit_occurrence_id           = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'visit_occurrence', 'visit_occurrence_id'])))
    visit_detail_id               = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'visit_detail', 'visit_detail_id'])))
    measurement_source_value      = Column(String(length=50))
    measurement_source_concept_id = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    unit_source_value             = Column(String(length=50))
    value_source_value            = Column(String(length=50))
    
    def __repr__(self):
        return (
            "<Measurement(measurement_id='%s', person_id='%s', measurement_concept_id='%s')>" % (
                self.measurement_id, self.person_id, self.measurement_concept_id
            )
        )

class Observation(Base):
    __tablename__ = 'observation'
    __table_args__ = {
      'schema': omop_schema
    }
    observation_id                = Column(BigInteger, primary_key=True, nullable=False)
    person_id                     = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'person', 'person_id'])), nullable=False)
    observation_concept_id        = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    observation_date              = Column(Date, nullable=False)
    observation_datetime          = Column(DateTime)                           
    observation_type_concept_id   = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    value_as_number               = Column(Float)
    value_as_concept_id           = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])))
    qualifier_concept_id          = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])))
    unit_concept_id               = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])))
    provider_id                   = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'provider', 'provider_id'])))
    visit_occurrence_id           = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'visit_occurrence', 'visit_occurrence_id'])))
    visit_detail_id               = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'visit_detail', 'visit_detail_id'])))
    observation_source_value      = Column(String(length=50))
    observation_source_concept_id = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    unit_source_value             = Column(String(length=50))
    qualifier_source_value        = Column(String(length=50))
    observation_event_id          = Column(BigInteger)                   
    obs_event_field_concept_id    = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])))
    value_as_datetime             = Column(DateTime)                                       

    def __repr__(self):
        return (
            "<Observation(observation_id='%s', person_id='%s', observation_concept_id='%s')>" % (
                self.observation_id, self.person_id, self.observation_concept_id
            )
        )

class Note(Base):
    __tablename__ = 'note'
    __table_args__ = {
      'schema': omop_schema
    }  
    note_id                       = Column(Integer, primary_key=True, nullable=False)
    person_id                     = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'person', 'person_id'])), nullable=False)
    note_event_id                 = Column(BigInteger)
    note_event_field_concept_id   = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])))
    note_event_date               = Column(Date, nullable=False)
    note_datetime                 = Column(DateTime)
    note_type_concept_id          = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    note_class_concept_id         = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    note_title                    = Column(String(length=250))
    note_text                     = Column(Text)
    encoding_concept_id           = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    language_concept_id           = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    provider_id                   = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'provider', 'provider_id'])))
    visit_occurrence_id           = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'visit_occurrence', 'visit_occurrence_id'])))
    visit_detail_id               = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'visit_detail', 'visit_detail_id'])))
    note_source_value             = Column(String(length=50))
  
    def __repr__(self):
        return (
            "<Note(note_id='%s', person_id='%s', note_event_id='%s')>" % (
                self.note_id, self.person_id, self.note_event_id
            )
        )

class NoteNLP(Base):
    __tablename__ = 'note_NLP'
    __table_args__ = {
      'schema': omop_schema
    }  
    note_nlp_id                   = Column(BigInteger, primary_key=True, nullable=False)
    note_id                       = Column(Integer, nullable=False)
    section_concept_id            = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])))
    snippet                       = Column(String(length=250))
    offset                        = Column(String(length=50))
    lexical_variant               = Column(String(length=250), nullable=False)
    note_nlp_concept_id           = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])))
    note_nlp_source_concept_id    = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])))
    nlp_system                    = Column(String(length=250))
    nlp_date                      = Column(Date, nullable=False)
    nlp_datetime                  = Column(DateTime)
    term_exists                   = Column(String(length=1))
    term_temporal                 = Column(String(length=50))
    term_modifiers                = Column(String(length=2000))
  
    def __repr__(self):
        return (
            "<NoteNLP(note_nlp_id='%s', note_id='%s', section_concept_id='%s')>" % (
                self.note_nlp_id, self.note_id, self.section_concept_id
            )
        )

class Specimen(Base):
    __tablename__ = 'specimen'
    __table_args__ = {
      'schema': omop_schema
    }  
    specimen_id                   = Column(BigInteger, primary_key=True, nullable=False)
    person_id                     = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'person', 'person_id'])), nullable=False)
    specimen_concept_id           = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    specimen_type_concept_id      = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    specimen_date                 = Column(Date, nullable=False)
    speciment_datetime            = Column(DateTime)
    quantity                      = Column(Float)
    unit_concept_id               = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])))
    anatomic_site_concept_id      = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])))
    disease_status_concept_id     = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])))
    specimen_source_id            = Column(String(length=50))
    specimen_source_value         = Column(String(length=50))
    unit_source_value             = Column(String(length=50))
    anatomic_site_source_value    = Column(String(length=50))
    disease_status_source_value   = Column(String(length=50))
    
    def __repr__(self):
        return (
            "<Specimen(specimen_id='%s', person_id='%s', specimen_concept_id='%s')>" % (
                self.specimen_id, self.person_id, self.specimen_concept_id
            )
        )

class FactRelationship():
    __tablename__ = 'fact_relationship'
    __table_args__ = {
      'schema': omop_schema
    }  
    domain_concept_id_1           = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    fact_id_1                     = Column(BigInteger, nullable=False)
    domain_concept_id_2           = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    fact_id_2                     = Column(BigInteger, nullable=False)
    relationship_concept_id       = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
  
    def __repr__(self):
        return (
            "<FactRelationship(domain_concept_id_1='%s', domain_concept_id_2='%s', relationship_concept_id='%s'>" %(
                self.domain_concept_id_1, self.domain_concept_id_2, self.relationship_concept_id
            )
        )

class SurveyConduct():
    __tablename__ = 'survey_conduct'
    __table_args__ = {
      'schema': omop_schema
    }  
    survey_conduct_id             = Column(BigInteger, primary_key=True, nullable=False)
    person_id                     = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'person', 'person_id'])), nullable=False)
    survey_concept_id             = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    survey_start_date             = Column(Date)
    survey_start_datetime         = Column(DateTime)
    survey_end_date               = Column(Date)
    survey_end_datetime           = Column(DateTime, nullable=False)
    provider_id                   = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'provider', 'provider_id'])))
    assisted_concept_id           = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    respondent_type_concept_id    = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    timing_concept_id             = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    collection_method_concept_id  = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    assisted_source_value         = Column(String(length=50))
    respondent_type_source_value  = Column(String(length=100))
    timing_source_value           = Column(String(length=100))
    collection_method_source_value= Column(String(length=100))
    survey_source_value           = Column(String(length=100))
    survey_source_concept_id      = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    survey_source_identifier      = Column(String(length=100))
    validated_survey_concept_id   = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    validated_survey_source_value = Column(Integer)
    survey_version_number         = Column(String(length=20))
    visit_occurrence_id           = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'visit_occurrence', 'visit_occurrence_id'])))
    response_visit_occurrence_id  = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'visit_occurrence', 'visit_occurrence_id'])))
  
    def __repr__(self):
        return (
            "<SurveyConduct(survey_conduct_id='%s', person_id='%s', survey_concept_id='%s'>" %(
                self.survey_conduct, self.person_id, self.survey_concept_id
            )
        ) 
    
"""
Standardized health system data (COMPLETE)
"""

class Provider(Base):
    __tablename__ = 'provider'
    __table_args__ = {
      'schema': omop_schema
    }
    provider_id                   = Column(BigInteger, primary_key=True, nullable=False)
    provider_name                 = Column(String(length=255))
    npi                           = Column(String(length=50))
    dea                           = Column(String(length=50))
    specialty_concept_id          = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    care_site_id                  = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'care_site', 'care_site_id'])))
    year_of_birth                 = Column(Integer)
    gender_concept_id             = Column(Integer)
    provider_source_value         = Column(String(length=50))
    specialty_source_value        = Column(String(length=50))
    specialty_source_concept_id   = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    gender_source_value           = Column(String(length=50))
    gender_source_concept_id      = Column(Integer)

    def __repr__(self):
        return (
            "<Provider(provider_id='%s', provider_name='%s', specialty_concept_id='%s')>" % (
                self.provider_id, self.provider_name, self.specialty_concept_id
            )
        )

class Location(Base):
    __tablename__ = 'location'
    __table_args__ = {
      'schema': omop_schema
    }
    location_id                     = Column(BigInteger, primary_key=True, nullable=False)
    address_1                       = Column(String(length=50))
    address_2                       = Column(String(length=50))
    city                            = Column(String(length=50))
    state                           = Column(String(length=2))
    zip                             = Column(String(length=9))
    county                          = Column(String(length=20))
    location_source_value           = Column(String(length=50))
    latitude                        = Column(Float)
    longitude                       = Column(Float)
  
    def __repr__(self):
        return (
            "<Location(location_id='%s', location_source_value='%s')>" % (
                self.location_id, self.location_source_value
            )
        )

# class LocationHistory(Base):
#     __tablename__ = 'location_history'
#     __table_args__ = {
#       'schema': omop_schema
#     }  
#     location_id                     = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'location', 'location_id'])), nullable=False)
#     relationship_type_concept_id    = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
#     domain_id                       = Column(String(length=50), nullable=False)
#     entity_id                       = Column(BigInteger, nullable=False)
#     start_date                      = Column(Date, nullable=False)
#     end_date                        = Column(Date)
  
#     def __repr__(self):
#         return(
#             "<LocationHistory(location_id='%s', relationship_type_concept_id='%s')>" % (
#                 self.location_id, self.relationship_type_concept_id
#             ) 
#         )
  

class CareSite(Base):
    __tablename__ = 'care_site'
    __table_args__ = {
      'schema': omop_schema
    }
    care_site_id                    = Column(BigInteger, primary_key=True, nullable=False)
    care_site_name                  = Column(String(length=255))
    place_of_service_concept_id     = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    location_id                     = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'location', 'location_id'])))
    care_site_source_value          = Column(String(length=50))
    place_of_service_source_value   = Column(String(length=50))

    def __repr__(self):
        return (
            "<CareSite(care_site_id='%s', care_site_name='%s', location_id='%s')>" % (
                self.care_site_id, self.care_site_name, self.location_id
            )
        )

"""
Health economics data (COMPLETE)

"""    
    
class PayerPlanPeriod(Base):
    __tablename__ = 'payer_plan_period'
    __table_args__ = {
      'schema': omop_schema
    }  
    payer_plan_period_id          = Column(BigInteger, primary_key=True, nullable=False)
    person_id                     = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'person', 'person_id'])), nullable=False)
    contract_person_id            = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'person', 'person_id'])))
    payer_plan_period_start_date  = Column(Date, nullable=False)
    payer_plan_period_end_date    = Column(Date, nullable=False)
    payer_concept_id              = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    payer_source_value            = Column(String(length=50))
    payer_source_concept_id       = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    plan_concept_id               = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    plan_source_value             = Column(String(length=50))
    plan_source_concept_id        = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    contract_concept_id           = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    contract_source_value         = Column(String(length=50))
    contract_source_concept_id    = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    sponsor_concept_id            = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    sponsor_source_value          = Column(String(length=50))
    sponsor_source_concept_id     = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    family_source_value           = Column(String(length=50))
    stop_reason_concept_id        = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])))
    stop_reason_source_value      = Column(String(length=50))
    stop_reason_source_concept_id = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])))

    def __repr__(self):
        return (
            "<PayerPlanPeriod(payer_plan_period_id='%s', person_id='%s', contract_person_id='%s')>" % (
                self.payer_plan_period_id, self.person_id, self.contract_person_id
            )
        )

class Cost(Base):
    __tablename__ = 'cost'
    __table_args__ = {
      'schema': omop_schema
    }  
    cost_id                       = Column(Integer, primary_key=True, nullable=False)
    cost_event_id                 = Column(BigInteger)
    cost_domain_id                = Column(String(length=20), ForeignKey('.'.join([omop_schema, 'domain', 'domain_id'])))
    cost_type_concept_id          = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    currency_concept_id           = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])))
    total_charge                  = Column(Float)
    total_cost                    = Column(Float)
    total_paid                    = Column(Float)
    paid_by_payer                 = Column(Float)
    paid_by_patient               = Column(Float)
    paid_patient_copay            = Column(Float) 
    paid_patient_coinsurance      = Column(Float)
    paid_patient_deductible       = Column(Float)
    paid_by_primary               = Column(Float)
    paid_ingredient_cost          = Column(Float)
    paid_dispensing_fee           = Column(Float)
    payer_plan_period_id          = Column(BigInteger)
    amount_allowed                = Column(Float)
    revenue_code_concept_id       = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])))
    revenue_code_source_value     = Column(String(length=50))
    drg_concept_id                = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])))
    drg_source_value              = Column(String(length=3))
  
    def __repr__(self):
        return (
          "<Cost(cost_id='%s', cost_event_id='%s', cost_domain_id='%s')>" % (
            self.cost_id, self.cost_event_id, self.cost_domain_id
          )
        )

"""
Standardized derived elements (COMPLETE)
"""  

class DrugEra(Base):
    __tablename__ = 'drug_era'
    __table_args__ = {
      'schema': omop_schema
    }  
    drug_era_id                   = Column(BigInteger, primary_key=True, nullable=False)
    person_id                     = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'person', 'person_id'])), nullable=False)
    drug_concept_id               = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    drug_era_start_date           = Column(DateTime, nullable=False)
    drug_era_end_date             = Column(DateTime, nullable=False)
    drug_exposure_count           = Column(Integer)
    gap_days                      = Column(Integer)
  
    def __repr__(self):
        return (
            "<DrugEra(drug_era_id='%s', person_id='%s', drug_concept_id='%s')>" % (
                self.drug_era_id, self.person_id, self.drug_concept_id
            )
        )

class DoseEra(Base):
    __tablename__ = 'dose_era'
    __table_args__ = {
      'schema': omop_schema
    }  
    dose_era_id                   = Column(BigInteger, primary_key=True, nullable=False)
    person_id                     = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'person', 'person_id'])), nullable=False)
    drug_concept_id               = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    unit_concept_id               = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    dose_value                    = Column(Float, nullable=False)
    dose_era_start_date           = Column(DateTime, nullable=False)
    dose_era_end_date             = Column(DateTime, nullable=False)
  
    def __repr__(self):
        return (
            "<DoseEra(dose_era_id='%s', person_id='%s', drug_concept_id='%s'>" % (
                self.dose_era_id, self.person_id, self.drug_concept_id
            )
        )

class ConditionEra(Base):
    __tablename__ = 'condition_era'
    __table_args__ = {
      'schema': omop_schema
    }  
    condition_era_id              = Column(BigInteger, primary_key=True, nullable=False)
    person_id                     = Column(BigInteger, ForeignKey('.'.join([omop_schema, 'person', 'person_id'])), nullable=False)
    condition_concept_id          = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
    condition_era_start_date      = Column(DateTime, nullable=False)
    condition_era_end_date        = Column(DateTime, nullable=False)
    condition_occurrence_count    = Column(Integer)

    def __repr__(self):
        return (
            "<ConditionEra(condition_era_id='%s', person_id='%s', condition_concept_id='%s'>" % (
                self.condition_era_id, self.person_id, self.condition_concept_id
            )
        )

"""
Metadata tables (COMPLETE)
"""

# class Metadata(Base):
#     __tablename__ = 'metadata'
#     __table_args__ = {
#       'schema': omop_schema
#     }  
#     metadata_concept_id      = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
#     metadata_type_concept_id = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
#     name                     = Column(String(length=250), nullable=False)
#     value_as_string          = Column(String(length=250))
#     value_as_concept_id      = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])))
#     metadata_date            = Column(Date)
#     metadata_datetime        = Column(DateTime)
  
#     def __repr__(self):
#         return (
#             "<Metadata(metadata_concept_id='%s', metadata_type_concept_id='%s', name='%s'>" % (
#                 self.metadata_concept_id, self.metadata_type_concept_id, self.name
#             )
#         )

# class CDMSource(Base):
#     __tablename__ = 'cdm_source'
#     __table_args__ = {
#       'schema': omop_schema
#     }  
#     cdm_source_name                = Column(String(length=255))
#     cdm_source_abbreviation        = Column(String(length=25))
#     cdm_holder                     = Column(String(length=255))
#     source_description             = Column(Text)
#     source_documentation_reference = Column(String(length=255))
#     cdm_etl_reference              = Column(String(length=255))
#     source_release_date            = Column(Date)
#     cdm_release_date               = Column(Date)
#     cdm_version                    = Column(String(length=10))
#     vocabulary_version             = Column(String(length=20))

#     def __repr__(self):
#         return (
#             "<CDMSource(cdm_source_name='%s', cdm_source_abbreviation='%s', cdm_holder='%s'>" % (
#                 self.cdm_source_name, self.cdm_source_abbreviation, self.cdm_holder
#             )
#         )

"""
Standardized vocabularies (COMPLETE)
"""
class Concept(Base):
    __tablename__ = 'concept'
    __table_args__ = {
      'schema': omop_schema
    }
    concept_id                    = Column(Integer, primary_key=True, nullable=False)
    concept_name                  = Column(String(length=255), nullable=False)
    domain_id                     = Column(String(length=50), ForeignKey('.'.join([omop_schema, 'domain', 'domain_id'])), primary_key=True, nullable=False)
    vocabulary_id                 = Column(String(length=20), ForeignKey('.'.join([omop_schema, 'vocabulary', 'vocabulary_id'])), nullable=False)
    concept_class_id              = Column(String(length=20), ForeignKey('.'.join([omop_schema, 'concept_class', 'concept_class_id'])), nullable=False)
    standard_concept              = Column(String(length=1))
    concept_code                  = Column(String(length=50))
    valid_start_date              = Column(Date, nullable=False)
    valid_end_date                = Column(Date, nullable=False)
    invalid_reason                = Column(String(length=1))

    def __repr__(self):
        return (
            "<Concept(concept_id='%s', concept_name='%s', domain_id='%s', vocabulary_id='%s')>" % (
                self.concept_id, self.concept_name, self.domain_id, self.vocabulary_id
            )
        )

class Vocabulary(Base):
    __tablename__ = 'vocabulary'
    __table_args__ = {
      'schema': omop_schema
    }
    vocabulary_id                 = Column(String(length=20), primary_key=True, nullable=False)
    vocabulary_name               = Column(String(length=255), nullable=False)
    vocabulary_reference          = Column(String(length=255), nullable=False)
    vocabulary_version            = Column(String(length=255))
    vocabulary_concept_id         = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)

    def __repr__(self):
        return (
            "<Vocabulary(vocabulary_id='%s', vocabulary_name='%s', vocabulary_reference='%s'>" % (
                self.vocabulary_id
            )
        )

class Domain(Base):
    __tablename__ = 'domain'
    __table_args__ = {
      'schema': omop_schema
    }
    domain_id                     = Column(String(length=20), primary_key=True, nullable=False)
    domain_name                   = Column(String(length=255), nullable=False)
    domain_concept_id             = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)

    def __repr__(self):
        return (
            "<Domain(domain_id='%s', domain_name='%s', domain_concept_id='%s')>" % (
                self.domain_id, self.domain_name, self.domain_concept_id
            )
        )

class ConceptClass(Base):
    __tablename__ = 'concept_class'

    concept_class_id              = Column(String(length=20), primary_key=True, nullable=False)
    concept_class_name            = Column(String(length=255), nullable=False)
    concept_class_concept_id      = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)

    def __repr__(self):
        return (
            "<ConceptClass(concept_class_id='%s', concept_class_name='%s', concept_class_concept_id='%s')>" % (
                self.concept_class_id, self.concept_class_name, self.concept_class_concept_id
            )
        )

class ConceptRelationship(Base):
    __tablename__ = "concept_relationship"
    __table_args__ = {
      'schema': omop_schema
    }  
    concept_id_1                  = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), primary_key=True, nullable=False)
    concept_id_2                  = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), primary_key=True, nullable=False)
    relationship_id               = Column(String(length=20), ForeignKey('.'.join([omop_schema, 'relationship', 'relationship_id'])), nullable=False)
    visit_start_date              = Column(DateTime, nullable=False)
    visit_end_date                = Column(DateTime, nullable=False)
    invalid_reason                = Column(String(length=1))
  
    def __repr__(self):
        return (
            "<ConceptRelationship(concept_id_1='%s', concept_id_2='%s', relationship_id='%s')>" % (
                self.concept_id_1, self.concept_id_2, self.relationship_id
            )
        )

class Relationship(Base):
    __tablename__ = "relationship"
    __table_args__ = {
      'schema': omop_schema
    }  
    relationship_id               = Column(String(length=20), primary_key=True, nullable=False)
    relationship_name             = Column(String(length=255), nullable=False)
    is_hierarchical               = Column(String(length=1), nullable=False)
    defines_ancestry              = Column(String(length=1), nullable=False)
    reverse_relationship_id	      = Column(String(length=20), nullable=False)
    relationship_concept_id       = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
  
    def __repr__(self):
        return (
            "<Relationship(relationship_id='%s', relationship_name='%s', is_hierarchical='%s')>" % (
                self.relationship_id, self.relationship_name, self.is_hierarchical
            )
        )

class ConceptSynonym(Base):
    __tablename__ = "concept_synonym"

    concept_id                    = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), primary_key=True, nullable=False)
    concept_synonym_name          = Column(String(length=1000))
    language_concept_id           = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), primary_key=True, nullable=False)

    def __repr__(self):
        return (
            "<ConceptSynonym(concept_id='%s', concept_synonym_name='%s', language_concept_id='%s')>" % (
                self.concept_id, self.concept_synonym_name, self.language_concept_id
            )
        )
    
class ConceptAncestor(Base):
    __tablename__ = "concept_ancestor"
    __table_args__ = {
      'schema': omop_schema
    }  
    ancestor_concept_id           = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), primary_key=True, nullable=False)
    descendant_concept_id         = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), primary_key=True, nullable=False)
    min_levels_of_separation      = Column(Integer, nullable=False)
    max_levels_of_separation      = Column(Integer, nullable=False)
  
    def __repr__(self):
        return (
            "<ConceptAncestor(ancestor_concept_id='%s', descendant_concept_id='%s')>" % (
                self.ancestor_concept_id, self.descendant_concept_id
            )
        )

# class SourceToConceptMap(Base):
#     __tablename__ = "source_to_concept_map"
#     __table_args__ = {
#       'schema': omop_schema
#     }  
#     source_code                   = Column(String(length=50), nullable=False)
#     source_concept_id             = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
#     source_vocabulary_id          = Column(String(length=20), ForeignKey('.'.join([omop_schema, 'vocabulary', 'vocabulary_id'])), nullable=False)
#     source_code_description       = Column(String(length=255))
#     target_concept_id             = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
#     target_vocabulary_id          = Column(String(length=20), ForeignKey('.'.join([omop_schema, 'vocabulary', 'vocabulary_id'])), nullable=False)
#     visit_start_date              = Column(DateTime, nullable=False)
#     visit_end_date                = Column(DateTime, nullable=False)
#     invalid_reason                = Column(String(length=1))
  
#     def __repr__(self):
#         return(
#             "<SourceToConceptMap(source_concept_id='%s', target_concept_id='%s')>" %(
#                 self.source_concept_id, self.target_concept_id
#             )
#         )

class DrugStrength(Base):
    __tablename__ = "drug_strength"
    __table_args__ = {
      'schema': omop_schema
    }  
    drug_concept_id               = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), primary_key=True, nullable=False)
    ingredient_concept_id         = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), primary_key=True, nullable=False)
    amount_value                  = Column(Float)
    amount_unit_concept_id        = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])))
    numerator_value               = Column(Float)
    numerator_unit_concept_id     = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])))
    denominator_value             = Column(Float)
    denominator_unit_concept_id   = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])))
    box_size                      = Column(Integer)
    visit_start_date              = Column(DateTime, nullable=False)
    visit_end_date                = Column(DateTime, nullable=False)
    invalid_reason                = Column(String(length=1))

    def __repr__(self):
        return(
            "<DrugStrength(drug_concept_id='%s', ingredient_concept_id='%s')>" %(
                self.drug_concept_id, self.ingredient_concept_id
            )
        )
    
# class CohortDefinition(Base):
#     __tablename__ = "cohort_definition"
#     __table_args__ = {
#       'schema': omop_schema
#     }  
#     cohort_definition_id          = Column(Integer, nullable=False)
#     cohort_definition_name        = Column(String(length=255), nullable=False)
#     cohort_definition_description = Column(Text, nullable=False)
#     definition_type_concept_id    = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
#     cohort_definition_syntax      = Column(Text)
#     subject_concept_id            = Column(Integer, ForeignKey('.'.join([omop_schema, 'concept', 'concept_id'])), nullable=False)
#     cohort_initiation_date        = Column(Date)
  
#     def __repr__(self):
#         return(
#           "<CohortDefinition(cohort_definition_id='%s', cohort_definition_name='%s')>" %(
#             self.cohort_definition_id, self.cohort_definition_name
#           )
#         )
