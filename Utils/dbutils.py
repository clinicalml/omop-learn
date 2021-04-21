import sys
sys.path.append('..')

import pandas as pd
import sqlalchemy
import io
import time
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

import config
        
class Database(object):  # noqa

    def __init__(self, config_path, schema_name, connect_args, cdm_schema_name="cdm", echo=False): 
        
        self.engine = sqlalchemy.create_engine(
            config_path,
            echo=echo,
            connect_args=connect_args
        )
        self.meta = sqlalchemy.MetaData(
            bind=self.engine,
            reflect=True
        )
        self.cdmMeta = sqlalchemy.MetaData(
            bind=self.engine,
            reflect=True,
            schema=cdm_schema_name
        )
        self.selfMeta = sqlalchemy.MetaData(
            bind=self.engine,
            reflect=True,
            schema=schema_name
        )
        
    @contextmanager
    def _session_scope(self):

        Session = sessionmaker(bind=self.engine)
        session = Session()
        
        try:
            yield session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()
        
    def build_table(self, table_name, sql):
        '''
        (Re)builds table_name using sql.
        Note that the table will be deleted if it already exists.
        
        Args:
            table_name: The name of the table to be rebuilt
            sql: A SQL command string to create the table table_name
        Returns:
            None
        '''
        with self._session_scope() as session:

            drop_sql = "drop table if exists {}".format(table_name)
            session.execute(sqlalchemy.text(drop_sql))
            
            session.execute(sqlalchemy.text(sql))
            session.commit()
    
    def build_table_from_sql_file(
        self, sql_path, table_name, params,
        schema_name, replace=False
    ):

        if replace or table_name not in self.get_all_tables(
                schema=schema_name
            ).values:
                t = time.time()
                if replace:
                    print('Regenerating Table (replace=True)')
                else:
                    print(
                        'Table not found in schema {}, regenerating'.format(
                            schema_name
                        )
                    )
                
                with open(sql_path, 'r') as f:
                    cohort_generation_sql_raw = f.read()
                if params is not None:
                    cohort_generation_sql = cohort_generation_sql_raw.format(
                        **params
                    )
                else:
                    cohort_generation_sql = cohort_generation_sql_raw
                    
                self.build_table('{}.{}'.format(
                    schema_name,
                    table_name
                ), cohort_generation_sql)
                
                print('Regenerated Cohort in {} seconds'.format(
                    time.time() - t
                ))
        else:
            print('Table already exists, set replace=True to rebuild')
    
    def fast_query(self, sql):
        '''
        Run sql and dump results to a Pandas Dataframe.
        Runs faster by using an intermediate string buffer.
        Args:
            sql: A SQL command to fetch desired results
        Returns:
            pandas.DataFrame with query results
        '''
        copy_sql = """
            copy 
                ({query})
            to 
                stdout 
            with 
                csv {head}
        """.format(
            query=sql,
            head="HEADER"
        )
        conn = self.engine.raw_connection()
        cur = conn.cursor()
        store = io.StringIO()
        cur.copy_expert(copy_sql, store)
        store.seek(0)
        df = pd.read_csv(store, engine='python')
        return df

    def query(self, sql):
        '''
        Run sql and dump results to a Pandas Dataframe.
        Args:
            sql: A SQL command to fetch desired results
        Returns:
            pandas.DataFrame with query results
        '''
        return pd.read_sql(sql, self.engine)
        
    def execute(self, *sqls):
        '''
        Run each command in sqls.
        Args:
            sqls: Any number of SQL command strings
        Returns:
            None
        '''
        with self._session_scope() as session:
            for sql in sqls:
                session.execute(sql)
            session.commit()
            print('Executed {} SQLs'.format(
                len(sqls)
            ))
            
    def get_all_tables(
        self,
        schema=config.OMOP_CDM_SCHEMA
    ):
        '''
        Return a pandas.Series of all tables in schema
        Args:
            schema: The schema to look for tables in
        Returns:
            pandas.Series of all tables in schema
        '''
        sql = """
            select table_name
            from information_schema.tables
            {};
        """.format(
            '' if schema is None
            else "where table_schema = '{}'".format(schema)
        )
        return self.query(sql)['table_name']
            

            
                    
