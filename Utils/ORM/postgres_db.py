from sqlalchemy import create_engine
from Utils.ORM.session_maker import SessionMaker

class PostgresDatabase:
    def __init__(self, username, password, database_name = None, schema_name = None):
        self.engine = create_engine(
            'postgresql://{username}:{password}@{database_name}'.format(
                username = username,
                password = password,
                database_name = database_name
            )
        )
        self.session = SessionMaker(self.engine)
        self.schema_name = schema_name
        
        # Create the user schema if it doesn't exist
        if self.schema_name is not None:
            with self.session.session_manager() as session:
                try:
                    session.execute('create schema if not exists {}'.format(self.schema_name))
                except RuntimeError as err: 
                    print("Runtime error: {0}".format(err))
    
    def drop_schema(self):
        if self.schema_name is not None:
            with self.session.session_manager() as session:
                try:
                    session.execute('drop schema if exists {}'.format(self.schema_name))
                except:
                    pass