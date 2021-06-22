from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

class SessionMaker(object):
    def __init__(self, engine):
        self.maker   = sessionmaker(bind=engine)
        self.session = self.maker()
    
    @contextmanager
    def session_manager(self):

        try:
            yield self.session
            self.session.commit()
        except:
            self.session.rollback()
            raise
            
    def close(self):
        self.session.close()