from database import Base
from sqlalchemy import Column,Integer,String, DateTime
import datetime

class Data(Base):
    __tablename__='qa_logs'
    id=Column(Integer,primary_key=True)
    user_query=Column(String(50000))
    ai_response=Column(String(500000))
    document_url = Column(String(200))
    timestamp=Column(DateTime, nullable=False, default=datetime.datetime.now)
    
    def __repr__(self):
        return f"<Data query={self.user_query}>"