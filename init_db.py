from database import engine,Base
from models import Data


Base.metadata.create_all(bind=engine)