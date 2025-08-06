from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base,sessionmaker

engine=create_engine('postgresql://postgres.obitimvwpybuhnhscdcr:Pranav12412770@aws-0-ap-south-1.pooler.supabase.com:5432/postgres')

Base=declarative_base()


Session=sessionmaker()
