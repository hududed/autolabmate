import os
import supabase
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect
load_dotenv()

# Load Supabase credentials from .env file
SUPA_URL = os.getenv('SUPABASE_URL')
SUPA_KEY = os.getenv('SUPABASE_KEY')

# Initialize Supabase client
supabase_client = supabase.create_client(SUPA_URL, SUPA_KEY)
PG_PASS = os.getenv('PG_PASS')
DATABASE_URL = f'postgresql://postgres:{PG_PASS}@db.zugnayzgayyoveqcmtcd.supabase.co:5432/postgres'
engine = create_engine(DATABASE_URL)