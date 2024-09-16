import supabase
from dotenv import load_dotenv
from config import settings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

load_dotenv()


def connect_to_db():
    """
    Connects to the database based on the environment.
    """
    supabase_url = settings.SUPABASE_URL
    supabase_key = settings.SUPABASE_KEY
    db_pass = settings.DB_PASSWORD
    db_user = settings.DB_USER
    db_name = settings.DB_NAME
    db_host = settings.DB_HOST
    db_port = settings.DB_PORT

    supabase_client = supabase.create_client(supabase_url, supabase_key)

    database_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

    engine = create_engine(
        url=database_url,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )

    return engine, supabase_client


# PLACEHOLDER, NOT USED
def connect_to_local_db():  # pragma: no cover
    """
    Initializes a connection pool for a local Postgres instance.
    """
    SQLALCHEMY_DATABASE_URL = f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}/{settings.DB_NAME}"
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, SessionLocal
