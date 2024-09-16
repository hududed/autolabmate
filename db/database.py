from config import Environment, settings
from .connectors import connect_to_db, connect_to_local_db

engine, supabase_client = (
    connect_to_db()
    if settings.ENVIRONMENT == Environment.DEVELOPMENT.value
    else connect_to_local_db()
)


def get_supabase_client():
    try:
        yield supabase_client
    finally:
        pass
