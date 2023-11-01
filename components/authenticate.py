import os
import supabase
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect
import streamlit as st
load_dotenv()

# Load Supabase credentials from .env file
SUPA_URL = os.getenv('SUPABASE_URL')
SUPA_KEY = os.getenv('SUPABASE_KEY')

# Initialize Supabase client
supabase_client = supabase.create_client(SUPA_URL, SUPA_KEY)
PG_PASS = os.getenv('PG_PASS')
DATABASE_URL = f'postgresql://postgres:{PG_PASS}@db.zugnayzgayyoveqcmtcd.supabase.co:5432/postgres'
engine = create_engine(DATABASE_URL)

def initialise_st_state_vars():
    """
    Initialise Streamlit state variables.

    Returns:
        Nothing.
    """

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

def set_st_state_vars(credentials):
    """
    Sets the streamlit state variables after user authentication.
    Returns:
        Nothing.
    """
    initialise_st_state_vars()
    access_token = get_user_tokens(credentials)

    if access_token != "":
        st.session_state["authenticated"] = True


def login(credentials):
    # Login to existing account
    response = supabase_client.auth.sign_in_with_password(credentials)
    return response

def get_user_tokens(credentials):
    """
    Gets user tokens by making a post request call.

    Args:
        auth_code: Authorization code from supabase server.

    Returns:
        {
        'access_token': access token from supabase client if user is successfully authenticated.
        }

    """

    response = login(credentials=credentials)
    try:
        session = response.model_dump()["session"]
        access_token = session.get("access_token")
    except (KeyError, TypeError):
        access_token = ""
    return access_token