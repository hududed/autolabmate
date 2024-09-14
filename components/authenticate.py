import os
import supabase
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect
import streamlit as st

load_dotenv()

# Load Supabase credentials from .env file
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)
PG_PASS = os.getenv("PG_PASS")
DATABASE_URL = f"postgresql://postgres.zugnayzgayyoveqcmtcd:{PG_PASS}@aws-0-us-east-1.pooler.supabase.com:5432/postgres"
engine = create_engine(DATABASE_URL)
inspector = inspect(engine)


def initialize_session_state():
    """
    Initialise Streamlit state variables.

    Returns:
        Nothing.
    """

    if "authentication_status" not in st.session_state:
        st.session_state.authentication_status = False
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "retry_count" not in st.session_state:
        st.session_state.retry_count = 0
    if "retry_delay" not in st.session_state:
        st.session_state.retry_delay = 1  # Initial delay between retries in seconds


def check_authentication():
    """
    Checks if the user is authenticated.

    Returns:
        None
    """
    if not st.session_state.get("authentication_status", False):
        st.info("Please Login from the Home page and try again.")
        st.stop()


def set_session_state(credentials):
    """
    Sets the streamlit state variables after user authentication.
    Returns:
        Nothing.
    """
    initialize_session_state()
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


def refresh_jwt():
    global supabase_client
    # Get the current session
    session = supabase_client.auth.get_session()

    if session:
        refresh_token = session["refresh_token"]
        # Refresh the session using the refresh token
        new_session = supabase_client.auth.refresh_session(refresh_token)
        if new_session:
            new_jwt = new_session["access_token"]
            # Reinitialize the Supabase client with the new JWT
            supabase_client = supabase.create_client(SUPABASE_URL, new_jwt)
            return new_jwt
    return None
