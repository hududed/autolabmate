import streamlit as st
import supabase
from dotenv import load_dotenv

from config import settings
from db.database import supabase_client

load_dotenv()


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
    if "table_name" not in st.session_state:
        st.session_state.table_name = ""
    if "update_clicked" not in st.session_state:
        st.session_state.update_clicked = False
    if "propose_page_loaded" not in st.session_state:
        st.session_state.propose_page_loaded = False
    if "update_page_loaded" not in st.session_state:
        st.session_state.update_page_loaded = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "df_no_preds" not in st.session_state:
        st.session_state.df_no_preds = None
    if "zip_buffer" not in st.session_state:
        st.session_state.zip_buffer = None
    if "expander_what_in_file" not in st.session_state:
        st.session_state.expander_what_in_file = None
    if "expander_usage_examples" not in st.session_state:
        st.session_state.expander_usage_examples = None


def clear_session_state(keys):
    for key in keys:
        if key in st.session_state:
            if key == "messages":
                st.session_state[key] = []
            elif key == "df_no_preds" or key == "zip_buffer":
                st.session_state[key] = None
            elif key == "expander_what_in_file":
                st.session_state[key] = False
            elif key == "expander_usage_examples":
                st.session_state[key] = False
            elif key == "update_clicked":
                st.session_state[key] = False
            else:
                st.session_state[key] = None


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
            supabase_url = settings.SUPABASE_URL
            # Reinitialize the Supabase client with the new JWT
            supabase_client = supabase.create_client(supabase_url, new_jwt)
            return new_jwt
    return None
