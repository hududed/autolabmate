import time

import extra_streamlit_components as stx
import streamlit as st
import supabase
from dotenv import load_dotenv

from config import settings
from db.database import supabase_client

load_dotenv()

# Store this as a module-level variable instead of creating on import
_cookie_manager = None


def get_cookie_manager():
    """Get a cookie manager instance"""
    global _cookie_manager
    if _cookie_manager is None:
        _cookie_manager = stx.CookieManager()
    return _cookie_manager


def initialize_session_state_basic():
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


def check_cookie_auth():
    """
    Check for authentication from cookies.
    Call this AFTER st.set_page_config().
    """
    if not st.session_state.authentication_status:
        try:
            cookie_manager = get_cookie_manager()
            auth_token = cookie_manager.get("auth_token")
            user_id = cookie_manager.get("user_id")

            if auth_token and user_id:
                # Verify token by trying to get user data
                response = supabase_client.auth.get_user(auth_token)
                if response and hasattr(response, "user") and response.user:
                    # Token is valid, restore the session
                    st.session_state.authentication_status = True
                    st.session_state.user_id = user_id
                    # Refresh the token to extend its validity
                    refresh_jwt_with_cookie(auth_token)
        except Exception as e:
            # Token invalid or expired, remove it
            clear_auth_cookies()
            print(f"Session restoration error: {str(e)}")


def initialize_session_state():
    """
    Full initialization including cookie checks.
    Call this AFTER st.set_page_config().
    """
    initialize_session_state_basic()
    check_cookie_auth()


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
    Redirects to login page if not.
    """
    if not st.session_state.get("authentication_status", False):
        st.warning("Please login to access this page.")
        st.switch_page("app.py")  # Redirect to login page


def save_auth_cookies(access_token, user_id):
    """
    Save authentication token and user ID to cookies

    Args:
        access_token: The JWT access token
        user_id: The user's ID
    """
    cookie_manager = get_cookie_manager()
    import datetime

    expiry = datetime.datetime.now() + datetime.timedelta(days=7)

    # Add unique keys to avoid the StreamlitDuplicateElementKey error
    cookie_manager.set(
        "auth_token", access_token, expires_at=expiry, key="auth_token_cookie"
    )

    time.sleep(0.1)

    cookie_manager.set("user_id", user_id, expires_at=expiry, key="user_id_cookie")


def clear_auth_cookies():
    """Clear authentication tokens from cookies"""
    try:
        cookie_manager = get_cookie_manager()

        # Check if cookies exist before trying to delete them
        cookies = cookie_manager.get_all()

        if "auth_token" in cookies:
            cookie_manager.delete("auth_token", key="delete_auth_token")

        time.sleep(0.1)

        if "user_id" in cookies:
            cookie_manager.delete("user_id", key="delete_user_id")

    except Exception as e:
        print(f"Error clearing cookies: {str(e)}")


def set_session_state(credentials):
    """
    Sets the streamlit state variables after user authentication.
    Returns:
        Nothing.
    """
    initialize_session_state()
    access_token, user_id = get_user_tokens(credentials)

    if access_token and user_id:
        st.session_state.authentication_status = True
        st.session_state.user_id = user_id
        save_auth_cookies(access_token, user_id)


def login(credentials):
    # Login to existing account
    response = supabase_client.auth.sign_in_with_password(credentials)
    return response


def get_user_tokens(credentials):
    """
    Gets user tokens by making a post request call.

    Args:
        credentials: User credentials for authentication.

    Returns:
        Tuple of (access_token, user_id)
    """
    response = login(credentials=credentials)
    try:
        session = response.model_dump()["session"]
        access_token = session.get("access_token")
        user = response.model_dump()["user"]
        user_id = user.get("id")
        return access_token, user_id
    except (KeyError, TypeError, AttributeError):
        return "", ""


def refresh_jwt_with_cookie(token=None):
    """
    Refresh the JWT token using the provided token or current session
    and update the cookie
    """
    new_token = refresh_jwt(token)
    if new_token and st.session_state.user_id:
        save_auth_cookies(new_token, st.session_state.user_id)
    return new_token


def refresh_jwt(token=None):
    """
    Refresh the JWT token using the refresh token

    Args:
        token: Optional token to use, otherwise uses the current session

    Returns:
        New JWT token if successful, None otherwise
    """
    global supabase_client
    try:
        # Get the current session
        if token:
            response = supabase_client.auth.get_user(token)
            session = response.session if hasattr(response, "session") else None
        else:
            session = supabase_client.auth.get_session()

        if session and hasattr(session, "refresh_token") and session.refresh_token:
            # Refresh the session using the refresh token
            new_session = supabase_client.auth.refresh_session(session.refresh_token)
            if new_session and hasattr(new_session, "session"):
                new_jwt = new_session.session.access_token
                supabase_url = settings.SUPABASE_URL
                supabase_key = settings.SUPABASE_KEY

                # Reinitialize the Supabase client with the new JWT
                supabase_client = supabase.create_client(supabase_url, supabase_key)
                return new_jwt
    except Exception as e:
        print(f"JWT refresh error: {str(e)}")
    return None


def logout():
    """Logout the user by clearing session state and cookies"""
    # Clear session state
    st.session_state.authentication_status = False
    st.session_state.user_id = None

    # Clear cookies
    clear_auth_cookies()

    # Sign out from Supabase
    try:
        supabase_client.auth.sign_out()
    except Exception as e:
        print(f"Logout error: {str(e)}")
