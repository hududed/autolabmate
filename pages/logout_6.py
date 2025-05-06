import time

import streamlit as st

from db.database import supabase_client
from dependencies.authentication import (
    clear_auth_cookies,
    initialize_session_state,
)
from dependencies.navigation import authenticate_and_show_nav

st.set_page_config(page_title="Upload | Autolabmate", page_icon="⬆️")

authenticate_and_show_nav()
initialize_session_state()


def logout():
    """Log out the user"""
    try:
        # Clear session state
        st.session_state.authentication_status = (
            False  # set the logged_in state to False
        )
        st.session_state.user_id = None  # clear the user ID

        # Clear cookies
        clear_auth_cookies()  # clear the authentication cookies

        # Sign out from Supabase
        _ = supabase_client.auth.sign_out()

        # Show success message and redirect
        st.success("You have been logged out successfully.")
        time.sleep(1)  # Brief pause to show the message
        st.switch_page("app.py")
    except Exception as e:
        st.error(f"Error during logout: {str(e)}")
        # Still try to navigate back even if there's an error
        st.switch_page("app.py")


def main():
    if st.session_state.authentication_status:
        st.title("Logout")
        logout()


if __name__ == "__main__":
    main()
