import streamlit as st
from streamlit_extras.switch_page_button import (
    switch_page,
)
from components.authenticate import supabase_client
from time import sleep
from session_state import initialize_session_state

initialize_session_state()


# Logout page
def logout():
    if not st.session_state.authentication_status:
        st.info("Please Login from the Home page and try again.")
        st.stop()
    st.session_state.authentication_status = False  # set the logged_in state to False
    res = supabase_client.auth.sign_out()
    if res:
        st.error(f"Error logging out: {res}")
    else:
        st.success("Logged out successfully")
        sleep(5)
        switch_page("")  # switch back to the login page


def main():
    if st.session_state.authentication_status:
        st.title("Logout")
        logout()


if __name__ == "__main__":
    main()
