import time

import streamlit as st

from db.crud.table import (
    create_experiments_table,
    create_policy_for_table,
    enable_rls_for_table,
)
from db.database import supabase_client
from dependencies.authentication import (
    check_cookie_auth,
    initialize_session_state_basic,
    save_auth_cookies,
)
from dependencies.navigation import show_navigation_menu

st.set_page_config(page_title="Autolabmate", page_icon="👨‍🔬")
# Initialize session state and check cookies
initialize_session_state_basic()
check_cookie_auth()

# Configure page with title and icon
st.title("Welcome to Autolabmate!!")

# Show navigation if authenticated
if st.session_state.get("authentication_status", False):
    show_navigation_menu()


def login():
    st.header("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        # Retrieve user details from the database
        response = supabase_client.auth.sign_in_with_password(
            credentials={"email": email, "password": password}
        )

        if response and hasattr(response, "user") and response.user:
            # Extract session and user data
            session = response.session
            user = response.user

            # Update session state
            st.session_state.authentication_status = True
            st.session_state.user_id = user.id

            # Save token for persistent session
            save_auth_cookies(session.access_token, user.id)

            st.success(f"Logged In Successfully {email}")

            # Direct navigation - no rerun, no flags
            st.switch_page("pages/upload_2.py")
        else:
            st.error("Invalid email or password")
            st.session_state.authentication_status = False


def signup():
    st.header("Sign Up")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Sign Up"):
        try:
            # Attempt to sign up the user
            user_data = supabase_client.auth.sign_up(
                {"email": email, "password": password}
            )
            if user_data:
                st.success("Successfully registered!")
                st.session_state.retry_count = 0  # Reset retry count on success
                return
        except Exception as e:
            if "429" in str(e):
                if st.session_state.retry_count < 3:  # Maximum number of retries
                    st.warning(
                        f"Too many requests. Please wait {st.session_state.retry_delay} seconds before trying again."
                    )
                    st.session_state.retry_count += 1
                    st.session_state.retry_delay *= 2  # Exponential backoff
                else:
                    st.error(
                        "Failed to register after multiple attempts. Please try again later."
                    )
            else:
                st.error(f"An error occurred: {e}")


# Run the Streamlit app
def main():
    create_experiments_table()
    enable_rls_for_table("experiments")
    create_policy_for_table("experiments")

    # First, always check for "do_redirect" flag
    if st.session_state.get("do_redirect", False):
        st.session_state.pop("do_redirect", None)
        time.sleep(0.1)  # Small delay
        st.switch_page("pages/upload_2.py")
        return  # Exit early

    # Handle navigation after login
    if st.session_state.get("authentication_status", False):
        st.switch_page("pages/upload_2.py")
    else:
        # Display the login or sign-up form based on user selection
        form_choice = st.selectbox("Select an option:", ("Login", "Sign Up"))

        if form_choice == "Login":
            login()
        elif form_choice == "Sign Up":
            signup()


if __name__ == "__main__":
    main()
