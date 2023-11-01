import streamlit as st
from components.authenticate import engine, supabase_client, set_st_state_vars
import uuid
from streamlit_extras.switch_page_button import switch_page
from streamlit.source_util import _on_pages_changed, get_pages
from pathlib import Path
import json
from st_pages import show_pages, Page, hide_pages


# DEFAULT_PAGE = "landing.py"
SECOND_PAGE_NAME = "upload"

st.title("Welcome to Autolabmate!")

# Login form
def login():
    hide_pages(["upload", "dashboard"])
    st.header("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        # Retrieve user details from the database
        user = supabase_client.auth.sign_in_with_password(credentials={'email': email, 'password': password})

        if user:
            
            st.success("Logged In Sucessfully {}".format(email))
            
            hide_pages(["home"])
            show_pages(
                [
                    Page("pages/upload.py", "upload", icon="⬆️"),
                    Page("pages/dashboard.py", "dashboard", icon="📈"),
                    Page("pages/propose.py", "propose", icon="🤖")
                ]
            )
            switch_page("Upload")   # switch to second page
            
        else:
            st.error("Invalid email or password")

# Sign-up form
def signup():
    st.header("Sign Up")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Sign Up"):
        # Insert user details into the database
        user_data = supabase_client.auth.sign_up({
        'email': email,
        'password': password
    })
        if user_data:
            st.success("Successfully registered!")
        else:
            st.error("Registration failed!")

# Run the Streamlit app
def main():
    show_pages([Page("app.py", "home")])
    # Display the login or sign-up form based on user selection
    form_choice = st.selectbox("Select an option:", ("Login", "Sign Up"))

    if form_choice == "Login":
        login()
    elif form_choice == "Sign Up":
        signup()


if __name__ == '__main__':
    main()