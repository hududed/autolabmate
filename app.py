import streamlit as st
from components.authenticate import supabase_client
from st_pages import show_pages, Page
from streamlit_extras.switch_page_button import switch_page
from time import sleep
import os
import subprocess
from utils import (
    enable_rls,
    create_policy,
    create_experiments_table,
)

st.title("Welcome to Autolabmate!")


# Login form
def login():
    st.header("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        # Retrieve user details from the database
        response = supabase_client.auth.sign_in_with_password(
            credentials={"email": email, "password": password}
        )

        if response:
            st.success("Logged In Sucessfully {}".format(email))
            st.session_state.authentication_status = True
            st.session_state.user_id = response.user.id

            show_pages(
                [
                    Page("pages/generate.py", "generate", icon="📝"),
                    Page("pages/upload.py", "upload", icon="⬆️"),
                    Page("pages/dashboard.py", "dashboard", icon="📈"),
                    Page("pages/propose.py", "propose", icon="🤖"),
                    Page("pages/update.py", "update", icon="🔄"),
                    Page("pages/logout.py", "logout", icon="🚪"),
                    Page("app.py", ""),
                ]
            )

            switch_page("Upload")  # switch to second page

        else:
            st.error("Invalid email or password")
            st.session_state.authentication_status = False


# Sign-up form
def signup():
    st.header("Sign Up")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Sign Up"):
        # Insert user details into the database
        user_data = supabase_client.auth.sign_up({"email": email, "password": password})
        if user_data:
            st.success("Successfully registered!")
        else:
            st.error("Registration failed!")


# Logout page
def logout():
    st.session_state.authentication_status = False  # set the logged_in state to False
    res = supabase_client.auth.sign_out()
    if res:
        st.error(f"Error logging out: {res}")
    else:
        st.success("Logged out successfully")
        sleep(5)
        switch_page("")  # switch back to the login page


# Run the Streamlit app
def main():
    if "authentication_status" not in st.session_state:
        st.session_state.authentication_status = False

    show_pages([Page("app.py", "home")])

    r_home = os.environ.get("R_HOME", "Undefined")
    if r_home is not None:
        # List all subdirectories in R_HOME
        for root, dirs, files in os.walk(r_home):
            for dir in dirs:
                print(os.path.join(root, dir))

    # TODO: add subprocess to run custom R script install.R to install R packages mlr3mbo, mlr3, mlr3learners, data.table, tibble, bbotk, R.utils

    try:
        process = subprocess.Popen(
            ["Rscript", "install.R"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(f"Error installing R packages: {stderr}")
        else:
            print("R packages installed successfully!")
    except Exception as e:
        print(f"Error: {e}")

    # process2 = subprocess.Popen(["Rscript", "plot.R"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # result2 = process2.communicate()
    create_experiments_table()
    enable_rls("experiments")
    create_policy("experiments")

    # Display the login or sign-up form based on user selection
    form_choice = st.selectbox("Select an option:", ("Login", "Sign Up"))

    if form_choice == "Login":
        login()
    elif form_choice == "Sign Up":
        signup()


if __name__ == "__main__":
    main()
