import streamlit as st
from components.authenticate import supabase_client
from st_pages import show_pages, Page
from streamlit_extras.switch_page_button import switch_page
from time import sleep
from utils import (
    enable_rls,
    create_policy,
    create_experiments_table,
)

st.title("Welcome to Autolabmate!!")


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


def signup():
    if "retry_count" not in st.session_state:
        st.session_state.retry_count = 0
    if "retry_delay" not in st.session_state:
        st.session_state.retry_delay = 1  # Initial delay between retries in seconds

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
            else:
                st.error("Registration failed!")
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


# # Sign-up form
# def signup():
#     st.header("Sign Up")
#     email = st.text_input("Email")
#     password = st.text_input("Password", type="password")

#     if st.button("Sign Up"):
#         # Insert user details into the database
#         user_data = supabase_client.auth.sign_up({"email": email, "password": password})
#         print(user_data)
#         if user_data:
#             st.success("Successfully registered!")
#         else:
#             st.error("Registration failed!")


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

    # TODO: need to move to other cloud provider as streamlit cloud as r2u / r-cran-mlr3 is not supported

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
