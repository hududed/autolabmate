import streamlit as st

from dependencies.authentication import check_authentication


# Create a function to show navigation menu
def show_navigation_menu():
    with st.sidebar:
        st.page_link("pages/generate_1.py", label="Generate", icon="📝")
        st.page_link("pages/upload_2.py", label="Upload", icon="⬆️")
        st.page_link("pages/dashboard_3.py", label="Dashboard", icon="📈")
        st.page_link("pages/propose_4.py", label="Propose", icon="🤖")
        st.page_link("pages/update_5.py", label="Update", icon="🔄")
        st.page_link("pages/logout_6.py", label="Logout", icon="🚪")


def authenticate_and_show_nav():
    """Check authentication and show navigation if authenticated"""
    check_authentication()
    show_navigation_menu()
