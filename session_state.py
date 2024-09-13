# session_state.py
import streamlit as st


def initialize_session_state():
    if "authentication_status" not in st.session_state:
        st.session_state.authentication_status = False
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "retry_count" not in st.session_state:
        st.session_state.retry_count = 0
    if "retry_delay" not in st.session_state:
        st.session_state.retry_delay = 1  # Initial delay between retries in seconds
