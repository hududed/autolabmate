import streamlit as st
from components.authenticate import initialize_session_state, check_authentication
import pytest


@pytest.fixture
def mock_streamlit(mocker):
    mocker.patch.object(st, "title")
    mocker.patch.object(st, "text_input")
    mocker.patch.object(st, "button")
    mocker.patch.object(st, "success")
    mocker.patch.object(st, "error")
    mocker.patch.object(st, "warning")
    mocker.patch.object(st, "selectbox")
    mocker.patch.object(st, "info")
    mocker.patch.object(st, "stop")
    mocker.patch.dict(st.session_state, {}, clear=True)


def test_initialize_session_state(mock_streamlit):
    # Run the initialize_session_state function
    initialize_session_state()

    # Assert that the session state variables are initialized correctly
    assert st.session_state["authentication_status"] is False
    assert st.session_state["user_id"] is None
    assert st.session_state["retry_count"] == 0
    assert st.session_state["retry_delay"] == 1


def test_check_authentication_not_authenticated(mock_streamlit):
    # Test when the user is not authenticated
    check_authentication()
    st.info.assert_called_once_with("Please Login from the Home page and try again.")
    st.stop.assert_called_once()


def test_check_authentication_authenticated(mock_streamlit):
    # Test when the user is authenticated
    st.session_state["authentication_status"] = True
    check_authentication()
    st.info.assert_not_called()
    st.stop.assert_not_called()
