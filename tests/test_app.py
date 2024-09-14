import pytest
from unittest.mock import MagicMock
import streamlit as st
from app import login, signup, logout, main
from components.authenticate import supabase_client, initialize_session_state


@pytest.fixture
def mock_streamlit(mocker):
    mocker.patch.object(st, "header")
    mocker.patch.object(st, "text_input")
    mocker.patch.object(st, "button")
    mocker.patch.object(st, "success")
    mocker.patch.object(st, "error")
    mocker.patch.object(st, "warning")
    mocker.patch.object(st, "selectbox")
    mocker.patch.object(st, "title")
    mocker.patch.object(st, "info")
    mocker.patch.dict(
        st.session_state,
        {
            "authentication_status": False,
            "retry_count": 0,
            "retry_delay": 1,
            "user_id": None,
        },
    )
    mocker.patch(
        "streamlit.runtime.scriptrunner.script_runner.ScriptRunner", autospec=True
    )
    mocker.patch(
        "streamlit.runtime.scriptrunner.script_runner.ScriptRunner._get_script_run_ctx",
        return_value=True,
    )
    mocker.patch("streamlit.runtime.runtime.Runtime.instance", return_value=MagicMock())


@pytest.fixture
def mock_supabase(mocker):
    mocker.patch.object(supabase_client.auth, "sign_in_with_password")
    mocker.patch.object(supabase_client.auth, "sign_up")
    mocker.patch.object(supabase_client.auth, "sign_out")


def test_login_success(mock_streamlit, mock_supabase):
    st.text_input.side_effect = ["test@example.com", "password"]
    st.button.return_value = True
    supabase_client.auth.sign_in_with_password.return_value = MagicMock(
        user=MagicMock(id="user_id")
    )

    try:
        login()
    except st.runtime.scriptrunner.script_runner.RerunException:
        pass

    st.success.assert_called_once_with("Logged In Successfully test@example.com")
    assert st.session_state["authentication_status"] is True
    assert st.session_state["user_id"] == "user_id"


def test_login_failure(mock_streamlit, mock_supabase):
    st.text_input.side_effect = ["test@example.com", "password"]
    st.button.return_value = True
    supabase_client.auth.sign_in_with_password.return_value = None

    login()

    st.error.assert_called_once_with("Invalid email or password")
    assert st.session_state["authentication_status"] is False


def test_signup_success(mock_streamlit, mock_supabase):
    st.text_input.side_effect = ["test@example.com", "password"]
    st.button.return_value = True
    supabase_client.auth.sign_up.return_value = MagicMock()

    signup()

    st.success.assert_called_once_with("Successfully registered!")
    assert st.session_state["retry_count"] == 0


def test_signup_failure(mock_streamlit, mock_supabase):
    st.text_input.side_effect = ["test@example.com", "password"]
    st.button.return_value = True
    supabase_client.auth.sign_up.side_effect = Exception("Registration failed!")

    signup()

    st.error.assert_called_once_with("An error occurred: Registration failed!")


def test_signup_too_many_requests(mock_streamlit, mock_supabase):
    st.text_input.side_effect = ["test@example.com", "password"]
    st.button.return_value = True
    st.session_state["retry_count"] = 0
    st.session_state["retry_delay"] = 1
    supabase_client.auth.sign_up.side_effect = Exception("429 Too Many Requests")

    signup()

    st.warning.assert_called_once_with(
        "Too many requests. Please wait 1 seconds before trying again."
    )
    assert st.session_state["retry_count"] == 1
    assert st.session_state["retry_delay"] == 2


def test_logout_success(mock_streamlit, mock_supabase):
    st.session_state["authentication_status"] = True
    supabase_client.auth.sign_out.return_value = None

    try:
        logout()
    except st.runtime.scriptrunner.script_runner.RerunException:
        pass

    st.success.assert_called_once_with("Logged out successfully")
    assert st.session_state["authentication_status"] is False


def test_logout_failure(mock_streamlit, mock_supabase):
    st.session_state["authentication_status"] = True
    supabase_client.auth.sign_out.return_value = "Error"

    try:
        logout()
    except st.runtime.scriptrunner.script_runner.RerunException:
        pass

    st.error.assert_called_once_with("Error logging out: Error")


def test_main_login(mock_streamlit, mocker):
    mock_login = mocker.patch("app.login")
    mock_signup = mocker.patch("app.signup")
    mocker.patch("app.create_experiments_table")
    mocker.patch("app.enable_rls")
    mocker.patch("app.create_policy")
    st.selectbox.return_value = "Login"

    main()

    mock_login.assert_called_once()
    mock_signup.assert_not_called()


def test_main_signup(mock_streamlit, mocker):
    mock_login = mocker.patch("app.login")
    mock_signup = mocker.patch("app.signup")
    mocker.patch("app.create_experiments_table")
    mocker.patch("app.enable_rls")
    mocker.patch("app.create_policy")
    st.selectbox.return_value = "Sign Up"

    main()

    mock_signup.assert_called_once()
    mock_login.assert_not_called()
