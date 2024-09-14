import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
from pages.generate import main


# TODO: new test with the refactored generator
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


def test_main(mock_streamlit):
    with patch("pages.generate.CSVGenerator") as mock_csv_generator:
        mock_generator_instance = mock_csv_generator.return_value
        mock_generator_instance.generate = MagicMock()

        # Ensure check_authentication returns True
        with patch("components.authenticate.check_authentication", return_value=True):
            main()

        # Assert that the Streamlit title was set
        st.title.assert_called_once_with("CSV Generator")

        # Assert that the CSVGenerator instance was created and its generate method was called
        mock_csv_generator.assert_called_once()
        mock_generator_instance.generate.assert_called_once()

        # Assert that the authentication functions were called
        assert "authentication_status" in st.session_state
        assert "user_id" in st.session_state
        assert "retry_count" in st.session_state
        assert "retry_delay" in st.session_state
