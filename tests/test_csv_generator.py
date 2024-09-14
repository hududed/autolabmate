import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
from auto_csv_generator.csv_generator import CSVGenerator


@pytest.fixture
def mock_streamlit(mocker):
    mocker.patch.object(st, "selectbox")
    mocker.patch.object(st, "number_input")
    mocker.patch.object(st, "text_input")
    mocker.patch.object(st, "download_button")
    mocker.patch.object(st, "info")
    mocker.patch.object(st, "stop")
    mocker.patch.dict(st.session_state, {}, clear=True)


def test_get_randomization_type(mock_streamlit):
    generator = CSVGenerator()
    generator.get_randomization_type()
    st.selectbox.assert_called_once_with(
        "Select randomization type:", ["Random", "Latin Hypercube", "Sobol"]
    )


def test_get_precision(mock_streamlit):
    generator = CSVGenerator()
    generator.param_types = ["Float"]
    generator.get_precision()
    st.number_input.assert_called_once_with(
        "Enter the precision for float values:", value=1, min_value=0, max_value=3
    )


def test_get_input_values(mock_streamlit):
    generator = CSVGenerator()
    generator.get_input_values()
    st.number_input.assert_any_call("Number of parameters:", value=3)
    st.number_input.assert_any_call("Number of random lines:", value=10)


def test_get_parameter_info(mock_streamlit):
    generator = CSVGenerator()
    generator.series = 2
    generator.get_parameter_info()
    assert len(generator.param_names) == 2
    assert len(generator.param_types) == 2
    st.text_input.assert_any_call("Parameter 1 name:", "param1")
    st.selectbox.assert_any_call(
        "Parameter 1 type:", ["Integer", "Float", "Categorical"]
    )


def test_get_data_header(mock_streamlit):
    generator = CSVGenerator()
    generator.param_names = ["param1", "param2"]
    generator.get_data_header()
    assert generator.data_header == ["param1", "param2", "output1", "output2"]


def test_get_optimization_type(mock_streamlit):
    generator = CSVGenerator()
    generator.get_optimization_type()
    st.selectbox.assert_called_once_with(
        "Select optimization type:", ["Single", "Multi"]
    )


def test_get_parameter_ranges(mock_streamlit):
    generator = CSVGenerator()
    generator.param_names = ["param1", "param2"]
    generator.param_types = ["Integer", "Float"]

    # Set return values for number_input mocks
    st.number_input.side_effect = [0, 100, 1, 0.0, 100.0, 0.1]

    generator.get_parameter_ranges()
    st.number_input.assert_any_call("Minimum value for param1:", value=0)
    st.number_input.assert_any_call("Maximum value for param1:", value=100)
    st.number_input.assert_any_call("Interval for param1:", value=1)
    st.number_input.assert_any_call("Minimum value for param2:", value=0.0)
    st.number_input.assert_any_call("Maximum value for param2:", value=100.0)
    st.number_input.assert_any_call("Interval for param2:", value=0.1)
    assert generator.param_ranges == [(0, 100), (0.0, 100.0)]
    assert generator.param_intervals == [1, 0.1]


def test_generate_random_values(mock_streamlit):
    generator = CSVGenerator()
    generator.series = 2
    generator.nr_random_lines = 2
    generator.param_types = ["Integer", "Float"]
    generator.param_ranges = [(0, 10), (0.0, 1.0)]
    generator.param_intervals = [2, 0.2]
    generator.precision = 2
    generator.generate_random_values()
    assert len(generator.param_values) == 2
    assert len(generator.param_values[0]) == 2
    for values in generator.param_values:
        assert values[0] % 2 == 0  # Check integer interval
        assert round(values[1] % 0.2, 2) == 0.0  # Check float interval
        assert 0 <= values[0] <= 10  # Check integer range
        assert 0.0 <= values[1] <= 1.0  # Check float range


def test_write_csv_file(mock_streamlit):
    generator = CSVGenerator()
    generator.param_values = [[1, 2], [3, 4]]
    generator.data_header = ["param1", "param2"]
    generator.write_csv_file()
    with open("data.csv", "r") as f:
        content = f.read()
    assert "param1,param2\n1,2\n3,4\n" in content


def test_download_csv_file(mock_streamlit):
    generator = CSVGenerator()
    generator.download_csv_file()
    st.download_button.assert_called_once_with(
        label="Download CSV",
        data=open("data.csv", "rb").read(),
        file_name="data.csv",
        mime="text/csv",
    )


def test_generate(mock_streamlit):
    generator = CSVGenerator()
    with patch.object(
        generator, "get_randomization_type"
    ) as mock_get_randomization_type, patch.object(
        generator, "get_input_values"
    ) as mock_get_input_values, patch.object(
        generator, "get_parameter_info"
    ) as mock_get_parameter_info, patch.object(
        generator, "get_data_header"
    ) as mock_get_data_header, patch.object(
        generator, "get_optimization_type"
    ) as mock_get_optimization_type, patch.object(
        generator, "get_parameter_ranges"
    ) as mock_get_parameter_ranges, patch.object(
        generator, "get_precision"
    ) as mock_get_precision, patch.object(
        generator, "generate_parameter_values"
    ) as mock_generate_parameter_values, patch.object(
        generator, "write_csv_file"
    ) as mock_write_csv_file, patch.object(
        generator, "download_csv_file"
    ) as mock_download_csv_file:
        generator.generate()

        mock_get_randomization_type.assert_called_once()
        mock_get_input_values.assert_called_once()
        mock_get_parameter_info.assert_called_once()
        mock_get_data_header.assert_called_once()
        mock_get_optimization_type.assert_called_once()
        mock_get_parameter_ranges.assert_called_once()
        mock_get_precision.assert_called_once()
        mock_generate_parameter_values.assert_called_once()
        mock_write_csv_file.assert_called_once()
        mock_download_csv_file.assert_called_once()
