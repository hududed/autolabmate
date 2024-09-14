import pytest
from unittest.mock import Mock, patch, call
from auto_csv_generator.csv_generator import CSVGenerator
from auto_csv_generator.value_generator import (
    generate_random_values,
    generate_lhs_values,
    generate_sobol_values,
)


@pytest.fixture
def csv_generator():
    param_info_func = Mock(return_value=(["param1", "param2"], ["Integer", "Float"]))
    param_ranges_func = Mock(return_value=([(0, 10), (0.0, 1.0)], [1, 0.1]))
    value_generator_func = Mock(
        return_value=[[1, 0.1], [2, 0.2], [3, 0.3], [4, 0.4], [5, 0.5]]
    )
    csv_writer_func = Mock()
    csv_downloader_func = Mock()

    generator = CSVGenerator(
        param_info_func=param_info_func,
        param_ranges_func=param_ranges_func,
        value_generator_func=value_generator_func,
        csv_writer_func=csv_writer_func,
        csv_downloader_func=csv_downloader_func,
    )
    generator.param_names = ["param1", "param2"]
    return generator


@patch("streamlit.selectbox")
def test_get_randomization_type(mock_selectbox, csv_generator):
    mock_selectbox.return_value = "Random"
    csv_generator.get_randomization_type()
    assert csv_generator.value_generator_func == generate_random_values

    mock_selectbox.return_value = "Latin Hypercube"
    csv_generator.get_randomization_type()
    assert csv_generator.value_generator_func == generate_lhs_values

    mock_selectbox.return_value = "Sobol"
    csv_generator.get_randomization_type()
    assert csv_generator.value_generator_func == generate_sobol_values


@patch("streamlit.number_input")
def test_get_input_values(mock_number_input, csv_generator):
    mock_number_input.return_value = 10
    csv_generator.get_input_values()
    assert csv_generator.nr_random_lines == 10


@patch("streamlit.number_input")
def test_get_decimal_places(mock_number_input, csv_generator):
    mock_number_input.return_value = 3
    csv_generator.get_decimal_places()
    assert csv_generator.decimal_places == 3


@patch("streamlit.selectbox")
@patch("streamlit.text_input")
def test_get_optimization_type(mock_text_input, mock_selectbox, csv_generator):
    mock_selectbox.return_value = "Single"
    mock_text_input.return_value = "objective"
    csv_generator.get_optimization_type()
    assert csv_generator.optimization_type == "Single"
    assert csv_generator.final_col_name == "objective"
    assert csv_generator.data_header == ["param1", "param2", "objective"]

    mock_selectbox.return_value = "Multi"
    mock_text_input.side_effect = ["objective1", "objective2"]
    csv_generator.get_optimization_type()
    assert csv_generator.optimization_type == "Multi"
    assert csv_generator.final_col_name1 == "objective1"
    assert csv_generator.final_col_name2 == "objective2"
    assert csv_generator.data_header == ["param1", "param2", "objective1", "objective2"]


# TODO: Fix the test_generate test case
# @patch("streamlit.selectbox")
# @patch("streamlit.number_input")
# @patch("streamlit.text_input")
# def test_generate(mock_text_input, mock_number_input, mock_selectbox, csv_generator):
#     # Mock the user inputs
#     mock_selectbox.side_effect = ["Random", "Single"]
#     mock_number_input.side_effect = [5, 2]
#     mock_text_input.return_value = "objective"

#     # Call the generate method
#     csv_generator.generate()

#     # Verify the sequence of method calls
#     assert mock_selectbox.call_args_list == [
#         call("Select randomization type:", ["Random", "Latin Hypercube", "Sobol"]),
#         call("Select optimization type:", ["Single", "Multi"]),
#     ]
#     assert mock_number_input.call_args_list == [
#         call("Number of random lines:", value=5),
#         call("Number of decimal places:", value=2, min_value=1, step=1),
#     ]
#     assert mock_text_input.call_args_list == [
#         call("Enter the name of the final column"),
#     ]

#     # Verify the final state of the CSVGenerator object
#     assert csv_generator.value_generator_func == generate_random_values
#     assert csv_generator.nr_random_lines == 5
#     assert csv_generator.decimal_places == 2
#     assert csv_generator.optimization_type == "Single"
#     assert csv_generator.final_col_name == "objective"
#     assert csv_generator.data_header == ["param1", "param2", "objective"]

#     # Verify the calls to the mocked functions
#     csv_generator.param_info_func.assert_called_once()
#     csv_generator.param_ranges_func.assert_called_once_with(
#         ["param1", "param2"], ["Integer", "Float"]
#     )
#     csv_generator.value_generator_func.assert_called_once_with(
#         ["param1", "param2"],
#         ["Integer", "Float"],
#         [(0, 10), (0.0, 1.0)],
#         [1, 0.1],
#         5,
#         2,
#     )
#     csv_generator.csv_writer_func.assert_called_once_with(
#         ["param1", "param2", "objective"],
#         [[1, 0.1], [2, 0.2], [3, 0.3], [4, 0.4], [5, 0.5]],
#     )
#     csv_generator.csv_downloader_func.assert_called_once()
