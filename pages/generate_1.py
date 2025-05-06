import streamlit as st

# Import navigation after page config
from auto_csv_generator.csv_generator import (
    CSVGenerator,
)
from auto_csv_generator.csv_handler import download_csv, write_csv
from auto_csv_generator.parameter_handler import (
    get_parameter_info,
    get_parameter_ranges,
)
from auto_csv_generator.value_generator import generate_random_values
from dependencies.authentication import check_authentication, initialize_session_state
from dependencies.navigation import authenticate_and_show_nav

# Set page config first
st.set_page_config(page_title="Upload | Autolabmate", page_icon="⬆️")
# Check authentication and show navigation
authenticate_and_show_nav()


def main() -> None:
    initialize_session_state()

    check_authentication()

    st.title("CSV Generator")
    generator = CSVGenerator(
        param_info_func=get_parameter_info,
        param_ranges_func=get_parameter_ranges,
        value_generator_func=generate_random_values,  # Initial function, will be replaced
        csv_writer_func=write_csv,
        csv_downloader_func=download_csv,
    )
    generator.generate()


if __name__ == "__main__":
    main()
