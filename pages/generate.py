import streamlit as st
from auto_csv_generator.csv_generator import CSVGenerator
from components.authenticate import initialize_session_state, check_authentication


def main() -> None:
    initialize_session_state()

    check_authentication()

    st.title("CSV Generator")
    generator = CSVGenerator()
    generator.generate()


if __name__ == "__main__":
    main()
