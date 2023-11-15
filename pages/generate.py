import streamlit as st
from auto_csv_generator.csv_generator import CSVGenerator


def main() -> None:
    if not st.session_state.authentication_status:
        st.info("Please Login from the Home page and try again.")
        st.stop()
    st.title("CSV Generator")
    generator = CSVGenerator()
    generator.generate()


if __name__ == "__main__":
    main()
