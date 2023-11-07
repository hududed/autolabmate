import streamlit as st
from auto_csv_generator.csv_generator import CSVGenerator


def main() -> None:
    st.title("CSV Generator")
    generator = CSVGenerator()
    generator.generate()


if __name__ == "__main__":
    main()
