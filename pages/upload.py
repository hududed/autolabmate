import streamlit as st
from utils import create_table, insert_data
from streamlit_extras.switch_page_button import switch_page
from st_pages import hide_pages
import time


st.title("Upload CSV")

def main():
    st.write('Please upload a CSV file to create a new table.')
    file = st.file_uploader('Upload CSV', type='csv')
    if file is not None:
        st.write('Please enter a name for the new table.')
        with st.form(key='create_table_form'):
            table_name = st.text_input('Table Name')
            if st.form_submit_button('Create Table') and table_name != '':
                create_table(table_name)
                insert_data(table_name, file)
                st.write('Time for some preprocessing: switching to `Clean`.')
                time.sleep(2)
                # Switch to the clean page
                switch_page("clean")

if __name__ == '__main__':
    main()