import streamlit as st
from utils import create_table, insert_data
from st_pages import hide_pages


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
                st.write('You can now view the table in `Dashboard`.')

if __name__ == '__main__':
    main()