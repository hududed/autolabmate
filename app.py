import streamlit as st
import supabase
import psycopg2
from sqlalchemy import create_engine, text
import pandas as pd
import uuid
import os
from dotenv import load_dotenv
load_dotenv()

# Load Supabase credentials from .env file
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')

# Initialize Supabase client
supabase_client = supabase.create_client(supabase_url, supabase_key)
PG_PASS = os.getenv('PG_PASS')
DATABASE_URL = f'postgresql://postgres:{PG_PASS}@db.zugnayzgayyoveqcmtcd.supabase.co:5432/postgres'
engine = create_engine(DATABASE_URL)

# Define routes
def create_account(email, password):
    # Create new account
    response = supabase_client.auth.sign_up({
        'email': email,
        'password': password
    })
    return response

def login(credentials):
    # Login to existing account
    response = supabase_client.auth.sign_in_with_password(credentials)
    return response

def upload_to_bucket(bucket_name, file_name, file_content):
    # Generate new UUID for file name
    new_file_name = str(uuid.uuid4()) + '-' + file_name
    st.write(f'Uploading file "{file_name}" to bucket "{bucket_name}" as "{new_file_name}"')
    # Upload file to bucket
    supabase_client.storage.from_(bucket_name).upload(new_file_name, file_content)
    st.write(f'"{new_file_name}" uploaded to bucket "{bucket_name}"')

def create_table(table_name):
    # Create new table
    with engine.connect() as conn:
        query = text(f'CREATE TABLE IF NOT EXISTS {table_name} (id UUID NOT NULL DEFAULT uuid_generate_v4(), PRIMARY KEY (id));')
        conn.execute(query)
        st.write(f'Table "{table_name}" created in database')

def insert_data(table_name, data):
    # Insert data into table
    with engine.connect() as conn:
        df = pd.read_csv(data)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        st.write(f'{data.name} inserted into table "{table_name}"')

def display_table(table_name):
    # Display table
    with engine.connect() as conn:
        query = text(f'SELECT * FROM {table_name};')
        df = pd.read_sql(query, conn)
        st.write(df)


def main():
    st.set_page_config(page_title='Web App', page_icon=':chart_with_upwards_trend:')
    st.title('Web App')
    # Check if user is logged in
    session = supabase_client.auth.get_session()
    if session and session.user:
        # User is logged in
        st.write('Please upload a CSV file to create a new table.')
        file = st.file_uploader('Upload CSV', type='csv')
        if file is not None:
            st.write('Please enter a name for the new table.')
            with st.form(key='create_table_form'):
                table_name = st.text_input('Table Name')
                if st.form_submit_button('Create Table') and table_name != '':
                    create_table(table_name)
                    insert_data(table_name, file)
                    st.write('Redirecting to new page...')
                    display_table(table_name)
                    # st.rerun()
                    
    else:
        # User is not logged in
        st.write('Please login or create a new account to get started.')
        email = st.text_input('Email')
        password = st.text_input('Password', type='password')
        if st.button('Login'):
            response = login(credentials={'email': email, 'password': password})
            # Ignore any errors for now
            st.write('Login successful!')
            # Store login state in SessionState object
            st.session_state.logged_in = True
            # Redirect to new page
            st.rerun()
        elif st.button('Create Account'):
            response = create_account(email, password)
            # Ignore any errors for now
            st.write('Account created successfully!')
            st.write('Please login to continue.')

if __name__ == '__main__':
    
    main()