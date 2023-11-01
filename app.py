import streamlit as st
import altair as alt
from sqlalchemy import text, inspect
from components.authenticate import engine, supabase_client
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import pandas as pd
import uuid

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
    # Query the table
    with engine.connect() as conn:
        query = f'SELECT * FROM {table_name}'
        df = pd.read_sql(query, conn)

    # Highlight the maximum values in the last column
    df_styled = df.style.highlight_max(subset=[df.columns[-1]])

    # Create two columns
    col1, col2 = st.columns(2)

    # Display the DataFrame in the first column
    col1.dataframe(df_styled)

    # Plot output as a function of each parameter in the second column
    output = df.iloc[:, -1]
    for param in df.columns[:-1]:
        chart_data = pd.DataFrame({
            'x': df[param],
            'y': output
        })
        chart = alt.Chart(chart_data).mark_circle().encode(
            x=alt.X('x', title=param),
            y=alt.Y('y', title=output.name)
        )
        col2.altair_chart(chart)
    sns.set_theme(context='talk')
    pairplot_fig = sns.pairplot(df, hue="param3", diag_kind='kde')
    st.pyplot(pairplot_fig)

def upload_csv_page():
    st.write('Please upload a CSV file to create a new table.')
    file = st.file_uploader('Upload CSV', type='csv')
    if file is not None:
        st.write('Please enter a name for the new table.')
        with st.form(key='create_table_form'):
            table_name = st.text_input('Table Name')
            if st.form_submit_button('Create Table') and table_name != '':
                create_table(table_name)
                insert_data(table_name, file)
                st.write('You can now view the table in the `Display Table` tab.')

def display_table_page():
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    if table_names:
        table_name = st.selectbox('Select a table to display', table_names)
        display_table(table_name)
    else:
        st.write('No tables found in the database.')

def main():
    st.set_page_config(page_title='Web App', page_icon=':chart_with_upwards_trend:')
    st.title('Web App')

    # Check if user is logged in
    session = supabase_client.auth.get_session()
    if session and session.user:
        # User is logged in

        # Define pages
        pages = ['Upload CSV', 'Display Table']

        # Add buttons to the sidebar for page navigation
        for page in pages:
            if st.sidebar.button(page):
                st.session_state.selected_page = page
        
        # Add logout button to the sidebar
        if st.sidebar.button('Logout'):
            supabase_client.auth.sign_out()
            st.session_state.clear()  # Clear the session state
            st.rerun()  # Rerun the script to update the UI

        if 'selected_page' in st.session_state:
            if st.session_state.selected_page == 'Upload CSV':
                upload_csv_page()
            elif st.session_state.selected_page == 'Display Table':
                display_table_page()
    else:
        # User is not logged in
        st.write('Please login or create a new account to get started.')
        email = st.text_input('Email')
        password = st.text_input('Password', type='password')
        if st.button('Login'):
            response = login(credentials={'email': email, 'password': password})
            # Ignore any errors for now
            import pprint
            pprint.pprint(response)
            st.write(f'Login successful!')
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