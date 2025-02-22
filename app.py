import streamlit as st
import duckdb
import pandas as pd
import os
import tempfile
import shutil
from pathlib import Path

st.title("DuckDatabase Interactive Dashboard")

# Connect to an in-memory DuckDB database
conn = duckdb.connect(database=':memory:', read_only=False)
st.write("Connected to DuckDB in-memory database.")

# New section: Option to load DuckDB Database file
db_upload = st.file_uploader("Upload DuckDB Database file (.db or .duckdb)", type=["db", "duckdb"])
if db_upload:
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".duckdb") as temp_db:
        temp_db.write(db_upload.getbuffer())
        temp_db_path = temp_db.name
    # Close the current in-memory connection and connect to the uploaded database file
    conn.close()
    conn = duckdb.connect(database=temp_db_path, read_only=False)
    st.success("Connected to uploaded DuckDB database file.")

# Helper function to load CSV into a table using DuckDB's native CSV reader

def load_csv_table(conn, file_path, table_name):
    conn.execute(f"DROP TABLE IF EXISTS {table_name};")
    conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{file_path}', sample_size=-1);")

# Section: File upload for CSV/Excel files. Each file is loaded into its own table.
uploaded_files = st.file_uploader("Upload CSV/Excel files", accept_multiple_files=True, type=["csv", "xlsx"])

if uploaded_files:
    uploaded_table_names = []
    temp_dir = tempfile.mkdtemp()
    for uploaded_file in uploaded_files:
        file_suffix = uploaded_file.name.split('.')[-1].lower()
        safe_name = uploaded_file.name.split('.')[0].replace(' ', '_').replace('-', '_')
        table_name = f'{safe_name}'
        temp_file_path = Path(temp_dir) / uploaded_file.name
        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        if file_suffix == 'csv':
            try:
                load_csv_table(conn, str(temp_file_path), table_name)
                st.success(f"CSV file '{uploaded_file.name}' loaded into table '{table_name}'.")
                uploaded_table_names.append(table_name)
            except Exception as e:
                st.error(f"Error loading CSV file '{uploaded_file.name}': {e}")
        else:
            try:
                df = pd.read_excel(temp_file_path)
                conn.execute(f"DROP TABLE IF EXISTS {table_name};")
                conn.register(table_name, df)
                st.success(f"Excel file '{uploaded_file.name}' loaded into table '{table_name}'.")
                st.write(f"Preview of {table_name}:", df.head())
                uploaded_table_names.append(table_name)
            except Exception as e:
                st.error(f"Error loading Excel file '{uploaded_file.name}': {e}")
    shutil.rmtree(temp_dir)

# Section: Option to load files from a folder path
folder_path = st.text_input("Or enter folder path to load CSV/Excel files")
if folder_path and st.button("Load Folder Files"):
    try:
        folder_path = folder_path.strip().replace('\\', '/')
        folder = Path(folder_path)
        csv_files = list(folder.glob('*.csv'))
        excel_files = list(folder.glob('*.xlsx'))
        if not csv_files and not excel_files:
            st.warning("No CSV or Excel files found in the given folder.")
        else:
            folder_table_names = []
            # Process CSV files individually
            if csv_files:
                for f in csv_files:
                    table_name = f.stem.replace(' ', '_').replace('-', '_')
                    try:
                        load_csv_table(conn, str(f), table_name)
                        st.success(f"CSV file '{f.name}' loaded into table '{table_name}'.")
                        folder_table_names.append(table_name)
                    except Exception as e:
                        st.error(f"Error loading CSV file '{f.name}': {e}")
                st.write("Note: CSV files loaded. They are not previewed due to large size.")
            # Process Excel files individually
            if excel_files:
                for f in excel_files:
                    table_name = f.stem.replace(' ', '_').replace('-', '_')
                    try:
                        df = pd.read_excel(f)
                        conn.execute(f"DROP TABLE IF EXISTS {table_name};")
                        conn.register(table_name, df)
                        st.success(f"Excel file '{f.name}' loaded into table '{table_name}'.")
                        st.write(f"Preview of {table_name}:", df.head())
                        folder_table_names.append(table_name)
                    except Exception as e:
                        st.error(f"Error loading Excel file '{f.name}': {e}")
    except Exception as e:
        st.error(f"Error: {e}")

# Display Available Tables
try:
    tables_df = conn.execute("SHOW TABLES;").fetchdf()
    st.subheader("Available Tables")
    st.dataframe(tables_df)
except Exception as e:
    st.error(f"Error retrieving tables: {e}")

# Text area for SQL query input
query = st.text_area("Enter your DuckDB command", "SELECT * FROM <table_name>;", height=250)

if st.button("Run Query"):
    try:
        result = conn.execute(query).fetchdf()
        st.session_state.query_result = result  # store result for export
        st.dataframe(result)
    except Exception as e:
        st.error(f"Error: {e}")

# Button to create a sample table for demonstration purposes
if st.button("Create Sample Table"):
    try:
        conn.execute("CREATE TABLE my_table AS SELECT range(10) AS id, random() AS value;")
        st.success("Sample table created. Try running: SELECT * FROM my_table;")
    except Exception as e:
        st.error(f"Error creating sample table: {e}")

# Section to export the loaded data as CSV with a selectable delimiter
st.markdown("---")
st.subheader("Export Data as CSV")

# Query for available tables for export selection
try:
    available_tables = conn.execute("SHOW TABLES;").fetchdf()['name'].tolist()
except Exception as e:
    st.error(f"Error fetching table names: {e}")
    available_tables = []

if available_tables:
    export_table = st.selectbox("Select table to export", available_tables)
    delimiter = st.text_input("Enter CSV delimiter", value=",")
    if st.button("Export CSV"):
        try:
            output_file = f"exported_{export_table}.csv"
            conn.execute(f"COPY (SELECT * FROM {export_table}) TO '{output_file}' (DELIMITER '{delimiter}', HEADER true);")
            with open(output_file, "rb") as f:
                st.download_button(label="Download CSV", data=f, file_name=output_file, mime="text/csv")
        except Exception as e:
            st.error(f"Error exporting CSV: {e}")
else:
    st.info("No tables available for export.")

# Modify Export Section to export query result as CSV, Parquet or Excel
st.markdown("---")
st.subheader("Export Query Result")

if 'query_result' in st.session_state and not st.session_state.query_result.empty:
    export_format = st.radio("Select export format:", options=["CSV", "Parquet", "Excel"])
    if st.button("Export Query Result"):
        try:
            if export_format == "CSV":
                output_file = "query_result.csv"
                st.session_state.query_result.to_csv(output_file, index=False)
                with open(output_file, "rb") as f:
                    st.download_button(label="Download CSV", data=f, file_name=output_file, mime="text/csv")
            elif export_format == "Parquet":
                output_file = "query_result.parquet"
                st.session_state.query_result.to_parquet(output_file, index=False)
                with open(output_file, "rb") as f:
                    st.download_button(label="Download Parquet", data=f, file_name=output_file, mime="application/octet-stream")
            elif export_format == "Excel":
                output_file = "query_result.xlsx"
                st.session_state.query_result.to_excel(output_file, index=False)
                with open(output_file, "rb") as f:
                    st.download_button(label="Download Excel", data=f, file_name=output_file, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.error(f"Error exporting query result: {e}")
else:
    st.info("No query result available to export. Run a query first.")
