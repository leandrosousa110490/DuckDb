# DuckDB GUI Application

A powerful desktop application for interacting with DuckDB databases through a user-friendly graphical interface.

## Features

### Database Management
- Create new DuckDB databases
- Open existing database files
- View and manage database tables
- Track recently used databases

### SQL Query Interface
- Multiple query tabs for running different queries simultaneously
- SQL syntax highlighting with dark/light theme support
- SQL autocompletion for tables, columns, and SQL keywords
- Save and manage frequently used queries
- Query pagination for large result sets

### Data Import
- Import data from CSV files with delimiter options
- Import data from Excel spreadsheets
- Import data from Parquet files
- **NEW**: Bulk import multiple Excel files from a folder
  - Import files containing the same sheet name
  - Choose between using only common columns or all columns
  - Add to existing table, create new, or replace existing
- **NEW**: Bulk import multiple CSV files from a folder
  - Select delimiter for all files in the folder
  - Choose between using only common columns or all columns
  - Add to existing table, create new, or replace existing

### Data Export
- Export query results to CSV
- Export query results to Excel
- Export query results to Parquet
- Customizable export settings

### Data Exploration
- Display table schemas
- Browse table data
- Filter query results
- Copy data (cell, row, column, or entire table)

### User Interface
- Dark and light theme support
- Context menus for common operations
- Tabbed interface for managing multiple queries
- Progress indicators for long-running operations

## Installation

### Requirements
- Python 3.7 or higher
- PyQt6
- DuckDB
- Pandas
- Dependencies for specific file formats:
  - openpyxl (for Excel)
  - pyarrow (for Parquet)

### Setup

1. Clone this repository:
```
git clone https://github.com/yourusername/duckdb-gui.git
cd duckdb-gui
```

2. Install dependencies:
```
pip install pyqt6 duckdb pandas openpyxl pyarrow
```

3. Run the application:
```
python app.py
```

## Usage

### Working with Databases
- **Create Database**: File → New Database
- **Open Database**: File → Open Database
- **Close Database**: File → Close Database

### Running Queries
- Type SQL queries in the query editor
- Press the "Run Query" button to execute
- Results appear in the table below
- Use Ctrl+T to open a new query tab
- Use Ctrl+W to close the current tab

### Importing Data
- Choose File → Import → CSV/Parquet/Excel
- Select your file and follow the import wizard
- For bulk Excel import, select "Bulk Import Excel Files from Folder"

### Exporting Results
- Run a query to get results
- Select Export → Export as CSV/Excel/Parquet
- Choose your destination file and export options

### Saving Queries
- Write a useful query
- Choose Query → Save Query
- Give it a name and description
- Access saved queries from Query → Saved Queries

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 