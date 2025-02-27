import re
import sys
import duckdb
import pandas as pd
import os
import tempfile
import shutil
from pathlib import Path

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLineEdit, QFileDialog, QTableView, QMessageBox, QLabel, QPlainTextEdit, QCompleter,
                             QListWidget, QSplitter, QTabWidget, QProgressDialog, QStyle, QComboBox, QMenuBar, QDialog, QDialogButtonBox,
                             QComboBox, QFormLayout)
from PyQt6.QtCore import QAbstractTableModel, Qt, QThread, pyqtSignal, QRegularExpression, QTimer, QObject
from PyQt6.QtGui import QTextCursor, QSyntaxHighlighter, QTextCharFormat, QAction


SQL_KEYWORDS = [
    'SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE', 'JOIN', 'INNER', 'LEFT', 'RIGHT',
    'FULL', 'ON', 'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT', 'OFFSET', 'CREATE', 'DROP', 'ALTER',
    'TABLE', 'VIEW', 'INDEX', 'DISTINCT', 'VALUES', 'INTO', 'AS', 'AND', 'OR', 'NOT', 'NULL',
    'PRIMARY', 'KEY', 'FOREIGN', 'REFERENCES', 'DEFAULT', 'CONSTRAINT', 'UNIQUE', 'CHECK',
    'AUTO_INCREMENT', 'CASCADE', 'SET', 'BETWEEN', 'LIKE', 'IN', 'EXISTS', 'ALL', 'ANY', 'SOME',
    'UNION', 'INTERSECT', 'EXCEPT', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'WITH',
    'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'FIRST', 'LAST', 'COALESCE', 'NULLIF',
    'CAST', 'CONVERT', 'UPPER', 'LOWER', 'TRIM', 'SUBSTRING', 'CONCAT',
    'DESC', 'ASC', 'IS', 'TRUE', 'FALSE', 'USING', 'NATURAL', 'CROSS', 'OUTER',
    'OVER', 'PARTITION', 'BY', 'ROWS', 'RANGE', 'UNBOUNDED', 'PRECEDING', 'FOLLOWING',
    'CURRENT', 'ROW', 'RANK', 'DENSE_RANK', 'ROW_NUMBER', 'LAG', 'LEAD'
]

class SQLHighlighter(QSyntaxHighlighter):
    def __init__(self, parent):
        super().__init__(parent)
        self.highlightingRules = []

        # SQL Keywords format (blue)
        keywordFormat = QTextCharFormat()
        keywordFormat.setForeground(Qt.GlobalColor.blue)
        keywordFormat.setFontWeight(700)  # Bold
        for word in SQL_KEYWORDS:
            pattern = QRegularExpression(f"\\b{word}\\b")
            pattern.setPatternOptions(QRegularExpression.PatternOption.CaseInsensitiveOption)
            self.highlightingRules.append((pattern, keywordFormat))

        # Function format (dark cyan)
        functionFormat = QTextCharFormat()
        functionFormat.setForeground(Qt.GlobalColor.darkCyan)
        functionPattern = QRegularExpression(r"\b[A-Za-z0-9_]+(?=\s*\()")
        self.highlightingRules.append((functionPattern, functionFormat))

        # String format (dark red)
        stringFormat = QTextCharFormat()
        stringFormat.setForeground(Qt.GlobalColor.darkRed)
        self.highlightingRules.append((QRegularExpression("'[^']*'"), stringFormat))
        self.highlightingRules.append((QRegularExpression('"[^"]*"'), stringFormat))

        # Comment format (dark green)
        commentFormat = QTextCharFormat()
        commentFormat.setForeground(Qt.GlobalColor.darkGreen)
        self.highlightingRules.append((QRegularExpression("--[^\n]*"), commentFormat))

        # Multi-line comment format
        self.multiLineCommentFormat = QTextCharFormat()
        self.multiLineCommentFormat.setForeground(Qt.GlobalColor.darkGreen)

        # Multi-line comment expressions
        self.commentStartExpression = QRegularExpression("/\\*")
        self.commentEndExpression = QRegularExpression("\\*/")

    def highlightBlock(self, text):
        # Single-line comments and keywords
        for pattern, fmt in self.highlightingRules:
            iterator = pattern.globalMatch(text)
            while iterator.hasNext():
                match = iterator.next()
                start = match.capturedStart()
                length = match.capturedLength()
                self.setFormat(start, length, fmt)

        # Multi-line comments
        self.setCurrentBlockState(0)
        startIndex = 0
        if self.previousBlockState() != 1:
            startIndex = self.commentStartExpression.match(text).capturedStart()

        while startIndex >= 0:
            endMatch = self.commentEndExpression.match(text, startIndex)
            endIndex = endMatch.capturedStart()
            commentLength = 0
            if endIndex == -1:
                self.setCurrentBlockState(1)
                commentLength = len(text) - startIndex
            else:
                commentLength = endIndex - startIndex + endMatch.capturedLength()

            self.setFormat(startIndex, commentLength, self.multiLineCommentFormat)
            startIndex = self.commentStartExpression.match(text, startIndex + commentLength).capturedStart()


class CodeEditor(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighter = SQLHighlighter(self.document())
        self.setStyleSheet("""
            QPlainTextEdit {
                font-family: Consolas, Monaco, 'Courier New', monospace;
                font-size: 12px;
                background-color: #f8f9fa;
                color: #212529;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
            }
        """)

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        # Trigger rehighlighting after each keypress
        self.highlighter.rehighlight()

class PandasModel(QAbstractTableModel):
    CHUNK_SIZE = 1000  # Number of rows to load at a time

    def __init__(self, df=None, parent=None):
        super().__init__(parent)
        self._df = pd.DataFrame() if df is None else df
        self._total_rows = self._df.shape[0]
        self._loaded_chunks = {}
        self._column_names = list(self._df.columns)

    def _ensure_chunk_loaded(self, row_index):
        chunk_index = row_index // self.CHUNK_SIZE
        if chunk_index not in self._loaded_chunks:
            start_idx = chunk_index * self.CHUNK_SIZE
            end_idx = min(start_idx + self.CHUNK_SIZE, self._total_rows)
            self._loaded_chunks[chunk_index] = self._df.iloc[start_idx:end_idx]
            # Keep only the most recent chunks to manage memory
            if len(self._loaded_chunks) > 3:  # Keep only 3 chunks in memory
                min_key = min(k for k in self._loaded_chunks.keys() if k != chunk_index)
                del self._loaded_chunks[min_key]

    def rowCount(self, parent=None):
        return self._total_rows

    def columnCount(self, parent=None):
        return len(self._column_names)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None

        row = index.row()
        col = index.column()

        if 0 <= row < self._total_rows and 0 <= col < len(self._column_names):
            chunk_index = row // self.CHUNK_SIZE
            self._ensure_chunk_loaded(row)
            chunk = self._loaded_chunks[chunk_index]
            value = chunk.iat[row % self.CHUNK_SIZE, col]
            return str(value)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None

        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self._column_names):
                return str(self._column_names[section])
        else:
            if 0 <= section < self._total_rows:
                return str(section + 1)
        return None


class QueryWorker(QThread):
    resultReady = pyqtSignal(object)  # will emit a DataFrame
    errorOccurred = pyqtSignal(str)
    progressUpdate = pyqtSignal(str)  # for updating progress message

    def __init__(self, db_path, query):
        super().__init__()
        self.db_path = db_path
        self.query = query
        self.start_time = None
        self.timer = None
        self.current_status = ""
        self.chunk_size = 250000  # Increased chunk size for better performance
        self.max_workers = os.cpu_count() or 2  # Number of parallel workers

    def update_elapsed_time(self):
        if self.start_time:
            elapsed = (pd.Timestamp.now() - self.start_time).total_seconds()
            self.progressUpdate.emit(f"{self.current_status} (Elapsed: {elapsed:.1f}s)")

    def cleanup_timer(self):
        if self.timer:
            self.timer.stop()
            self.timer.deleteLater()
            self.timer = None

    def start_timer(self):
        self.start_time = pd.Timestamp.now()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_elapsed_time)
        self.timer.start(100)  # Update more frequently (every 100ms)

    def process_chunk(self, chunk):
        return pd.DataFrame(chunk)

    def run(self):
        try:
            self.start_timer()
            # Open a new connection to avoid threading issues
            conn = duckdb.connect(database=self.db_path, read_only=False)
            # Split the queries by semicolon
            queries = [q.strip() for q in self.query.split(';') if q.strip()]
            last_df = None

            for i, q in enumerate(queries):
                self.current_status = f"Executing query {i+1}/{len(queries)}"
                self.update_elapsed_time()
                
                # Automatically quote the reserved keyword 'Transaction' if unquoted
                q = re.sub(r'(?<!")\b(transaction)\b(?!")', '"Transaction"', q, flags=re.IGNORECASE)
                
                # For SELECT queries, automatically quote column names containing spaces and table names with spaces
                if q.strip().lower().startswith('select'):
                    m = re.search(r'(?i)^select\s+(.*?)\s+from\s+([^\s;]+(?:\s+[^\s;]+)*)(\s+where|\s+group by|\s+order by|\s+limit|;|$)', q, flags=re.DOTALL)
                    if m is not None:
                        cols_str = m.group(1)
                        table_name = m.group(2).strip()
                        clause = m.group(3) if m.group(3) is not None else ""
                        
                        # Process columns
                        cols = [col.strip() for col in cols_str.split(",")]
                        new_cols = []
                        for col in cols:
                            if col == '*':
                                new_cols.append(col)
                            else:
                                if not (col.startswith('"') and col.endswith('"')) and ' ' in col:
                                    col = f'"{col}"'
                                new_cols.append(col)
                        new_cols_str = ', '.join(new_cols)

                        # Quote table name if it contains spaces and isn't already quoted
                        if not (table_name.startswith('"') and table_name.endswith('"')) and ' ' in table_name:
                            table_name = f'"{table_name}"'

                        # Process the clause
                        clause = clause.strip()
                        if clause == ";" or clause == "":
                            clause = ""
                        else:
                            if not clause.startswith(" "):
                                clause = " " + clause

                        # Rebuild the query using captured clause
                        q = f"SELECT {new_cols_str} FROM {table_name}{clause}"
                
                cur = conn.execute(q)
                
                # Only fetch data for SELECT queries
                if q.lower().startswith('select'):
                    self.current_status = f"Fetching data for query {i+1}/{len(queries)}"
                    self.update_elapsed_time()
                    
                    # Optimized chunk processing with parallel execution
                    chunks = []
                    from concurrent.futures import ThreadPoolExecutor
                    
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        while True:
                            chunk = cur.fetch_df_chunk(self.chunk_size)
                            if chunk is None or len(chunk) == 0:
                                break
                            # Process chunks in parallel
                            future = executor.submit(self.process_chunk, chunk)
                            chunks.append(future)
                            self.update_elapsed_time()
                        
                        # Collect results from futures
                        processed_chunks = [future.result() for future in chunks]
                    
                    if processed_chunks:
                        last_df = pd.concat(processed_chunks, ignore_index=True)
                    else:
                        last_df = pd.DataFrame()
            
            conn.close()
            if last_df is None:
                # For commands that do not return data, show a success message
                last_df = pd.DataFrame({'message': ['Command executed successfully']})
            
            self.cleanup_timer()
            self.resultReady.emit(last_df)
        except Exception as e:
            self.cleanup_timer()
            self.errorOccurred.emit(str(e))

class ExportWorker(QObject):
    finished = pyqtSignal(str)  # Signal to emit when export is complete
    progress = pyqtSignal(str)  # Signal to emit progress updates
    error = pyqtSignal(str)    # Signal to emit if an error occurs
    success = pyqtSignal(str)  # Signal to emit on successful export

    CHUNK_SIZE = 50000  # Chunk size for better memory management
    
    def __init__(self, df, file_path, format):
        super().__init__()
        self.df = df
        self.file_path = file_path
        self.format = format
        self.is_cancelled = False
    
    def export_csv_in_chunks(self):
        # Write header first
        self.df.iloc[0:0].to_csv(self.file_path, index=False)
        
        # Then append data in chunks
        total_rows = len(self.df)
        for i in range(0, total_rows, self.CHUNK_SIZE):
            if self.is_cancelled:
                return
            
            end_idx = min(i + self.CHUNK_SIZE, total_rows)
            chunk = self.df.iloc[i:end_idx]
            
            chunk.to_csv(self.file_path, mode='a', header=False, index=False)
            
            progress_pct = min(100, int((end_idx / total_rows) * 100))
            self.progress.emit(f"Exporting CSV: {progress_pct}% complete ({end_idx}/{total_rows} rows)")
    
    def export_excel_in_chunks(self):
        # For Excel, we need to use ExcelWriter
        with pd.ExcelWriter(self.file_path, engine='openpyxl') as writer:
            # Write data in chunks
            total_rows = len(self.df)
            rows_written = 0
            
            # Write the header
            header_df = pd.DataFrame(columns=self.df.columns)
            header_df.to_excel(writer, sheet_name='Sheet1', index=False)
            
            # Write the data in chunks
            for i in range(0, total_rows, self.CHUNK_SIZE):
                if self.is_cancelled:
                    return
                
                end_idx = min(i + self.CHUNK_SIZE, total_rows)
                chunk = self.df.iloc[i:end_idx]
                
                # Write this chunk to Excel, starting after the header
                chunk.to_excel(
                    writer, 
                    sheet_name='Sheet1', 
                    index=False,
                    header=False,
                    startrow=rows_written + 1  # +1 for the header
                )
                
                rows_written += len(chunk)
                progress_pct = min(100, int((rows_written / total_rows) * 100))
                self.progress.emit(f"Exporting Excel: {progress_pct}% complete ({rows_written}/{total_rows} rows)")
    
    def export_parquet_in_chunks(self):
        # Parquet is already optimized for large datasets
        self.progress.emit("Exporting to Parquet format...")
        self.df.to_parquet(self.file_path, index=False)
        self.progress.emit("Parquet export complete")
    
    def run(self):
        try:
            self.progress.emit(f"Starting export to {self.format} format...")
            
            if self.format == 'csv':
                self.export_csv_in_chunks()
            elif self.format == 'excel':
                self.export_excel_in_chunks()
            elif self.format == 'parquet':
                self.export_parquet_in_chunks()
            else:
                self.error.emit(f"Unsupported format: {self.format}")
                return
            
            if not self.is_cancelled:
                self.success.emit(self.file_path)
            self.finished.emit(self.file_path)
            
        except Exception as e:
            self.error.emit(f"Unexpected error: {str(e)}")
            self.finished.emit(self.file_path)

    def cancel(self):
        self.is_cancelled = True

class MergeFilesWorker(QObject):
    progress = pyqtSignal(str)  # Signal to emit progress updates
    error = pyqtSignal(str)     # Signal to emit if an error occurs
    success = pyqtSignal(str)   # Signal to emit on successful merge
    finished = pyqtSignal()     # Signal to emit when process is complete
    table_created = pyqtSignal(str, str)  # Signal to emit when a table is created (db_path, table_name)
    
    CHUNK_SIZE = 100000  # Chunk size for better memory management
    
    def __init__(self, folder_path, db_path, table_name=None, use_existing_table=False):
        super().__init__()
        self.folder_path = folder_path
        self.db_path = db_path
        self.table_name = table_name or "merged_data"
        self.use_existing_table = use_existing_table
        self.is_cancelled = False
        self.is_duckdb = db_path.lower().endswith('.duckdb')
        
    def get_file_list(self):
        """Get list of supported files in the folder"""
        supported_extensions = ['.csv', '.xlsx', '.xls', '.parquet']
        files = []
        
        for ext in supported_extensions:
            files.extend(list(Path(self.folder_path).glob(f'*{ext}')))
        
        return files
    
    def analyze_column_types(self, sample_df):
        """Analyze column types to ensure proper database insertion"""
        column_types = {}
        
        for col in sample_df.columns:
            # Check for numeric columns
            if pd.api.types.is_numeric_dtype(sample_df[col]):
                if pd.api.types.is_integer_dtype(sample_df[col]):
                    column_types[col] = 'INTEGER'
                else:
                    column_types[col] = 'DOUBLE'
            # Check for datetime columns
            elif pd.api.types.is_datetime64_dtype(sample_df[col]):
                column_types[col] = 'TIMESTAMP'
            # Check for boolean columns
            elif pd.api.types.is_bool_dtype(sample_df[col]):
                column_types[col] = 'BOOLEAN'
            # Default to text for other types
            else:
                column_types[col] = 'TEXT'
                
        return column_types
    
    def get_existing_columns(self, conn):
        """Get existing columns from the table"""
        try:
            if self.is_duckdb:
                result = conn.execute(f"PRAGMA table_info({self.table_name})").fetchall()
                return [row[1] for row in result]  # Column name is at index 1
            else:
                import sqlite3
                cursor = conn.cursor()
                result = cursor.execute(f"PRAGMA table_info({self.table_name})").fetchall()
                return [row[1] for row in result]  # Column name is at index 1
        except Exception as e:
            self.progress.emit(f"Error getting existing columns: {str(e)}")
            return []
    
    def add_missing_columns(self, conn, sample_df, existing_columns):
        """Add missing columns to the existing table"""
        new_columns = []
        column_types = self.analyze_column_types(sample_df)
        
        for col in sample_df.columns:
            if col not in existing_columns:
                new_columns.append((col, column_types[col]))
        
        if not new_columns:
            return
            
        self.progress.emit(f"Adding {len(new_columns)} new columns to the table...")
        
        try:
            for col_name, col_type in new_columns:
                if self.is_duckdb:
                    conn.execute(f'ALTER TABLE "{self.table_name}" ADD COLUMN "{col_name}" {col_type}')
                else:
                    cursor = conn.cursor()
                    cursor.execute(f'ALTER TABLE "{self.table_name}" ADD COLUMN "{col_name}" {col_type}')
                    conn.commit()
                    
            self.progress.emit(f"Added {len(new_columns)} new columns to the table.")
        except Exception as e:
            self.error.emit(f"Error adding new columns: {str(e)}")
    
    def create_table_if_not_exists(self, conn, sample_df):
        """Create table with appropriate column types if it doesn't exist"""
        column_types = self.analyze_column_types(sample_df)
        
        # Build CREATE TABLE statement
        columns_sql = ', '.join([f'"{col}" {dtype}' for col, dtype in column_types.items()])
        create_table_sql = f'CREATE TABLE IF NOT EXISTS "{self.table_name}" ({columns_sql})'
        
        # Execute the statement
        if self.is_duckdb:
            conn.execute(create_table_sql)
        else:
            cursor = conn.cursor()
            cursor.execute(create_table_sql)
            conn.commit()
    
    def read_file_in_chunks(self, file_path):
        """Read a file in chunks to handle large files efficiently"""
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.csv':
            # For CSV, use chunked reading
            for chunk in pd.read_csv(file_path, chunksize=self.CHUNK_SIZE):
                if self.is_cancelled:
                    return
                yield chunk
                
        elif file_ext in ['.xlsx', '.xls']:
            # For Excel, we need to read it all at once, but we can process it in chunks
            df = pd.read_excel(file_path)
            total_rows = len(df)
            
            for i in range(0, total_rows, self.CHUNK_SIZE):
                if self.is_cancelled:
                    return
                end_idx = min(i + self.CHUNK_SIZE, total_rows)
                yield df.iloc[i:end_idx]
                
        elif file_ext == '.parquet':
            # For Parquet, use chunked reading if available
            try:
                # Try using pyarrow for chunked reading
                import pyarrow.parquet as pq
                table = pq.read_table(file_path)
                total_rows = table.num_rows
                
                for i in range(0, total_rows, self.CHUNK_SIZE):
                    if self.is_cancelled:
                        return
                    end_idx = min(i + self.CHUNK_SIZE, total_rows)
                    chunk = table.slice(i, end_idx - i).to_pandas()
                    yield chunk
            except ImportError:
                # Fall back to pandas if pyarrow is not available
                df = pd.read_parquet(file_path)
                total_rows = len(df)
                
                for i in range(0, total_rows, self.CHUNK_SIZE):
                    if self.is_cancelled:
                        return
                    end_idx = min(i + self.CHUNK_SIZE, total_rows)
                    yield df.iloc[i:end_idx]
    
    def insert_data(self, conn, df):
        """Insert data into the database"""
        if self.is_duckdb:
            # For DuckDB, we can use the append method
            conn.execute(f"INSERT INTO {self.table_name} SELECT * FROM df")
        else:
            # For SQLite, we need to use the pandas to_sql method
            df.to_sql(self.table_name, conn, if_exists='append', index=False)
    
    # Make run a slot that can be connected to QThread.started signal
    def run(self):
        try:
            files = self.get_file_list()
            
            if not files:
                self.error.emit("No supported files found in the selected folder.")
                self.finished.emit()
                return
                
            self.progress.emit(f"Found {len(files)} files to process.")
            
            # Connect to the database
            if self.is_duckdb:
                conn = duckdb.connect(self.db_path)
            else:
                import sqlite3
                conn = sqlite3.connect(self.db_path)
            
            # Process the first file to get the schema
            first_file = files[0]
            self.progress.emit(f"Analyzing schema from {first_file.name}...")
            
            # Read a small sample to determine schema
            if first_file.suffix.lower() == '.csv':
                sample_df = pd.read_csv(first_file, nrows=1000)
            elif first_file.suffix.lower() in ['.xlsx', '.xls']:
                sample_df = pd.read_excel(first_file, nrows=1000)
            else:  # parquet
                sample_df = pd.read_parquet(first_file)
                if len(sample_df) > 1000:
                    sample_df = sample_df.iloc[:1000]
            
            # Check if we're using an existing table
            if self.use_existing_table:
                # Get existing columns
                existing_columns = self.get_existing_columns(conn)
                
                # Add any missing columns
                self.add_missing_columns(conn, sample_df, existing_columns)
            else:
                # Create the table if it doesn't exist
                self.create_table_if_not_exists(conn, sample_df)
            
            # Process all files
            total_files = len(files)
            for i, file_path in enumerate(files, 1):
                if self.is_cancelled:
                    break
                    
                self.progress.emit(f"Processing file {i}/{total_files}: {file_path.name}")
                
                # Process the file in chunks
                chunk_count = 0
                rows_processed = 0
                
                for chunk in self.read_file_in_chunks(file_path):
                    if self.is_cancelled:
                        break
                        
                    chunk_count += 1
                    rows_processed += len(chunk)
                    
                    # Insert the chunk into the database
                    if self.is_duckdb:
                        # Register the DataFrame as a view
                        conn.register('df', chunk)
                        self.insert_data(conn, None)
                    else:
                        self.insert_data(conn, chunk)
                    
                    self.progress.emit(f"Processed {rows_processed} rows from {file_path.name} (chunk {chunk_count})")
                
                self.progress.emit(f"Completed file {i}/{total_files}: {file_path.name}")
            
            # Close the database connection
            if self.is_duckdb:
                conn.close()
            else:
                conn.commit()
                conn.close()
            
            if not self.is_cancelled:
                self.success.emit(f"Successfully merged {len(files)} files into {self.db_path}")
                # Emit signal that table was created/updated
                self.table_created.emit(self.db_path, self.table_name)
            
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(f"Error merging files: {str(e)}")
            self.finished.emit()
    
    def cancel(self):
        self.is_cancelled = True
        self.progress.emit("Cancelling operation...")

class QueryTab(QWidget):
    database_loaded = pyqtSignal(str)  # Signal when database is loaded
    database_closed = pyqtSignal()     # Signal when database is closed
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.current_df = None
        self.current_db_path = None
        self.close_db_button = None  # Will be set by MainWindow

    def export_data(self, format_type):
        current_tab = self.tab_widget.currentWidget()
        if not current_tab or not hasattr(current_tab, 'current_df') or current_tab.current_df is None:
            QMessageBox.warning(self, "Warning", "No data to export.")
            return

        file_dialog = QFileDialog(self)
        file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        
        file_types = {
            'csv': ('CSV Files (*.csv)', '.csv'),
            'excel': ('Excel Files (*.xlsx)', '.xlsx'),
            'parquet': ('Parquet Files (*.parquet)', '.parquet')
        }
        
        file_type, extension = file_types[format_type]
        file_dialog.setNameFilter(file_type)
        
        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            file_path = file_dialog.selectedFiles()[0]
            if not file_path.endswith(extension):
                file_path += extension
                
            try:
                # Create a progress dialog
                progress_dialog = QProgressDialog("Exporting data...", "Cancel", 0, 0, self)
                progress_dialog.setWindowTitle("Export Progress")
                progress_dialog.setMinimumDuration(0)
                progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
                progress_dialog.setAutoClose(False)
                progress_dialog.setAutoReset(False)
                
                # Create and configure the export worker
                self.export_worker = ExportWorker(current_tab.current_df, file_path, format_type)
                
                # Create a thread to run the worker
                self.export_thread = QThread()
                self.export_worker.moveToThread(self.export_thread)
                
                # Connect signals
                self.export_worker.progress.connect(progress_dialog.setLabelText)
                self.export_worker.success.connect(lambda path: QMessageBox.information(self, "Success", f"File saved successfully to {path}"))
                self.export_worker.error.connect(lambda msg: QMessageBox.critical(self, "Error", f"Export failed: {msg}"))
                self.export_worker.finished.connect(progress_dialog.close)
                self.export_worker.finished.connect(self.export_thread.quit)
                
                # Connect thread signals
                self.export_thread.started.connect(self.export_worker.run)
                self.export_thread.finished.connect(self.export_thread.deleteLater)
                
                # Connect cancel button
                progress_dialog.canceled.connect(self.export_worker.cancel)
                
                # Start the thread
                progress_dialog.show()
                self.export_thread.start()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export file: {str(e)}")

    def setup_ui(self):
        self.setWindowTitle("SQL Query Editor")
        layout = QVBoxLayout(self)

        # Create a splitter to hold the available tables list and the query/result area
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left side: collapsible tables list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Header with collapse button and file menu
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        self.collapse_button = QPushButton("◀")
        self.collapse_button.setFixedSize(20, 20)
        self.collapse_button.clicked.connect(self.toggle_table_list)
        header_layout.addWidget(self.collapse_button)
        
        # Remove the file menu as it's no longer needed
        
        header_layout.addWidget(QLabel("Available Tables:"))
        header_layout.addStretch()
        left_layout.addWidget(header_widget)
        
        # Add database list widget
        self.db_list = QListWidget()
        self.db_list.setMaximumWidth(200)
        self.db_list.setMinimumWidth(100)
        self.db_list.setStyleSheet(
            "QListWidget { background-color: palette(base); color: palette(text); border: 1px solid palette(mid); }"
            "QListWidget::item:selected { background-color: palette(highlight); color: palette(highlighted-text); }"
        )
        left_layout.addWidget(self.db_list)
        
        self.table_list = QListWidget()
        self.table_list.setDisabled(True)
        self.table_list.setMaximumWidth(200)  # Set maximum width
        self.table_list.setMinimumWidth(100)  # Set minimum width
        self.table_list.setStyleSheet(
            "QListWidget { background-color: palette(base); color: palette(text); border: 1px solid palette(mid); }"
            "QListWidget::item:selected { background-color: palette(highlight); color: palette(highlighted-text); }"
        )
        left_layout.addWidget(self.table_list)
        splitter.addWidget(left_panel)

        # Right side: query tabs
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Tab widget for queries
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)
        right_layout.addWidget(self.tab_widget)

        # Create bottom button layout
        bottom_layout = QHBoxLayout()
        
        # Export buttons
        export_button_style = """
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
        """
        
        # Create export buttons
        export_csv_btn = QPushButton("Export CSV")
        export_csv_btn.setStyleSheet(export_button_style)
        export_csv_btn.clicked.connect(lambda: self.export_data('csv'))
        bottom_layout.addWidget(export_csv_btn)
        
        export_excel_btn = QPushButton("Export Excel")
        export_excel_btn.setStyleSheet(export_button_style)
        export_excel_btn.clicked.connect(lambda: self.export_data('excel'))
        bottom_layout.addWidget(export_excel_btn)
        
        export_parquet_btn = QPushButton("Export Parquet")
        export_parquet_btn.setStyleSheet(export_button_style)
        export_parquet_btn.clicked.connect(lambda: self.export_data('parquet'))
        bottom_layout.addWidget(export_parquet_btn)
        
        bottom_layout.addStretch()
        
        # Add Query button
        self.add_query_button = QPushButton("+")
        self.add_query_button.setFixedSize(30, 30)
        self.add_query_button.setStyleSheet(
            "QPushButton { background-color: palette(button); border-radius: 15px; }"
            "QPushButton:hover { background-color: palette(highlight); color: palette(highlighted-text); }"
        )
        self.add_query_button.clicked.connect(self.add_query_tab)
        bottom_layout.addWidget(self.add_query_button)
        
        right_layout.addLayout(bottom_layout)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)  # Left panel (tables)
        splitter.setStretchFactor(1, 5)  # Right panel (query)

        # Create initial tab
        self.add_query_tab()

    def load_database(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Database File", "",
                                                   "Database Files (*.duckdb *.db);;All Files (*)")
        if file_path:
            # Add the database to the list if it's not already there
            items = [self.db_list.item(i).text() for i in range(self.db_list.count())]
            if file_path not in items:
                self.db_list.addItem(file_path)
            
            # Select the newly loaded database
            for i in range(self.db_list.count()):
                if self.db_list.item(i).text() == file_path:
                    self.db_list.setCurrentRow(i)
                    break
            
            # Switch to the selected database
            self.switch_database(self.db_list.currentItem())
            # Emit signal that database was loaded
            self.database_loaded.emit(file_path)

    def toggle_table_list(self):
        if self.table_list.isVisible():
            self.table_list.hide()
            self.collapse_button.setText("▶")
        else:
            self.table_list.show()
            self.collapse_button.setText("◀")

    def close_database(self):
        current_item = self.db_list.currentItem()
        if current_item:
            row = self.db_list.row(current_item)
            self.db_list.takeItem(row)
            
            # Switch to another database if available, otherwise clear the current database
            if self.db_list.count() > 0:
                self.db_list.setCurrentRow(0)
                self.switch_database(self.db_list.currentItem())
            else:
                self.current_db_path = None
                
                # Clear all query tabs
                for i in range(self.tab_widget.count()):
                    tab = self.tab_widget.widget(i)
                    if tab:
                        df_message = pd.DataFrame({'message': ['No database loaded. Please load a database first.']})
                        tab.current_df = df_message
                        model = PandasModel(df_message)
                        tab.table_view.setModel(model)
                        tab.status_label.setText("No database loaded")
                self.table_list.clear()
                self.table_list.setDisabled(True)
                self.database_closed.emit()

    def update_table_list(self):
        try:
            # Clear the current list
            self.table_list.clear()
            
            if not self.current_db_path:
                return
                
            # Connect to the database and get table list
            if self.current_db_path.lower().endswith('.duckdb'):
                conn = duckdb.connect(database=self.current_db_path, read_only=True)
                tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()
            else:
                import sqlite3
                conn = sqlite3.connect(self.current_db_path)
                cursor = conn.cursor()
                tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                
            conn.close()
            
            # Add tables to the list widget
            for table in tables:
                self.table_list.addItem(table[0])
                
            # Enable the table list if there are tables
            self.table_list.setEnabled(len(tables) > 0)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to update table list: {str(e)}")

    def new_query(self):
        # Create a new tab with query editor and result view
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        # Add query editor with improved height
        tab.query_edit = CodeEditor()  # Use CodeEditor instead of QPlainTextEdit
        tab.query_edit.setPlaceholderText("Enter your SQL query here...")
        tab.query_edit.setMinimumHeight(150)  # Set minimum height
        tab_layout.addWidget(tab.query_edit)

        # Add Run Query button with styling
        run_button = QPushButton("Run Query")
        run_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        run_button.clicked.connect(lambda: self.run_query(tab))
        tab_layout.addWidget(run_button)

        # Add table view for results
        tab.table_view = QTableView()
        tab.table_view.setModel(PandasModel())
        tab_layout.addWidget(tab.table_view)

        # Add status label
        tab.status_label = QLabel("Ready")
        tab_layout.addWidget(tab.status_label)

        # Add the tab to the tab widget
        self.tab_widget.addTab(tab, f"Query {self.tab_widget.count() + 1}")
        self.tab_widget.setCurrentWidget(tab)

    def switch_database(self, item):
        # Handle both QListWidgetItem and string inputs
        db_path = item.text() if hasattr(item, 'text') else item
        self.switch_to_database(db_path)

    def switch_to_database(self, db_path):
        if db_path == self.current_db_path:
            return

        try:
            # Connect to the new database without closing the current one
            conn = duckdb.connect(database=db_path, read_only=False)
            tables = conn.execute("SHOW TABLES").fetchall()
            conn.close()

            # Update UI
            self.table_list.clear()
            self.table_list.addItems([table[0] for table in tables])
            self.table_list.setEnabled(True)
            self.current_db_path = db_path
            self.database_loaded.emit(db_path)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to switch database: {str(e)}")
            self.database_closed.emit()

    def close_database(self):
        if self.current_db_path:
            self.table_list.clear()
            self.table_list.setEnabled(False)
            self.current_db_path = None
            self.database_closed.emit()

    def close_tab(self, index):
        if self.tab_widget.count() > 1:  # Keep at least one tab open
            self.tab_widget.removeTab(index)
        else:
            # If it's the last tab, clear it instead of closing
            tab = self.tab_widget.widget(0)
            tab.query_edit.clear()
            tab.current_df = None
            tab.table_view.setModel(PandasModel())

    def run_query(self, tab):
        if not self.current_db_path:
            QMessageBox.warning(self, "Error", "No database loaded. Please load a database first.")
            return

        query = tab.query_edit.toPlainText().strip()
        if not query:
            QMessageBox.warning(self, "Error", "Please enter a query.")
            return

        tab.status_label.setText("Executing query...")
        
        # Create and start the worker thread
        self.worker = QueryWorker(self.current_db_path, query)
        self.worker.resultReady.connect(lambda df: self.handle_query_result(tab, df))
        self.worker.errorOccurred.connect(lambda error: self.handle_query_error(tab, error))
        self.worker.progressUpdate.connect(lambda msg: tab.status_label.setText(msg))
        self.worker.start()

    def handle_query_result(self, tab, df):
        tab.current_df = df
        model = PandasModel(df)
        tab.table_view.setModel(model)
        tab.status_label.setText(f"Query completed successfully. Rows: {len(df)}")

    def handle_query_error(self, tab, error):
        tab.status_label.setText("Error executing query")
        QMessageBox.critical(self, "Error", f"Failed to execute query: {str(error)}")

    def add_query_tab(self):
        # Create a new tab with query editor and result view
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        # Add query editor with improved height
        tab.query_edit = CodeEditor()  # Use CodeEditor instead of QPlainTextEdit
        tab.query_edit.setPlaceholderText("Enter your SQL query here...")
        tab.query_edit.setMinimumHeight(150)  # Set minimum height
        tab_layout.addWidget(tab.query_edit)

        # Add Run Query button with styling
        run_button = QPushButton("Run Query")
        run_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        run_button.clicked.connect(lambda: self.run_query(tab))
        tab_layout.addWidget(run_button)

        # Add table view for results
        tab.table_view = QTableView()
        tab.table_view.setModel(PandasModel())
        tab_layout.addWidget(tab.table_view)

        # Add status label
        tab.status_label = QLabel("Ready")
        tab_layout.addWidget(tab.status_label)

        # Add the tab to the tab widget
        self.tab_widget.addTab(tab, f"Query {self.tab_widget.count() + 1}")
        self.tab_widget.setCurrentWidget(tab)

class CreateDatabaseDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Database")
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Create form layout
        form_layout = QFormLayout()
        
        # Database name field
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter database name")
        form_layout.addRow("Database Name:", self.name_input)
        
        # Database type selection
        self.type_combo = QComboBox()
        self.type_combo.addItem("DuckDB (Recommended for analytics)")
        self.type_combo.addItem("SQLite")
        form_layout.addRow("Database Type:", self.type_combo)
        
        # Location field with browse button
        location_layout = QHBoxLayout()
        self.location_input = QLineEdit()
        self.location_input.setText(os.path.expanduser("~"))  # Default to user's home directory
        location_layout.addWidget(self.location_input)
        
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_location)
        location_layout.addWidget(browse_button)
        
        form_layout.addRow("Location:", location_layout)
        
        layout.addLayout(form_layout)
        
        # Status label for error messages
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: red;")
        layout.addWidget(self.status_label)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.create_database)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def browse_location(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Location", self.location_input.text())
        if folder:
            self.location_input.setText(folder)
            
    def create_database(self):
        # Get values from form
        name = self.name_input.text().strip()
        db_type = self.type_combo.currentText()
        location = self.location_input.text()
        
        # Validate input
        if not name:
            self.status_label.setText("Database name cannot be empty.")
            return
        
        # Determine file extension based on selected type
        extension = ".duckdb" if "DuckDB" in db_type else ".db"
        
        # Make sure the filename has the correct extension
        if not name.endswith(extension):
            name += extension
        
        # Create full path
        path = Path(location) / name
        
        # Check if file already exists
        if path.exists():
            self.status_label.setText(f"A file named '{name}' already exists in this location.")
            return
        
        try:
            # Create the database
            if extension == ".duckdb":
                conn = duckdb.connect(str(path))
                conn.close()
            else:
                import sqlite3
                conn = sqlite3.connect(str(path))
                conn.close()
            
            self.db_path = str(path)
            self.accept()
        except Exception as e:
            self.status_label.setText(f"Error creating database: {str(e)}")

class MergeFilesDialog(QDialog):
    def __init__(self, parent=None, available_databases=None):
        super().__init__(parent)
        self.setWindowTitle("Merge Files into Database")
        self.available_databases = available_databases or []
        self.parent = parent
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Create form layout
        form_layout = QFormLayout()
        
        # Source folder selection
        folder_layout = QHBoxLayout()
        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("Select folder containing files to merge")
        folder_layout.addWidget(self.folder_input)
        
        browse_folder_button = QPushButton("Browse...")
        browse_folder_button.clicked.connect(self.browse_folder)
        folder_layout.addWidget(browse_folder_button)
        
        form_layout.addRow("Source Folder:", folder_layout)
        
        # Target database selection
        self.db_combo = QComboBox()
        
        # Add available databases
        if self.available_databases:
            for db in self.available_databases:
                self.db_combo.addItem(db)
        else:
            self.db_combo.addItem("No database loaded")
            self.db_combo.setEnabled(False)
        
        # Add option to create new database
        self.db_combo.addItem("Create new database...")
        self.db_combo.currentIndexChanged.connect(self.on_database_changed)
        
        form_layout.addRow("Target Database:", self.db_combo)
        
        # Table selection options
        self.table_option_layout = QVBoxLayout()
        
        # Radio buttons for table selection
        self.new_table_radio = QPushButton("Create new table")
        self.new_table_radio.setCheckable(True)
        self.new_table_radio.setChecked(True)
        self.new_table_radio.clicked.connect(self.toggle_table_options)
        
        self.existing_table_radio = QPushButton("Use existing table")
        self.existing_table_radio.setCheckable(True)
        self.existing_table_radio.clicked.connect(self.toggle_table_options)
        
        # Style the buttons to look like radio buttons
        button_style = """
            QPushButton {
                background-color: palette(button);
                border: 1px solid palette(mid);
                border-radius: 4px;
                padding: 4px 8px;
                text-align: left;
            }
            QPushButton:checked {
                background-color: palette(highlight);
                color: palette(highlighted-text);
            }
        """
        self.new_table_radio.setStyleSheet(button_style)
        self.existing_table_radio.setStyleSheet(button_style)
        
        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.new_table_radio)
        button_layout.addWidget(self.existing_table_radio)
        button_layout.addStretch()
        
        self.table_option_layout.addLayout(button_layout)
        
        # New table name input
        self.new_table_layout = QHBoxLayout()
        self.table_name_input = QLineEdit("merged_data")
        self.new_table_layout.addWidget(self.table_name_input)
        self.table_option_layout.addLayout(self.new_table_layout)
        
        # Existing table selection
        self.existing_table_layout = QHBoxLayout()
        self.table_combo = QComboBox()
        self.table_combo.setEnabled(False)
        self.existing_table_layout.addWidget(self.table_combo)
        self.refresh_tables_button = QPushButton("Refresh")
        self.refresh_tables_button.clicked.connect(self.load_tables)
        self.refresh_tables_button.setEnabled(False)
        self.existing_table_layout.addWidget(self.refresh_tables_button)
        self.table_option_layout.addLayout(self.existing_table_layout)
        
        form_layout.addRow("Table Options:", self.table_option_layout)
        
        layout.addLayout(form_layout)
        
        # Status label for error messages
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: red;")
        layout.addWidget(self.status_label)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.validate_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Initialize UI state
        self.toggle_table_options()
        
        # Load tables if a database is selected
        if self.db_combo.currentText() not in ["No database loaded", "Create new database..."]:
            self.load_tables()
    
    def on_database_changed(self, index):
        # Load tables when database selection changes
        selected_db = self.db_combo.currentText()
        if selected_db not in ["No database loaded", "Create new database..."]:
            self.load_tables()
            self.existing_table_radio.setEnabled(True)
            self.refresh_tables_button.setEnabled(True)
        else:
            self.table_combo.clear()
            self.existing_table_radio.setEnabled(False)
            self.refresh_tables_button.setEnabled(False)
            # Force new table option if no database or creating new database
            self.new_table_radio.setChecked(True)
            self.existing_table_radio.setChecked(False)
            self.toggle_table_options()
    
    def load_tables(self):
        selected_db = self.db_combo.currentText()
        if selected_db in ["No database loaded", "Create new database..."]:
            return
            
        try:
            # Connect to the database and get table list
            if selected_db.lower().endswith('.duckdb'):
                conn = duckdb.connect(selected_db, read_only=True)
                tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()
            else:
                import sqlite3
                conn = sqlite3.connect(selected_db)
                cursor = conn.cursor()
                tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            
            conn.close()
            
            # Update the table combo box
            self.table_combo.clear()
            for table in tables:
                self.table_combo.addItem(table[0])
                
            # Enable existing table option if tables exist
            self.existing_table_radio.setEnabled(self.table_combo.count() > 0)
            
        except Exception as e:
            self.status_label.setText(f"Error loading tables: {str(e)}")
    
    def toggle_table_options(self):
        # Show/hide appropriate widgets based on selection
        if self.new_table_radio.isChecked():
            self.table_name_input.setEnabled(True)
            self.table_combo.setEnabled(False)
            self.refresh_tables_button.setEnabled(False)
        else:
            self.table_name_input.setEnabled(False)
            self.table_combo.setEnabled(True)
            self.refresh_tables_button.setEnabled(True)
            
            # Load tables if needed
            if self.table_combo.count() == 0 and self.db_combo.currentText() not in ["No database loaded", "Create new database..."]:
                self.load_tables()
        
    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with Files", self.folder_input.text() or os.path.expanduser("~"))
        if folder:
            self.folder_input.setText(folder)
            
    def validate_and_accept(self):
        # Get values from form
        folder_path = self.folder_input.text().strip()
        selected_db = self.db_combo.currentText()
        
        # Validate input
        if not folder_path:
            self.status_label.setText("Please select a source folder.")
            return
            
        if not os.path.isdir(folder_path):
            self.status_label.setText("Selected path is not a valid directory.")
            return
            
        if selected_db == "No database loaded":
            self.status_label.setText("Please load or create a database first.")
            return
        
        # Get table name based on selection
        if self.new_table_radio.isChecked():
            table_name = self.table_name_input.text().strip()
            if not table_name:
                self.status_label.setText("Table name cannot be empty.")
                return
            self.use_existing_table = False
        else:
            if self.table_combo.count() == 0:
                self.status_label.setText("No tables available. Please create a new table.")
                return
            table_name = self.table_combo.currentText()
            self.use_existing_table = True
        
        # If user selected "Create new database...", open the create database dialog
        if selected_db == "Create new database...":
            create_dialog = CreateDatabaseDialog(self)
            result = create_dialog.exec()
            
            if result == QDialog.DialogCode.Accepted and hasattr(create_dialog, 'db_path'):
                self.db_path = create_dialog.db_path
            else:
                # User cancelled database creation
                return
        else:
            self.db_path = selected_db
        
        # Store the values for later use
        self.folder_path = folder_path
        self.table_name = table_name
        
        # Accept the dialog
        self.accept()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SQL Query Editor")
        
        # Create and set the central widget first
        self.query_tab = QueryTab(self)
        self.setCentralWidget(self.query_tab)
        self.resize(1200, 800)
        
        # Create toolbar
        self.toolbar = self.addToolBar("Database Controls")
        self.toolbar.setMovable(False)
        
        # Create New Database button
        self.create_db_button = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon), "Create Database", self)
        self.create_db_button.setStatusTip("Create New Database")
        self.create_db_button.triggered.connect(self.create_database)
        self.toolbar.addAction(self.create_db_button)
        
        # Database control icons
        self.load_db_button = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton), "Load Database", self)
        self.load_db_button.setStatusTip("Load Database")
        self.load_db_button.triggered.connect(self.load_database)
        self.toolbar.addAction(self.load_db_button)
        
        # Database selection dropdown
        self.db_selector = QComboBox()
        self.db_selector.setMinimumWidth(300)
        self.db_selector.setStyleSheet("padding: 0 10px; background: palette(window); border: 1px solid palette(mid);")
        self.db_selector.addItem("No database loaded")
        self.db_selector.setEnabled(False)
        self.db_selector.currentTextChanged.connect(self.on_database_selected)
        self.toolbar.addWidget(self.db_selector)
        
        # Close database button
        self.close_db_button = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogCloseButton), "Close Database", self)
        self.close_db_button.setStatusTip("Close Database")
        self.close_db_button.setEnabled(False)
        self.close_db_button.triggered.connect(self.close_database)
        self.toolbar.addAction(self.close_db_button)
        
        # Add Merge Files button
        self.merge_files_button = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_DriveFDIcon), "Merge Files", self)
        self.merge_files_button.setStatusTip("Merge files from a folder into database")
        self.merge_files_button.triggered.connect(self.merge_files)
        self.toolbar.addAction(self.merge_files_button)
        
        # Connect signals
        self.query_tab.database_loaded.connect(self.on_database_loaded)
        self.query_tab.database_closed.connect(self.on_database_closed)

    def load_database(self):
        self.query_tab.load_database()

    def on_database_loaded(self, db_path):
        # Update the dropdown
        if self.db_selector.findText(db_path) == -1:
            self.db_selector.addItem(db_path)
        self.db_selector.setCurrentText(db_path)
        self.db_selector.setEnabled(True)
        self.close_db_button.setEnabled(True)

    def on_database_closed(self):
        current_text = self.db_selector.currentText()
        if current_text != "No database loaded":
            self.db_selector.removeItem(self.db_selector.findText(current_text))
        
        if self.db_selector.count() <= 1:
            self.db_selector.setCurrentText("No database loaded")
            self.db_selector.setEnabled(False)
            self.close_db_button.setEnabled(False)

    def on_database_selected(self, db_path):
        if db_path != "No database loaded":
            self.query_tab.switch_database(db_path)

    def close_database(self):
        self.query_tab.close_database()

    def create_database(self):
        """Open the dialog to create a new database"""
        dialog = CreateDatabaseDialog(self)
        result = dialog.exec()
        
        if result == QDialog.DialogCode.Accepted and hasattr(dialog, 'db_path'):
            # Database created successfully, load it
            db_path = dialog.db_path
            
            # Add the database to the selector if it's not already there
            if self.db_selector.findText(db_path) == -1:
                self.db_selector.addItem(db_path)
            
            # Select this database
            self.db_selector.setCurrentText(db_path)
            
            # Set it as the current database
            self.query_tab.switch_to_database(db_path)
            
            # Update UI state
            self.db_selector.setEnabled(True)
            self.close_db_button.setEnabled(True)
            
            # Show success message
            QMessageBox.information(self, "Success", f"Database created successfully at:\n{db_path}")
    
    def merge_files(self):
        """Open dialog to merge files from a folder into a database"""
        # Get list of available databases
        available_dbs = []
        for i in range(self.db_selector.count()):
            db_text = self.db_selector.itemText(i)
            if db_text != "No database loaded":
                available_dbs.append(db_text)
        
        # Create and show the merge files dialog
        dialog = MergeFilesDialog(self, available_dbs)
        result = dialog.exec()
        
        if result == QDialog.DialogCode.Accepted:
            # Get the values from the dialog
            folder_path = dialog.folder_path
            db_path = dialog.db_path
            table_name = dialog.table_name
            use_existing_table = dialog.use_existing_table
            
            # Create a progress dialog
            progress_dialog = QProgressDialog("Merging files...", "Cancel", 0, 0, self)
            progress_dialog.setWindowTitle("Merge Progress")
            progress_dialog.setMinimumDuration(0)
            progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            progress_dialog.setAutoClose(False)
            progress_dialog.setAutoReset(False)
            
            # Create the worker
            self.merge_worker = MergeFilesWorker(folder_path, db_path, table_name, use_existing_table)
            
            # Create a thread to run the worker
            self.merge_thread = QThread()
            self.merge_worker.moveToThread(self.merge_thread)
            
            # Connect signals
            self.merge_worker.progress.connect(progress_dialog.setLabelText)
            self.merge_worker.error.connect(lambda msg: QMessageBox.critical(self, "Error", msg))
            self.merge_worker.success.connect(lambda msg: QMessageBox.information(self, "Success", msg))
            self.merge_worker.finished.connect(progress_dialog.close)
            self.merge_worker.finished.connect(self.merge_thread.quit)
            
            # Connect table created signal to refresh tables
            self.merge_worker.table_created.connect(self.on_table_created)
            
            # Connect thread signals
            self.merge_thread.started.connect(self.merge_worker.run)
            self.merge_thread.finished.connect(self.merge_thread.deleteLater)
            
            # Connect cancel button
            progress_dialog.canceled.connect(self.merge_worker.cancel)
            
            # Start the thread
            progress_dialog.show()
            self.merge_thread.start()
            
            # If this is a new database, add it to the selector
            if self.db_selector.findText(db_path) == -1:
                self.db_selector.addItem(db_path)
                self.db_selector.setCurrentText(db_path)
                self.db_selector.setEnabled(True)
                self.close_db_button.setEnabled(True)
                self.query_tab.switch_to_database(db_path)
    
    def on_table_created(self, db_path, table_name):
        """Handle table created/updated signal"""
        # Make sure the database is selected
        if self.db_selector.currentText() != db_path:
            self.db_selector.setCurrentText(db_path)
        
        # Refresh the table list
        self.query_tab.update_table_list()
        
        # Select the table in the list if it exists
        for i in range(self.query_tab.table_list.count()):
            if self.query_tab.table_list.item(i).text() == table_name:
                self.query_tab.table_list.setCurrentRow(i)
                break

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
