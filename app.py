import re
import sys
import duckdb
import pandas as pd
import os
import tempfile
import shutil
from pathlib import Path
import time
import datetime
import traceback

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLineEdit, QFileDialog, QTableView, QMessageBox, QLabel, QPlainTextEdit, QCompleter,
                             QListWidget, QSplitter, QTabWidget, QProgressDialog, QStyle, QComboBox, QMenuBar, QDialog, QDialogButtonBox,
                             QComboBox, QFormLayout, QMenu, QCheckBox, QRadioButton, QButtonGroup, QInputDialog)
from PyQt6.QtCore import QAbstractTableModel, Qt, QThread, pyqtSignal, QRegularExpression, QTimer, QObject
from PyQt6.QtGui import QTextCursor, QSyntaxHighlighter, QTextCharFormat, QAction, QFont, QFontMetrics


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
        
        # Set font to a monospaced font
        font = QFont("Consolas" if sys.platform == "win32" else "Courier")
        font.setPointSize(10)
        self.setFont(font)
        
        # Set tab width
        metrics = QFontMetrics(font)
        self.setTabStopDistance(4 * metrics.horizontalAdvance(' '))
        
        # Create and set the highlighter
        self.highlighter = SQLHighlighter(self.document())
        
        # Set styling
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
        
        # Auto-indentation and auto-closing quotes
        self.auto_close_chars = {
            "'": "'",
            '"': '"',
            '(': ')',
            '[': ']',
            '{': '}'
        }
    
    def keyPressEvent(self, event):
        # Handle auto-closing quotes and brackets
        if event.text() in self.auto_close_chars:
            cursor = self.textCursor()
            # Get the opening character
            opening_char = event.text()
            # Get the corresponding closing character
            closing_char = self.auto_close_chars[opening_char]
            
            # Insert both characters and position cursor between them
            cursor.insertText(opening_char + closing_char)
            cursor.movePosition(QTextCursor.MoveOperation.Left, QTextCursor.MoveMode.MoveAnchor, 1)
            self.setTextCursor(cursor)
            return
        # Handle auto-indentation for Enter key
        elif event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            # Get the current line
            cursor = self.textCursor()
            block = cursor.block()
            text = block.text()
            
            # Find the indentation level
            indentation = ""
            for char in text:
                if char.isspace():
                    indentation += char
                else:
                    break
            
            # Call the parent implementation to insert a new line
            super().keyPressEvent(event)
            
            # Insert the indentation
            if indentation:
                self.textCursor().insertText(indentation)
            return
        # Handle backspace to delete matching closing character
        elif event.key() == Qt.Key.Key_Backspace:
            cursor = self.textCursor()
            if not cursor.hasSelection():
                cursor_pos = cursor.position()
                if cursor_pos > 0:
                    # Check if we're between an opening and closing character
                    cursor.movePosition(QTextCursor.MoveOperation.Left, QTextCursor.MoveMode.KeepAnchor, 1)
                    opening_char = cursor.selectedText()
                    cursor.movePosition(QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.MoveAnchor, 1)
                    
                    if opening_char in self.auto_close_chars:
                        cursor.movePosition(QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.KeepAnchor, 1)
                        closing_char = cursor.selectedText()
                        
                        if closing_char == self.auto_close_chars[opening_char]:
                            # Delete both characters
                            cursor.movePosition(QTextCursor.MoveOperation.Left, QTextCursor.MoveMode.MoveAnchor, 1)
                            cursor.movePosition(QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.KeepAnchor, 2)
                            cursor.removeSelectedText()
                            return
        
        # For all other keys, use the default behavior
        super().keyPressEvent(event)

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
        self.elapsed_time = 0
        self.timer = None
    
    def update_elapsed_time(self):
        self.elapsed_time += 1
        self.progressUpdate.emit(f"Executing query... ({self.elapsed_time}s)")
    
    def cleanup_timer(self):
        if self.timer and self.timer.isActive():
            self.timer.stop()
            self.timer.deleteLater()
            self.timer = None
    
    def clean_query(self, query):
        """Clean up the query to prevent syntax errors"""
        # Remove trailing semicolons
        q = query.strip()
        if q.endswith(';'):
            q = q[:-1]
        
        # For DuckDB, we need to be careful with quoted identifiers
        # The safest approach is to use unquoted identifiers for table names
        # This regex finds FROM clauses with quoted table names and removes the quotes
        q = re.sub(r'FROM\s+[\'"]([^\'"]+)[\'"]', r'FROM \1', q, flags=re.IGNORECASE)
        
        return q
    
    def start_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_elapsed_time)
        self.timer.start(1000)  # Update every second
    
    def process_chunk(self, chunk):
        return pd.DataFrame(chunk)
    
    def run(self):
        try:
            start_time = time.time()
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_elapsed_time)
            self.timer.start(1000)  # Update every second
            
            self.progressUpdate.emit("Executing query...")
            
            # Determine database type from file extension
            is_duckdb = self.db_path.lower().endswith('.duckdb')
            
            # Process the query - clean it up to prevent syntax errors
            original_query = self.query.strip()
            
            # DIRECT FIX FOR SINGLE QUOTES ISSUE
            # Check if this is a simple query with single quotes around table name
            single_quote_pattern = re.search(r"FROM\s+'([^']+)'", original_query, re.IGNORECASE)
            if is_duckdb and single_quote_pattern:
                table_name = single_quote_pattern.group(1)
                self.progressUpdate.emit(f"Detected single quotes around table name '{table_name}', fixing...")
                
                # Replace single quotes with no quotes for DuckDB
                fixed_query = re.sub(r"FROM\s+'([^']+)'", r"FROM \1", original_query, flags=re.IGNORECASE)
                
                try:
                    # Connect to the database
                    import duckdb
                    conn = duckdb.connect(database=self.db_path, read_only=False)
                    
                    # Try the fixed query
                    self.progressUpdate.emit(f"Executing query with unquoted table name...")
                    result = conn.execute(fixed_query).fetchdf()
                    
                    # Process the result
                    if result is not None and not result.empty:
                        self.resultReady.emit(result)
                    else:
                        self.progressUpdate.emit("Query executed successfully, but returned no results.")
                        self.resultReady.emit(pd.DataFrame())
                    
                    # Calculate execution time
                    execution_time = time.time() - start_time
                    self.progressUpdate.emit(f"Query executed in {execution_time:.2f} seconds.")
                    return
                except Exception as e:
                    self.progressUpdate.emit(f"Fixed query failed: {str(e)}")
                    # Continue with standard approach
            
            # Standard query processing
            q = self.clean_query(self.query)
            
            if is_duckdb:
                # For DuckDB
                import duckdb
                
                # DIRECT APPROACH: For simple SELECT queries, bypass SQL parsing entirely
                # First, check if this is a simple SELECT * FROM query
                simple_select_match = re.match(r'^\s*SELECT\s+\*\s+FROM\s+(?:[\'"]([^\'"]+)[\'"]|([^\s;]+))\s*;?\s*$', original_query, re.IGNORECASE)
                
                if simple_select_match:
                    # Get the table name (either quoted or unquoted)
                    table_name = simple_select_match.group(1) if simple_select_match.group(1) else simple_select_match.group(2)
                    self.progressUpdate.emit(f"Using direct table access for '{table_name}'...")
                    
                    try:
                        # Connect to the database
                        conn = duckdb.connect(database=self.db_path, read_only=False)
                        
                        # First check if the table exists
                        tables = conn.execute("SHOW TABLES").fetchall()
                        table_exists = False
                        actual_table_name = ""
                        
                        # Case-insensitive table name matching
                        for t in tables:
                            if t[0].lower() == table_name.lower():
                                table_exists = True
                                actual_table_name = t[0]  # Use the actual case of the table name
                                break
                        
                        if table_exists:
                            # Use the actual table name with the correct case
                            # IMPORTANT: Use the unquoted table name directly
                            self.progressUpdate.emit(f"Table '{actual_table_name}' found, executing query...")
                            result = conn.execute(f"SELECT * FROM {actual_table_name}").fetchdf()
                            
                            # Process the result
                            if result is not None and not result.empty:
                                self.resultReady.emit(result)
                            else:
                                self.progressUpdate.emit("Query executed successfully, but returned no results.")
                                self.resultReady.emit(pd.DataFrame())
                            
                            # Calculate execution time
                            execution_time = time.time() - start_time
                            self.progressUpdate.emit(f"Query executed in {execution_time:.2f} seconds.")
                            return
                        else:
                            self.progressUpdate.emit(f"Table '{table_name}' not found, trying alternative approaches...")
                    except Exception as e:
                        self.progressUpdate.emit(f"Direct approach failed: {str(e)}")
                        # Continue with standard approaches
                
                # If we get here, either it's not a simple query or the direct approach failed
                # Try standard approaches with various quote handling
                conn = duckdb.connect(database=self.db_path, read_only=False)
                conn.execute("PRAGMA sqlite_extension_functions.enable=TRUE")
                
                # Try multiple approaches in sequence
                approaches = [
                    # 1. Try with the original query
                    {"query": original_query, "description": "original query"},
                    # 2. Try with no quotes around table names
                    {"query": re.sub(r'FROM\s+[\'"]([^\'"]+)[\'"]', r'FROM \1', original_query, flags=re.IGNORECASE), 
                     "description": "no quotes around table names"},
                    # 3. Try with double quotes
                    {"query": re.sub(r"FROM\s+'([^']+)'", r'FROM "\1"', original_query, flags=re.IGNORECASE), 
                     "description": "double quotes around table names"},
                    # 4. Try with backticks
                    {"query": re.sub(r'FROM\s+[\'"]([^\'"]+)[\'"]', r'FROM `\1`', original_query, flags=re.IGNORECASE), 
                     "description": "backticks around table names"}
                ]
                
                # Try each approach in sequence
                for approach in approaches:
                    try:
                        self.progressUpdate.emit(f"Trying with {approach['description']}...")
                        result = conn.execute(approach['query']).fetchdf()
                        
                        # If we get here, the query succeeded
                        self.progressUpdate.emit(f"Query succeeded with {approach['description']}.")
                        
                        # Process the result
                        if result is not None and not result.empty:
                            self.resultReady.emit(result)
                        else:
                            self.progressUpdate.emit("Query executed successfully, but returned no results.")
                            self.resultReady.emit(pd.DataFrame())
                        
                        # Calculate execution time
                        execution_time = time.time() - start_time
                        self.progressUpdate.emit(f"Query executed in {execution_time:.2f} seconds.")
                        return
                    except Exception as e:
                        self.progressUpdate.emit(f"Approach with {approach['description']} failed: {str(e)}")
                        # Continue to the next approach
                
                # If all approaches failed, report the error
                self.errorOccurred.emit("All query approaches failed. Try removing quotes around table names or check if the table exists.")
                self.cleanup_timer()
                return
            else:
                # For SQLite
                import sqlite3
                conn = sqlite3.connect(self.db_path)
                
                try:
                    # Execute the query
                    result = pd.read_sql_query(q, conn)
                except Exception as e:
                    self.errorOccurred.emit(f"Error executing query: {str(e)}")
                    self.cleanup_timer()
                    return
            
            # Process the result
            if result is not None and not result.empty:
                self.resultReady.emit(result)
            else:
                self.progressUpdate.emit("Query executed successfully, but returned no results.")
                # Emit an empty DataFrame so the UI can update
                self.resultReady.emit(pd.DataFrame())
            
            # Calculate execution time
            execution_time = time.time() - start_time
            self.progressUpdate.emit(f"Query executed in {execution_time:.2f} seconds.")
            
        except Exception as e:
            self.errorOccurred.emit(f"Unexpected error: {str(e)}")
        finally:
            self.cleanup_timer()

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
    
    CHUNK_SIZE = 50000  # Reduced chunk size for better memory management
    
    def __init__(self, folder_path, db_path, table_name=None, use_existing_table=False, replace_table=False):
        super().__init__()
        self.folder_path = folder_path
        self.db_path = db_path
        self.table_name = table_name or "merged_data"
        self.use_existing_table = use_existing_table
        self.replace_table = replace_table
        self.is_cancelled = False
        self.is_duckdb = db_path.lower().endswith('.duckdb')
        self.total_rows_processed = 0
    
    def clean_column_names(self, columns):
        """Clean and deduplicate column names"""
        # Convert columns to list of strings
        cols = [str(col) if col is not None else f"Column_{i+1}" for i, col in enumerate(columns)]
        
        # Replace empty strings with placeholder names
        for i in range(len(cols)):
            if cols[i].strip() == '':
                cols[i] = f"Column_{i+1}"
        
        # Handle duplicate column names
        seen = {}
        for i in range(len(cols)):
            col = cols[i]
            if col in seen:
                seen[col] += 1
                cols[i] = f"{col}_{seen[col]}"
            else:
                seen[col] = 0
                
        return cols
        
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
                    column_types[col] = 'BIGINT'
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
                result = conn.execute(f"PRAGMA table_info(\"{self.table_name}\")").fetchall()
                return [row[1] for row in result]  # Column name is at index 1
            else:
                import sqlite3
                cursor = conn.cursor()
                result = cursor.execute(f"PRAGMA table_info(\"{self.table_name}\")").fetchall()
                return [row[1] for row in result]  # Column name is at index 1
        except Exception as e:
            self.progress.emit(f"Error getting existing columns: {str(e)}")
            return []
    
    def add_missing_columns(self, conn, sample_df, existing_columns):
        """Add missing columns to the existing table"""
        new_columns = []
        column_types = self.analyze_column_types(sample_df)
        
        # Create a mapping of lowercase column names to actual column names to check for case-insensitive duplicates
        existing_columns_lower = {col.lower(): col for col in existing_columns}
        
        for col in sample_df.columns:
            # Check if column already exists (case-insensitive)
            if col.lower() not in existing_columns_lower:
                new_columns.append((col, column_types[col]))
        
        if not new_columns:
            return
            
        self.progress.emit(f"Adding {len(new_columns)} new columns to the table...")
        
        for col_name, col_type in new_columns:
            added = False
            errors = []
            
            # Try different quoting styles for column names
            quoting_styles = [
                f'"{col_name}"',  # Double quotes
                f'[{col_name}]',  # Square brackets
                f'`{col_name}`',  # Backticks
                col_name          # No quotes
            ]
            
            for quoted_col in quoting_styles:
                if added:
                    break
                    
                try:
                    if self.is_duckdb:
                        conn.execute(f"ALTER TABLE \"{self.table_name}\" ADD COLUMN {quoted_col} {col_type}")
                        added = True
                    else:
                        cursor = conn.cursor()
                        cursor.execute(f"ALTER TABLE \"{self.table_name}\" ADD COLUMN {quoted_col} {col_type}")
                        conn.commit()
                        added = True
                except Exception as e:
                    # If column already exists with a different case, skip it
                    if "duplicate column name" in str(e).lower():
                        self.progress.emit(f"Column \"{col_name}\" already exists with a different case, skipping...")
                        added = True  # Consider it added since it exists
                        break
                    else:
                        errors.append(f"Error with {quoted_col}: {str(e)}")
            
            if not added:
                # Just log the error but don't stop processing
                self.progress.emit(f"Could not add column {col_name}. Errors: {'; '.join(errors)}")
                
        self.progress.emit(f"Added new columns to the table.")
        
    def insert_data(self, conn, df):
        """Insert data into the database"""
        try:
            if df is None:
                # This is the case when using DuckDB and the DataFrame is already registered
                if self.is_duckdb:
                    try:
                        # Try with double quotes first
                        conn.execute(f"INSERT INTO \"{self.table_name}\" SELECT * FROM df")
                    except Exception as e:
                        if "Parser Error" in str(e):
                            # Try without quotes
                            conn.execute(f"INSERT INTO {self.table_name} SELECT * FROM df")
                return
                
            # Get existing columns to ensure proper column alignment
            existing_columns = self.get_existing_columns(conn)
            self.insert_data_with_column_matching(conn, df, existing_columns)
        except Exception as e:
            self.progress.emit(f"Error inserting data: {str(e)}. Will try to continue with next chunk.")
            # Don't re-raise the exception so processing can continue
    
    def insert_data_with_column_matching(self, conn, df, existing_columns):
        """Insert data with column matching to handle different column sets"""
        try:
            self.progress.emit(f"Aligning columns with existing table structure...")
            
            # Create sets for easier comparison
            df_columns = set(df.columns)
            table_columns = set(existing_columns)
            
            # Find columns in DataFrame that aren't in the table
            # (This shouldn't happen as we've already added missing columns, but just in case)
            missing_in_table = df_columns - table_columns
            if missing_in_table:
                self.progress.emit(f"Found {len(missing_in_table)} columns in data that aren't in the table. Adding them...")
                self.add_missing_columns(conn, df[list(missing_in_table)], existing_columns)
                # Update existing columns
                existing_columns = self.get_existing_columns(conn)
                table_columns = set(existing_columns)
            
            # Find columns in the table that aren't in the DataFrame
            missing_in_df = table_columns - df_columns
            if missing_in_df:
                self.progress.emit(f"Found {len(missing_in_df)} columns in table that aren't in the data. Will fill with NULL values.")
                # Add missing columns to DataFrame with NULL values
                for col in missing_in_df:
                    df.loc[:, col] = None
            
            # Create a new DataFrame that contains all columns from the table
            # This ensures we maintain column order and include all columns
            ordered_df = pd.DataFrame()
            for col in existing_columns:
                if col in df.columns:
                    ordered_df.loc[:, col] = df[col]
                else:
                    ordered_df.loc[:, col] = None
            
            if len(ordered_df.columns) == 0:
                self.progress.emit("Warning: No valid columns found for insertion. Skipping this chunk.")
                return
            
            # Now insert the data
            if self.is_duckdb:
                # For DuckDB, register the aligned DataFrame
                conn.register('aligned_df', ordered_df)
                try:
                    # Try with double quotes first
                    conn.execute(f"INSERT INTO \"{self.table_name}\" SELECT * FROM aligned_df")
                except Exception as e:
                    if "Parser Error" in str(e):
                        # Try without quotes
                        conn.execute(f"INSERT INTO {self.table_name} SELECT * FROM aligned_df")
            else:
                # For SQLite, use pandas to_sql
                try:
                    ordered_df.to_sql(self.table_name, conn, if_exists='append', index=False)
                except Exception as e:
                    # If there's an error, try to get the actual columns from the database
                    # and filter the DataFrame to only include those columns
                    self.progress.emit(f"Error inserting data: {str(e)}. Trying with exact column matching...")
                    cursor = conn.cursor()
                    cursor.execute(f"PRAGMA table_info('{self.table_name}')")
                    db_columns = [row[1] for row in cursor.fetchall()]
                    
                    # Create a new DataFrame with exactly the columns from the database
                    final_df = pd.DataFrame()
                    for col in db_columns:
                        if col in ordered_df.columns:
                            final_df.loc[:, col] = ordered_df[col]
                        else:
                            final_df.loc[:, col] = None
                    
                    # Try to insert the precisely matched DataFrame
                    if len(final_df.columns) > 0:
                        final_df.to_sql(self.table_name, conn, if_exists='append', index=False)
                    else:
                        raise Exception("No matching columns found between DataFrame and database table")
            
            self.progress.emit(f"Inserted {len(df)} rows with column alignment.")
        except Exception as e:
            self.progress.emit(f"Error in column matching: {str(e)}. Will try to continue with next chunk.")
            # Don't re-raise the exception so processing can continue
    
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
                # Increase memory limit for DuckDB
                try:
                    conn.execute("SET memory_limit='8GB'")
                    conn.execute("PRAGMA threads=4")  # Use multiple threads for better performance
                except Exception as e:
                    self.progress.emit(f"Notice: Could not set memory limits: {str(e)}")
            else:
                import sqlite3
                conn = sqlite3.connect(self.db_path, timeout=300)  # Increase timeout to 5 minutes
                conn.execute("PRAGMA journal_mode=WAL")  # Use WAL mode for better performance
                conn.execute("PRAGMA synchronous=NORMAL")  # Reduce synchronous mode for better performance
            
            # First collect schema information from all files to create a comprehensive schema
            self.progress.emit("Analyzing schemas from all files to create a unified structure...")
            all_columns = set()
            all_column_types = {}
            sample_df = None
            
            # Sample up to 5 files to get a comprehensive schema
            sample_files = files[:min(5, len(files))]
            for file in sample_files:
                try:
                    # Read a small sample to determine schema
                    if file.suffix.lower() == '.csv':
                        temp_df = pd.read_csv(file, nrows=100)
                    elif file.suffix.lower() in ['.xlsx', '.xls']:
                        temp_df = pd.read_excel(file, nrows=100)
                    else:  # parquet
                        temp_df = pd.read_parquet(file)
                        if len(temp_df) > 100:
                            temp_df = temp_df.iloc[:100]
                    
                    # Clean column names
                    temp_df.columns = self.clean_column_names(temp_df.columns)
                    
                    # Update column info
                    all_columns.update(temp_df.columns)
                    
                    # Remember one sample for table creation
                    if sample_df is None:
                        sample_df = temp_df
                    
                    # Get column types
                    file_types = self.analyze_column_types(temp_df)
                    for col, col_type in file_types.items():
                        # Update column type if needed - prefer more general types
                        if col not in all_column_types:
                            all_column_types[col] = col_type
                        elif all_column_types[col] != col_type:
                            # If we have different types for the same column, use TEXT as a fallback
                            all_column_types[col] = 'TEXT'
                except Exception as e:
                    self.progress.emit(f"Warning: Could not analyze schema from {file.name}: {str(e)}")
            
            if sample_df is None:
                self.error.emit("Could not read any data from the files to determine schema.")
                self.finished.emit()
                conn.close()
                return
            
            # Create a comprehensive sample_df with all found columns
            for col in all_columns:
                if col not in sample_df.columns:
                    sample_df.loc[:, col] = None
            
            # Process the first file to get the schema
            self.progress.emit(f"Setting up table structure with all detected columns...")
            
            # Store for later use
            table_name = self.table_name
            db_path = self.db_path
            
            try:
                # Check if we're using an existing table
                if self.use_existing_table:
                    # Get existing columns
                    existing_columns = self.get_existing_columns(conn)
                    
                    # Add any missing columns
                    self.add_missing_columns(conn, sample_df, existing_columns)
                else:
                    # If replacing, drop the existing table first
                    if self.replace_table:
                        self.progress.emit(f"Replacing existing table '{self.table_name}'...")
                        try:
                            conn.execute(f"DROP TABLE IF EXISTS \"{self.table_name}\"")
                        except Exception as e:
                            self.error.emit(f"Error dropping existing table: {str(e)}")
                            self.finished.emit()
                            conn.close()  # Close connection on error
                            return
                    
                    # Create the table if it doesn't exist with all detected columns
                    self.progress.emit(f"Creating table '{self.table_name}' with {len(all_columns)} columns")
                    # Modify the sample_df to include all columns with proper types
                    self.create_table_with_schema(conn, sample_df, all_column_types)
            except Exception as e:
                self.error.emit(f"Error setting up table structure: {str(e)}")
                self.finished.emit()
                conn.close()  # Close connection on error
                return
            
            # Process all files
            total_files = len(files)
            files_processed = 0
            files_with_errors = 0
            self.total_rows_processed = 0
            
            # Start a transaction for better performance
            if not self.is_duckdb:
                conn.execute("BEGIN TRANSACTION")
                
            for i, file_path in enumerate(files, 1):
                if self.is_cancelled:
                    break
                    
                self.progress.emit(f"Processing file {i}/{total_files}: {file_path.name}")
                
                # Process the file in chunks
                chunk_count = 0
                rows_processed = 0
                file_error = False
                
                try:
                    for chunk in self.read_file_in_chunks(file_path):
                        if self.is_cancelled:
                            break
                            
                        chunk_count += 1
                        rows_processed += len(chunk)
                        self.total_rows_processed += len(chunk)
                        
                        try:
                            # Always check for and add any new columns for each chunk
                            existing_columns = self.get_existing_columns(conn)
                            self.add_missing_columns(conn, chunk, existing_columns)
                            
                            # Get updated list of columns
                            existing_columns = self.get_existing_columns(conn)
                            
                            # Insert the data with proper column matching
                            self.insert_data_with_column_matching(conn, chunk, existing_columns)
                            
                            self.progress.emit(f"Processed {rows_processed} rows from {file_path.name} (chunk {chunk_count}, total rows: {self.total_rows_processed})")
                            
                            # Commit intermediate transaction every 10 chunks for SQLite
                            if not self.is_duckdb and chunk_count % 10 == 0:
                                conn.commit()
                                conn.execute("BEGIN TRANSACTION")
                                self.progress.emit("Intermediate commit completed")
                                
                        except Exception as e:
                            self.error.emit(f"Error processing chunk {chunk_count} from file {file_path.name}: {str(e)}")
                            file_error = True
                            # Continue with next chunk
                            continue
                except Exception as e:
                    self.error.emit(f"Error reading file {file_path}: {str(e)}")
                    file_error = True
                    # Continue with next file
                    continue
                finally:
                    if file_error:
                        files_with_errors += 1
                    else:
                        files_processed += 1
                    
                    self.progress.emit(f"Completed file {i}/{total_files}: {file_path.name}" + 
                                      (" with errors" if file_error else ""))
            
            # Commit the final transaction for SQLite
            if not self.is_duckdb and not self.is_cancelled:
                try:
                    conn.commit()
                    self.progress.emit("Final commit completed")
                except Exception as e:
                    self.error.emit(f"Error committing final transaction: {str(e)}")
                    conn.rollback()
                    
            # Close the database connection
            conn.close()
            self.progress.emit("Database connection closed")
            
            if not self.is_cancelled:
                if files_with_errors > 0:
                    self.success.emit(f"Merged {files_processed} files into {self.db_path}. {files_with_errors} files had errors but were partially processed. Total rows processed: {self.total_rows_processed}")
                else:
                    self.success.emit(f"Successfully merged {files_processed} files into {self.db_path}. Total rows processed: {self.total_rows_processed}")
                    
                # Give the connection some time to fully close before emitting table_created
                QThread.msleep(200)  # Brief delay to ensure connection is fully closed
                # Emit signal that table was created/updated - after connection is closed
                self.table_created.emit(db_path, table_name)
                
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(f"Error merging files: {str(e)}")
            self.finished.emit()
            
    def create_table_with_schema(self, conn, sample_df, column_types):
        """Create table with comprehensive schema from all files"""
        # Build CREATE TABLE statement
        columns_sql = []
        
        # Use column types from our analysis
        for col in sample_df.columns:
            col_type = column_types.get(col, 'TEXT')  # Default to TEXT if no type info
            columns_sql.append(f'"{col}" {col_type}')
            
        create_table_sql = f"CREATE TABLE IF NOT EXISTS \"{self.table_name}\" ({', '.join(columns_sql)})"
        
        # Execute the statement
        if self.is_duckdb:
            conn.execute(create_table_sql)
        else:
            cursor = conn.cursor()
            cursor.execute(create_table_sql)
            conn.commit()
            
    def cancel(self):
        self.is_cancelled = True
        self.progress.emit("Cancelling operation...")
    
    def read_file_in_chunks(self, file_path):
        """Read file in chunks to avoid memory issues"""
        # Convert Path object to string if needed
        file_path_str = str(file_path)
        file_ext = file_path.suffix.lower() if hasattr(file_path, 'suffix') else os.path.splitext(file_path_str)[1].lower()
        
        try:
            if file_ext == '.csv':
                # Try with default settings first
                try:
                    # Try to count total lines for progress reporting
                    try:
                        with open(file_path_str, 'r') as f:
                            total_lines = sum(1 for _ in f)
                        self.progress.emit(f"CSV file has approximately {total_lines} lines")
                    except Exception:
                        pass  # Ignore if we can't count lines
                        
                    # Process with reasonable chunk size
                    for chunk in pd.read_csv(file_path_str, chunksize=self.CHUNK_SIZE):
                        # Clean column names
                        chunk.columns = self.clean_column_names(chunk.columns)
                        # Remove rows with all NaN values
                        chunk = chunk.dropna(how='all')
                        yield chunk
                except Exception as e:
                    self.progress.emit(f"Error reading CSV with default settings, trying more flexible settings: {str(e)}")
                    # Try with more flexible settings
                    for chunk in pd.read_csv(file_path_str, chunksize=self.CHUNK_SIZE, 
                                          encoding='latin1', on_bad_lines='skip', 
                                          low_memory=False, dtype=str):
                        # Clean column names
                        chunk.columns = self.clean_column_names(chunk.columns)
                        # Remove rows with all NaN values
                        chunk = chunk.dropna(how='all')
                        yield chunk
                        
            elif file_ext in ['.xlsx', '.xls']:
                # For Excel, we need to read the entire file at once
                # but we can process it in chunks
                
                # First, check if the file is actually a valid file and exists
                if not os.path.exists(file_path_str):
                    raise FileNotFoundError(f"Excel file not found: {file_path_str}")
                
                # Check file size - if it's too small it's probably corrupt or empty
                file_size = os.path.getsize(file_path_str)
                if file_size < 100:  # Extremely small for an Excel file
                    raise ValueError(f"Excel file is likely corrupt or empty (size: {file_size} bytes)")
                
                # Try multiple engines with proper error handling
                df = None
                success = False
                error_messages = []
                
                # Try with openpyxl first for .xlsx
                if file_ext == '.xlsx':
                    try:
                        self.progress.emit(f"Trying to read Excel file with openpyxl...")
                        df = pd.read_excel(file_path_str, engine='openpyxl')
                        success = True
                    except Exception as e:
                        error_messages.append(f"openpyxl error: {str(e)}")
                        self.progress.emit(f"Error with openpyxl: {str(e)}")
                
                # If .xls or openpyxl failed, try xlrd
                if not success:
                    try:
                        self.progress.emit(f"Trying to read Excel file with xlrd...")
                        df = pd.read_excel(file_path_str, engine='xlrd')
                        success = True
                    except Exception as e:
                        error_messages.append(f"xlrd error: {str(e)}")
                        self.progress.emit(f"Error with xlrd: {str(e)}")
                
                # If still no success, try as CSV as a last resort
                if not success:
                    try:
                        self.progress.emit("Trying to read file as CSV instead...")
                        df = pd.read_csv(file_path_str)
                        success = True
                    except Exception as e:
                        error_messages.append(f"CSV fallback error: {str(e)}")
                        self.progress.emit(f"CSV fallback failed: {str(e)}")
                
                if not success:
                    raise Exception(f"Could not read Excel file with any available engine: {' | '.join(error_messages)}")
                
                # Clean column names
                df.columns = self.clean_column_names(df.columns)
                
                # Remove rows with all NaN values
                df = df.dropna(how='all')
                
                # Process in chunks
                total_rows = len(df)
                self.progress.emit(f"Successfully read Excel file with {total_rows} rows. Processing in chunks...")
                for i in range(0, total_rows, self.CHUNK_SIZE):
                    end = min(i + self.CHUNK_SIZE, total_rows)
                    yield df.iloc[i:end]
                    
            elif file_ext == '.parquet':
                # Try using pyarrow for chunked reading
                try:
                    import pyarrow.parquet as pq
                    
                    # Open the file
                    parquet_file = pq.ParquetFile(file_path_str)
                    
                    # Get metadata
                    total_rows = parquet_file.metadata.num_rows
                    num_row_groups = parquet_file.num_row_groups
                    
                    self.progress.emit(f"Parquet file has {total_rows} rows in {num_row_groups} row groups")
                    
                    # Process each row group
                    for i in range(num_row_groups):
                        # Read row group
                        table = parquet_file.read_row_group(i)
                        
                        # Convert to pandas DataFrame
                        chunk = table.to_pandas()
                        
                        # Clean column names
                        chunk.columns = self.clean_column_names(chunk.columns)
                        
                        # Remove rows with all NaN values
                        chunk = chunk.dropna(how='all')
                        
                        # Further split if the row group is larger than chunk size
                        total_group_rows = len(chunk)
                        
                        for j in range(0, total_group_rows, self.CHUNK_SIZE):
                            end = min(j + self.CHUNK_SIZE, total_group_rows)
                            yield chunk.iloc[j:end]
                        
                except ImportError:
                    # Fall back to pandas if pyarrow is not available
                    self.progress.emit("PyArrow not available, falling back to pandas for Parquet reading")
                    df = pd.read_parquet(file_path_str)
                    
                    # Clean column names
                    df.columns = self.clean_column_names(df.columns)
                    
                    # Remove rows with all NaN values
                    df = df.dropna(how='all')
                    
                    # Process in chunks
                    total_rows = len(df)
                    for i in range(0, total_rows, self.CHUNK_SIZE):
                        end = min(i + self.CHUNK_SIZE, total_rows)
                        yield df.iloc[i:end]
            else:
                self.progress.emit(f"Unsupported file format: {file_ext}")
                raise ValueError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            self.error.emit(f"Error reading file {file_path}: {str(e)}")
            raise  # Re-raise to be caught by the calling function

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
        
        # Add help button
        self.help_button = QPushButton("?")
        self.help_button.setFixedSize(20, 20)
        self.help_button.setToolTip("Show SQL query examples")
        self.help_button.clicked.connect(self.show_query_examples)
        self.help_button.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        header_layout.addWidget(self.help_button)
        
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
        # Enable context menu for table list
        self.table_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table_list.customContextMenuRequested.connect(self.show_table_context_menu)
        # Connect double-click handler
        self.table_list.itemDoubleClicked.connect(self.on_table_double_clicked)
        
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
            
            # Add a small delay to allow any existing connection to be fully closed
            QApplication.processEvents()
                
            # Connect to the database and get table list
            if self.current_db_path.lower().endswith('.duckdb'):
                # Use a more consistent connection approach for DuckDB
                try:
                    # Create a new connection with standard settings
                    conn = duckdb.connect(database=self.current_db_path, read_only=True)
                    tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()
                except Exception as e:
                    # If that fails, try without specific settings
                    QMessageBox.warning(self, "Connection Warning", 
                                     f"Retrying connection with default settings: {str(e)}")
                    conn = duckdb.connect(self.current_db_path)
                    tables = conn.execute("SHOW TABLES").fetchall()
            else:
                import sqlite3
                conn = sqlite3.connect(self.current_db_path)
                cursor = conn.cursor()
                tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            
            # Make sure to close the connection
            conn.close()
            
            # Add tables to the list widget
            for table in tables:
                self.table_list.addItem(table[0])
                
            # Enable the table list if there are tables
            self.table_list.setEnabled(len(tables) > 0)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to update table list: {str(e)}")
            # Reset the connection status
            self.table_list.setEnabled(False)

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

        # Close current database connection first if needed
        if self.current_db_path:
            # We don't have a direct connection object to close, but this will help 
            # mark the current database as closed in our UI state
            self.current_db_path = None
            # Add a small delay to allow any existing connection to be fully closed
            QApplication.processEvents()

        try:
            # Connect to the new database with careful error handling
            try:
                # First try with standard connection settings
                conn = duckdb.connect(database=db_path, read_only=False)
                tables = conn.execute("SHOW TABLES").fetchall()
            except Exception as e1:
                # If that fails, try with minimal settings
                QMessageBox.warning(self, "Connection Warning", 
                                 f"Retrying with minimal connection settings: {str(e1)}")
                conn = duckdb.connect(db_path)
                tables = conn.execute("SHOW TABLES").fetchall()
                
            # Properly close the connection when done
            conn.close()
            
            # Update UI
            self.table_list.clear()
            self.table_list.addItems([table[0] for table in tables])
            self.table_list.setEnabled(True)
            self.current_db_path = db_path
            self.database_loaded.emit(db_path)

        except Exception as e:
            QMessageBox.critical(self, "Database Error", f"Failed to switch database: {str(e)}")
            self.table_list.clear()
            self.table_list.setEnabled(False)
            self.current_db_path = None
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
        
        # For DuckDB databases, handle queries differently
        if self.current_db_path.lower().endswith('.duckdb'):
            # Check if this is a simple SELECT * FROM query with any kind of quoted or unquoted table name
            simple_query_match = re.match(r'^\s*SELECT\s+\*\s+FROM\s+(?:[\'"]([^\'"]+)[\'"]|([^\s;]+))\s*;?\s*$', query, re.IGNORECASE)
            
            if simple_query_match:
                # Get the table name (either quoted or unquoted)
                table_name = simple_query_match.group(1) if simple_query_match.group(1) else simple_query_match.group(2)
                
                try:
                    # Try to directly access the table to verify it exists
                    import duckdb
                    conn = duckdb.connect(database=self.current_db_path, read_only=False)
                    tables = conn.execute("SHOW TABLES").fetchall()
                    
                    # Case-insensitive table name matching
                    table_found = False
                    for t in tables:
                        if t[0].lower() == table_name.lower():
                            # Use the actual table name with the correct case
                            query = f"SELECT * FROM {t[0]}"
                            tab.status_label.setText(f"Using direct table access for '{t[0]}'")
                            table_found = True
                            break
                    
                    if not table_found:
                        # If table not found with exact name, try without quotes
                        query = f"SELECT * FROM {table_name}"
                        tab.status_label.setText(f"Table not found with exact name, trying without quotes: '{table_name}'")
                    
                    conn.close()
                except Exception as e:
                    # If there's an error, fall back to the simplified query
                    query = f"SELECT * FROM {table_name}"
                    tab.status_label.setText(f"Error checking table: {str(e)}. Using simplified query format...")

        tab.status_label.setText("Executing query...")
        
        # Create and start the worker thread
        self.worker = QueryWorker(self.current_db_path, query)
        
        # Store the worker as an instance variable of the tab to prevent garbage collection
        tab.query_worker = self.worker
        
        # Connect signals
        self.worker.resultReady.connect(lambda df: self.handle_query_result(tab, df))
        self.worker.errorOccurred.connect(lambda error: self.handle_query_error(tab, error))
        self.worker.progressUpdate.connect(lambda msg: tab.status_label.setText(msg))
        
        # Start the worker thread
        self.worker.start()

    def handle_query_result(self, tab, df):
        tab.current_df = df
        model = PandasModel(df)
        tab.table_view.setModel(model)
        tab.status_label.setText(f"Query completed successfully. Rows: {len(df)}")

    def handle_query_error(self, tab, error):
        tab.status_label.setText("Error executing query")
        
        # Check if this is a parser error with quotes
        if "Parser Error: syntax error at or near" in error:
            # Try to extract the problematic part
            match = re.search(r'syntax error at or near "([^"]*)"', error)
            if match:
                problematic_part = match.group(1)
                
                # Check if this is a quoted table name issue
                table_match = re.search(r'FROM\s+[\'"]([^\'"]+)[\'"]', tab.query_edit.toPlainText(), re.IGNORECASE)
                if table_match:
                    table_name = table_match.group(1)
                    
                    # Ask user if they want to try without quotes
                    reply = QMessageBox.question(
                        self, 
                        "Parser Error", 
                        f"There was a syntax error with the quoted table name. Would you like to try running the query without quotes around '{table_name}'?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes
                    )
                    
                    if reply == QMessageBox.StandardButton.Yes:
                        # Modify the query to remove quotes around table name
                        original_query = tab.query_edit.toPlainText()
                        
                        # Handle both single and double quotes
                        if f'"{table_name}"' in original_query:
                            modified_query = original_query.replace(f'"{table_name}"', table_name)
                        elif f"'{table_name}'" in original_query:
                            modified_query = original_query.replace(f"'{table_name}'", table_name)
                        else:
                            # Use regex for more complex cases
                            modified_query = re.sub(r'FROM\s+[\'"]([^\'"]+)[\'"]', f'FROM {table_name}', original_query, flags=re.IGNORECASE)
                        
                        # Update the query editor
                        tab.query_edit.setPlainText(modified_query)
                        
                        # Run the simplified query
                        self.run_query(tab)
                        return
        
        # If we couldn't fix it automatically or it's a different error, show the error message
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

    def show_table_context_menu(self, position):
        """Show context menu for table list"""
        # Only show if a table is selected and database is connected
        if not self.current_db_path or not self.table_list.currentItem():
            return
            
        # Get the selected table name
        selected_table = self.table_list.currentItem().text()
        
        # Create context menu
        context_menu = QMenu(self)
        
        # Add view data action
        view_data_action = QAction(f"View data in '{selected_table}'", self)
        view_data_action.triggered.connect(lambda: self.view_table_data(selected_table))
        context_menu.addAction(view_data_action)
        
        # Add view structure action
        view_structure_action = QAction(f"View structure of '{selected_table}'", self)
        view_structure_action.triggered.connect(lambda: self.view_table_structure(selected_table))
        context_menu.addAction(view_structure_action)
        
        # Add separator
        context_menu.addSeparator()
        
        # Add rename action
        rename_action = QAction(f"Rename table '{selected_table}'", self)
        rename_action.triggered.connect(lambda: self.rename_table(selected_table))
        context_menu.addAction(rename_action)
        
        # Add delete action
        delete_action = QAction(f"Delete table '{selected_table}'", self)
        delete_action.triggered.connect(lambda: self.delete_table(selected_table))
        context_menu.addAction(delete_action)
        
        # Show the menu at the cursor position
        context_menu.exec(self.table_list.mapToGlobal(position))
    
    def view_table_data(self, table_name):
        """View the data in a table"""
        if not self.current_db_path:
            return
            
        # Create a new tab
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        
        # Add a query editor with the SELECT query
        tab.query_edit = CodeEditor()
        # Fix the query format with proper spacing around LIMIT
        tab.query_edit.setPlainText(f"SELECT * FROM '{table_name}' LIMIT 1000")
        tab.query_edit.setMinimumHeight(100)
        tab_layout.addWidget(tab.query_edit)
        
        # Add Run Query button
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
        self.tab_widget.addTab(tab, f"Data: {table_name}")
        self.tab_widget.setCurrentWidget(tab)
        
        # Run the query automatically
        self.run_query(tab)
    
    def view_table_structure(self, table_name):
        """View the structure of a table"""
        if not self.current_db_path:
            return
            
        try:
            # Connect to the database
            if self.current_db_path.lower().endswith('.duckdb'):
                conn = duckdb.connect(self.current_db_path, read_only=True)
                # Get table structure
                result = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
                # Create a DataFrame from the result
                columns = ["cid", "name", "type", "notnull", "dflt_value", "pk"]
                df = pd.DataFrame(result, columns=columns)
                conn.close()
            else:
                import sqlite3
                conn = sqlite3.connect(self.current_db_path)
                cursor = conn.cursor()
                # Get table structure
                result = cursor.execute(f"PRAGMA table_info('{table_name}')").fetchall()
                # Create a DataFrame from the result
                columns = ["cid", "name", "type", "notnull", "dflt_value", "pk"]
                df = pd.DataFrame(result, columns=columns)
                conn.close()
            
            # Create a new tab to display the structure
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            
            # Add a label with the table name
            header_label = QLabel(f"Structure of table '{table_name}':")
            header_label.setStyleSheet("font-weight: bold; font-size: 14px;")
            tab_layout.addWidget(header_label)
            
            # Add table view for the structure
            table_view = QTableView()
            table_view.setModel(PandasModel(df))
            tab_layout.addWidget(table_view)
            
            # Add the tab to the tab widget
            self.tab_widget.addTab(tab, f"Structure: {table_name}")
            self.tab_widget.setCurrentWidget(tab)
            
            # Store the DataFrame in the tab
            tab.current_df = df
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to view table structure: {str(e)}")
    
    def delete_table(self, table_name):
        """Delete a table from the database"""
        if not self.current_db_path:
            return
            
        # Confirm deletion
        confirm = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete the table '{table_name}'?\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if confirm != QMessageBox.StandardButton.Yes:
            return
            
        try:
            # Connect to the database
            if self.current_db_path.lower().endswith('.duckdb'):
                conn = duckdb.connect(self.current_db_path)
                conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                conn.close()
            else:
                import sqlite3
                conn = sqlite3.connect(self.current_db_path)
                cursor = conn.cursor()
                cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                conn.commit()
                conn.close()
                
            # Update the table list
            self.update_table_list()
            
            # Show success message
            QMessageBox.information(self, "Success", f"Table '{table_name}' has been deleted.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to delete table: {str(e)}")

    def on_table_double_clicked(self, item):
        """Handle double-click on a table item"""
        if item and self.current_db_path:
            self.view_table_data(item.text())

    def show_query_examples(self):
        """Show a dialog with popular SQL query examples"""
        examples_dialog = QDialog(self)
        examples_dialog.setWindowTitle("SQL Query Examples")
        examples_dialog.setMinimumWidth(600)
        examples_dialog.setMinimumHeight(400)
        
        layout = QVBoxLayout(examples_dialog)
        
        # Add a label with instructions
        header_label = QLabel("Common SQL Query Examples")
        header_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(header_label)
        
        # Create a text edit with examples
        examples_text = QPlainTextEdit()
        examples_text.setReadOnly(True)
        examples_text.setStyleSheet("font-family: monospace; font-size: 12px;")
        
        # Add example queries
        examples = [
            "-- Basic SELECT query",
            "SELECT * FROM 'table_name' LIMIT 100;",
            "\n",
            "-- SELECT specific columns",
            "SELECT column1, column2 FROM 'table_name' LIMIT 100;",
            "\n",
            "-- Filter with WHERE clause",
            "SELECT * FROM 'table_name' WHERE column_name > 100;",
            "\n",
            "-- Sort results with ORDER BY",
            "SELECT * FROM 'table_name' ORDER BY column_name DESC LIMIT 100;",
            "\n",
            "-- Group and aggregate data",
            "SELECT column1, COUNT(*) as count FROM 'table_name' GROUP BY column1;",
            "\n",
            "-- Join tables",
            "SELECT t1.column1, t2.column2 FROM 'table1' t1",
            "JOIN 'table2' t2 ON t1.id = t2.id LIMIT 100;",
            "\n",
            "-- Filter with multiple conditions",
            "SELECT * FROM 'table_name' WHERE column1 > 10 AND column2 = 'value';",
            "\n",
            "-- Use LIKE for pattern matching",
            "SELECT * FROM 'table_name' WHERE column_name LIKE '%pattern%';",
            "\n",
            "-- Calculate statistics",
            "SELECT AVG(column1) as average, MAX(column1) as maximum,",
            "MIN(column1) as minimum FROM 'table_name';",
        ]
        
        examples_text.setPlainText("\n".join(examples))
        layout.addWidget(examples_text)
        
        # Add a button to copy the selected example
        copy_button = QPushButton("Copy Selected Example")
        copy_button.clicked.connect(lambda: self.copy_selected_example(examples_text))
        layout.addWidget(copy_button)
        
        # Add a close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(examples_dialog.accept)
        layout.addWidget(close_button)
        
        examples_dialog.exec()

    def copy_selected_example(self, text_edit):
        """Copy the selected text to clipboard and create a new query tab with it"""
        selected_text = text_edit.textCursor().selectedText()
        
        if not selected_text:
            QMessageBox.information(self, "No Selection", "Please select an example query first.")
            return
        
        # Create a new query tab with the selected example
        tab = self.add_query_tab()
        tab.query_edit.setPlainText(selected_text)
        
        QMessageBox.information(self, "Example Copied", "The example has been copied to a new query tab.")

    def rename_table(self, table_name):
        """Rename a table in the database"""
        if not self.current_db_path:
            return
            
        # Prompt for new table name
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Table",
            f"Enter new name for table '{table_name}':",
            QLineEdit.EchoMode.Normal,
            table_name
        )
        
        if not ok or not new_name or new_name == table_name:
            return
            
        # Validate the new name (basic validation)
        if not re.match(r'^[a-zA-Z0-9_]+$', new_name):
            QMessageBox.warning(
                self,
                "Invalid Name",
                "Table name can only contain letters, numbers, and underscores."
            )
            return
            
        # Check if the new name already exists
        try:
            # Connect to the database
            if self.current_db_path.lower().endswith('.duckdb'):
                conn = duckdb.connect(self.current_db_path)
                tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()
            else:
                import sqlite3
                conn = sqlite3.connect(self.current_db_path)
                cursor = conn.cursor()
                tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                
            existing_tables = [t[0] for t in tables]
            
            if new_name in existing_tables:
                QMessageBox.warning(
                    self,
                    "Table Exists",
                    f"A table named '{new_name}' already exists. Please choose a different name."
                )
                conn.close()
                return
                
            # Rename the table
            if self.current_db_path.lower().endswith('.duckdb'):
                # For DuckDB
                conn.execute(f'ALTER TABLE "{table_name}" RENAME TO "{new_name}"')
            else:
                # For SQLite
                cursor.execute(f'ALTER TABLE "{table_name}" RENAME TO "{new_name}"')
                conn.commit()
                
            conn.close()
            
            # Update the table list
            self.update_table_list()
            
            # Show success message
            QMessageBox.information(
                self,
                "Success",
                f"Table '{table_name}' has been renamed to '{new_name}'."
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to rename table: {str(e)}"
            )

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
        self.table_button_group = QButtonGroup(self)
        
        self.new_table_radio = QRadioButton("Create new table")
        self.new_table_radio.setChecked(True)
        self.table_button_group.addButton(self.new_table_radio)
        
        self.existing_table_radio = QRadioButton("Use existing table")
        self.table_button_group.addButton(self.existing_table_radio)
        
        self.replace_table_radio = QRadioButton("Replace existing table")
        self.table_button_group.addButton(self.replace_table_radio)
        
        # Connect signals
        self.table_button_group.buttonClicked.connect(self.toggle_table_options)
        
        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.new_table_radio)
        button_layout.addWidget(self.existing_table_radio)
        button_layout.addWidget(self.replace_table_radio)
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
            self.replace_table = False
        elif self.existing_table_radio.isChecked():
            if self.table_combo.count() == 0:
                self.status_label.setText("No tables available. Please create a new table.")
                return
            table_name = self.table_combo.currentText()
            self.use_existing_table = True
            self.replace_table = False
        else:  # replace_table_radio is checked
            if self.table_combo.count() == 0:
                self.status_label.setText("No tables available to replace. Please create a new table.")
                return
            table_name = self.table_combo.currentText()
            self.use_existing_table = False
            self.replace_table = True
        
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

class ImportFileDialog(QDialog):
    def __init__(self, parent=None, available_databases=None):
        super().__init__(parent)
        self.setWindowTitle("Import File into Database")
        self.available_databases = available_databases or []
        self.parent = parent
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Create form layout
        form_layout = QFormLayout()
        
        # Source file selection
        file_layout = QHBoxLayout()
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("Select file to import")
        self.file_input.setReadOnly(True)
        file_layout.addWidget(self.file_input)
        
        browse_file_button = QPushButton("Browse...")
        browse_file_button.clicked.connect(self.browse_file)
        file_layout.addWidget(browse_file_button)
        
        form_layout.addRow("Source File:", file_layout)
        
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
        self.table_button_group = QButtonGroup(self)
        
        self.new_table_radio = QRadioButton("Create new table")
        self.new_table_radio.setChecked(True)
        self.table_button_group.addButton(self.new_table_radio)
        
        self.existing_table_radio = QRadioButton("Use existing table")
        self.table_button_group.addButton(self.existing_table_radio)
        
        self.replace_table_radio = QRadioButton("Replace existing table")
        self.table_button_group.addButton(self.replace_table_radio)
        
        # Connect signals
        self.table_button_group.buttonClicked.connect(self.toggle_table_options)
        
        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.new_table_radio)
        button_layout.addWidget(self.existing_table_radio)
        button_layout.addWidget(self.replace_table_radio)
        button_layout.addStretch()
        
        self.table_option_layout.addLayout(button_layout)
        
        # New table name input
        self.new_table_layout = QHBoxLayout()
        self.table_name_input = QLineEdit("imported_data")
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
        
        # Import options
        self.options_group = QWidget()
        options_layout = QVBoxLayout(self.options_group)
        
        # CSV options
        self.csv_options = QWidget()
        csv_layout = QFormLayout(self.csv_options)
        
        self.header_checkbox = QCheckBox("First row contains headers")
        self.header_checkbox.setChecked(True)
        csv_layout.addRow("", self.header_checkbox)
        
        self.delimiter_combo = QComboBox()
        self.delimiter_combo.addItem("Comma (,)", ",")
        self.delimiter_combo.addItem("Semicolon (;)", ";")
        self.delimiter_combo.addItem("Tab (\\t)", "\t")
        self.delimiter_combo.addItem("Pipe (|)", "|")
        csv_layout.addRow("Delimiter:", self.delimiter_combo)
        
        options_layout.addWidget(self.csv_options)
        
        # Excel options
        self.excel_options = QWidget()
        excel_layout = QFormLayout(self.excel_options)
        
        self.sheet_combo = QComboBox()
        self.sheet_combo.addItem("First sheet")
        excel_layout.addRow("Sheet:", self.sheet_combo)
        
        self.excel_header_checkbox = QCheckBox("First row contains headers")
        self.excel_header_checkbox.setChecked(True)
        excel_layout.addRow("", self.excel_header_checkbox)
        
        options_layout.addWidget(self.excel_options)
        
        # Parquet options (minimal since parquet has schema)
        self.parquet_options = QWidget()
        parquet_layout = QFormLayout(self.parquet_options)
        
        parquet_info = QLabel("Parquet files include schema information and will be imported as-is.")
        parquet_info.setWordWrap(True)
        parquet_layout.addRow("", parquet_info)
        
        options_layout.addWidget(self.parquet_options)
        
        # Hide all options initially
        self.csv_options.hide()
        self.excel_options.hide()
        self.parquet_options.hide()
        
        form_layout.addRow("Import Options:", self.options_group)
        
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
    
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select File to Import", 
            "", 
            "Data Files (*.csv *.xlsx *.xls *.parquet);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;Parquet Files (*.parquet);;All Files (*)"
        )
        
        if file_path:
            self.file_input.setText(file_path)
            
            # Set a default table name based on the file name
            file_name = Path(file_path).stem
            self.table_name_input.setText(file_name)
            
            # Show appropriate options based on file type
            self.update_options_for_file(file_path)
            
            # If it's an Excel file, load sheet names
            if file_path.lower().endswith(('.xlsx', '.xls')):
                self.load_excel_sheets(file_path)
    
    def update_options_for_file(self, file_path):
        # Hide all options first
        self.csv_options.hide()
        self.excel_options.hide()
        self.parquet_options.hide()
        
        # Show appropriate options based on file extension
        if file_path.lower().endswith('.csv'):
            self.csv_options.show()
        elif file_path.lower().endswith(('.xlsx', '.xls')):
            self.excel_options.show()
        elif file_path.lower().endswith('.parquet'):
            self.parquet_options.show()
    
    def load_excel_sheets(self, file_path):
        try:
            import openpyxl
            workbook = openpyxl.load_workbook(file_path, read_only=True)
            
            # Clear and update sheet combo
            self.sheet_combo.clear()
            
            for sheet_name in workbook.sheetnames:
                self.sheet_combo.addItem(sheet_name)
                
            workbook.close()
            
        except Exception as e:
            # If openpyxl fails, try pandas
            try:
                import pandas as pd
                xls = pd.ExcelFile(file_path)
                
                # Clear and update sheet combo
                self.sheet_combo.clear()
                
                for sheet_name in xls.sheet_names:
                    self.sheet_combo.addItem(sheet_name)
                    
            except Exception as e2:
                self.status_label.setText(f"Error reading Excel sheets: {str(e2)}")
                self.sheet_combo.clear()
                self.sheet_combo.addItem("First sheet")
    
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
    
    def validate_and_accept(self):
        # Get values from form
        file_path = self.file_input.text().strip()
        selected_db = self.db_combo.currentText()
        
        # Validate input
        if not file_path:
            self.status_label.setText("Please select a file to import.")
            return
            
        if not os.path.isfile(file_path):
            self.status_label.setText("Selected path is not a valid file.")
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
            self.replace_table = False
        elif self.existing_table_radio.isChecked():
            if self.table_combo.count() == 0:
                self.status_label.setText("No tables available. Please create a new table.")
                return
            table_name = self.table_combo.currentText()
            self.use_existing_table = True
            self.replace_table = False
        else:  # replace_table_radio is checked
            if self.table_combo.count() == 0:
                self.status_label.setText("No tables available to replace. Please create a new table.")
                return
            table_name = self.table_combo.currentText()
            self.use_existing_table = False
            self.replace_table = True
        
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
        self.file_path = file_path
        self.table_name = table_name
        
        # Store import options
        self.import_options = {}
        
        # CSV options
        if file_path.lower().endswith('.csv'):
            self.import_options['header'] = self.header_checkbox.isChecked()
            self.import_options['delimiter'] = self.delimiter_combo.currentData()
        
        # Excel options
        elif file_path.lower().endswith(('.xlsx', '.xls')):
            self.import_options['header'] = self.excel_header_checkbox.isChecked()
            self.import_options['sheet_name'] = self.sheet_combo.currentText()
        
        # Accept the dialog
        self.accept()

class ImportFileWorker(QObject):
    progress = pyqtSignal(str)  # Signal to emit progress updates
    error = pyqtSignal(str)     # Signal to emit if an error occurs
    success = pyqtSignal(str)   # Signal to emit on successful import
    finished = pyqtSignal()     # Signal to emit when process is complete
    table_created = pyqtSignal(str, str)  # Signal to emit when a table is created (db_path, table_name)
    
    CHUNK_SIZE = 50000  # Chunk size for better memory management
    
    def __init__(self, file_path, db_path, table_name, use_existing_table=False, replace_table=False, import_options=None):
        super().__init__()
        self.file_path = file_path
        self.db_path = db_path
        self.table_name = table_name
        self.use_existing_table = use_existing_table
        self.replace_table = replace_table
        self.import_options = import_options or {}
        self.is_cancelled = False
        self.is_duckdb = db_path.lower().endswith('.duckdb')
        self.chunk_size = self.CHUNK_SIZE  # Use the class variable
        self.total_rows_processed = 0
    
    def insert_data_with_column_matching(self, conn, df, existing_columns):
        """Insert data with column matching to handle different column sets"""
        try:
            self.progress.emit(f"Aligning columns with existing table structure...")
            
            # Create sets for easier comparison
            df_columns = set(df.columns)
            table_columns = set(existing_columns)
            
            # Find columns in DataFrame that aren't in the table
            # (This shouldn't happen as we've already added missing columns, but just in case)
            missing_in_table = df_columns - table_columns
            if missing_in_table:
                self.progress.emit(f"Found {len(missing_in_table)} columns in data that aren't in the table. Adding them...")
                self.add_missing_columns(conn, df[list(missing_in_table)], existing_columns)
                # Update existing columns
                existing_columns = self.get_existing_columns(conn)
                table_columns = set(existing_columns)
            
            # Find columns in the table that aren't in the DataFrame
            missing_in_df = table_columns - df_columns
            if missing_in_df:
                self.progress.emit(f"Found {len(missing_in_df)} columns in table that aren't in the data. Will fill with NULL values.")
                # Add missing columns to DataFrame with NULL values
                for col in missing_in_df:
                    df.loc[:, col] = None
            
            # Create a new DataFrame that contains all columns from the table
            # This ensures we maintain column order and include all columns
            ordered_df = pd.DataFrame()
            for col in existing_columns:
                if col in df.columns:
                    ordered_df.loc[:, col] = df[col]
                else:
                    ordered_df.loc[:, col] = None
            
            if len(ordered_df.columns) == 0:
                self.progress.emit("Warning: No valid columns found for insertion. Skipping this chunk.")
                return
            
            # Now insert the data
            if self.is_duckdb:
                # For DuckDB, register the aligned DataFrame
                conn.register('aligned_df', ordered_df)
                try:
                    # Try with double quotes first
                    conn.execute(f"INSERT INTO \"{self.table_name}\" SELECT * FROM aligned_df")
                except Exception as e:
                    if "Parser Error" in str(e):
                        # Try without quotes
                        conn.execute(f"INSERT INTO {self.table_name} SELECT * FROM aligned_df")
            else:
                # For SQLite, use pandas to_sql
                try:
                    ordered_df.to_sql(self.table_name, conn, if_exists='append', index=False)
                except Exception as e:
                    # If there's an error, try to get the actual columns from the database
                    # and filter the DataFrame to only include those columns
                    self.progress.emit(f"Error inserting data: {str(e)}. Trying with exact column matching...")
                    cursor = conn.cursor()
                    cursor.execute(f"PRAGMA table_info('{self.table_name}')")
                    db_columns = [row[1] for row in cursor.fetchall()]
                    
                    # Create a new DataFrame with exactly the columns from the database
                    final_df = pd.DataFrame()
                    for col in db_columns:
                        if col in ordered_df.columns:
                            final_df.loc[:, col] = ordered_df[col]
                        else:
                            final_df.loc[:, col] = None
                    
                    # Try to insert the precisely matched DataFrame
                    if len(final_df.columns) > 0:
                        final_df.to_sql(self.table_name, conn, if_exists='append', index=False)
                    else:
                        raise Exception("No matching columns found between DataFrame and database table")
            
            self.progress.emit(f"Inserted {len(df)} rows with column alignment.")
        except Exception as e:
            self.progress.emit(f"Error in column matching: {str(e)}. Will try to continue with next chunk.")
            # Don't re-raise the exception so processing can continue
    
    def run(self):
        try:
            self.progress.emit(f"Reading file: {self.file_path}")
            
            # Read the file
            df_chunks = self.read_file()
            if df_chunks is None or self.is_cancelled:
                self.finished.emit()
                return
            
            # Check if df_chunks is empty
            if isinstance(df_chunks, list) and len(df_chunks) == 0:
                self.error.emit(f"No data found in file: {self.file_path}")
                self.finished.emit()
                return
            
            # Connect to the database
            if self.is_duckdb:
                conn = duckdb.connect(self.db_path)
                # Increase memory limit for DuckDB
                try:
                    conn.execute("SET memory_limit='8GB'")
                    conn.execute("PRAGMA threads=4")  # Use multiple threads for better performance
                except Exception as e:
                    self.progress.emit(f"Notice: Could not set memory limits: {str(e)}")
            else:
                import sqlite3
                conn = sqlite3.connect(self.db_path, timeout=300)  # Increase timeout to 5 minutes
                conn.execute("PRAGMA journal_mode=WAL")  # Use WAL mode for better performance
                conn.execute("PRAGMA synchronous=NORMAL")  # Reduce synchronous mode for better performance
            
            # Process the first chunk to set up the table
            try:
                # Get the first chunk to analyze schema
                try:
                    first_chunk = next(df_chunks) if hasattr(df_chunks, '__next__') else df_chunks[0]
                except (StopIteration, IndexError):
                    self.error.emit(f"No data found in file: {self.file_path}")
                    self.finished.emit()
                    conn.close()
                    return
                
                # Clean column names for the first chunk
                first_chunk.columns = self.clean_column_names(first_chunk.columns)
                
                # Remove rows with all NaN values
                first_chunk = first_chunk.dropna(how='all')
                
                self.progress.emit(f"File read successfully. Processing data...")
                
                # Store table name and db path for later use
                table_name = self.table_name
                db_path = self.db_path
                
                # Check if we're using an existing table
                if self.use_existing_table:
                    # Get existing columns
                    existing_columns = self.get_existing_columns(conn)
                    
                    # Add any missing columns
                    self.add_missing_columns(conn, first_chunk, existing_columns)
                    
                    # Get updated list of columns after adding new ones
                    existing_columns = self.get_existing_columns(conn)
                    
                    # Insert data with column matching
                    self.insert_data_with_column_matching(conn, first_chunk, existing_columns)
                else:
                    # If replacing, drop the existing table first
                    if self.replace_table:
                        self.progress.emit(f"Replacing existing table '{self.table_name}'...")
                        try:
                            conn.execute(f"DROP TABLE IF EXISTS \"{self.table_name}\"")
                        except Exception as e:
                            self.error.emit(f"Error dropping existing table: {str(e)}")
                            self.finished.emit()
                            conn.close()  # Close connection on error
                            return
                    
                    # Create the table if it doesn't exist
                    self.create_table_if_not_exists(conn, first_chunk)
                    
                    # For a new table, we can insert directly
                    if self.is_duckdb:
                        # For DuckDB, we can use the append method
                        conn.register('df', first_chunk)
                        conn.execute(f"INSERT INTO \"{self.table_name}\" SELECT * FROM df")
                    else:
                        # For SQLite, we need to use the pandas to_sql method
                        first_chunk.to_sql(self.table_name, conn, if_exists='append', index=False)
                
                # Process remaining chunks if any
                self.total_rows_processed = len(first_chunk)
                rows_imported = self.total_rows_processed
                chunk_count = 1
                
                # Start a transaction for better performance on SQLite
                if not self.is_duckdb:
                    conn.execute("BEGIN TRANSACTION")
                
                # If df_chunks is a generator or list with more chunks
                if hasattr(df_chunks, '__next__'):
                    # Process remaining chunks from generator
                    for chunk in df_chunks:
                        if self.is_cancelled:
                            break
                        
                        # Clean column names
                        chunk.columns = self.clean_column_names(chunk.columns)
                        
                        # Remove rows with all NaN values
                        chunk = chunk.dropna(how='all')
                        
                        # Check for and add any new columns that might be in this chunk
                        existing_columns = self.get_existing_columns(conn)
                        self.add_missing_columns(conn, chunk, existing_columns)
                        
                        # Get updated columns after potentially adding new ones
                        existing_columns = self.get_existing_columns(conn)
                        
                        # Insert data with column matching
                        self.insert_data_with_column_matching(conn, chunk, existing_columns)
                        
                        # Update progress
                        rows_imported += len(chunk)
                        self.total_rows_processed += len(chunk)
                        chunk_count += 1
                        self.progress.emit(f"Imported chunk {chunk_count}: {rows_imported} rows total...")
                        
                        # Commit intermediate transaction every 10 chunks for SQLite
                        if not self.is_duckdb and chunk_count % 10 == 0:
                            conn.commit()
                            conn.execute("BEGIN TRANSACTION")
                            self.progress.emit("Intermediate commit completed")
                
                elif isinstance(df_chunks, list) and len(df_chunks) > 1:
                    # Process remaining chunks from list
                    for i in range(1, len(df_chunks)):
                        if self.is_cancelled:
                            break
                        
                        chunk = df_chunks[i]
                        
                        # Clean column names
                        chunk.columns = self.clean_column_names(chunk.columns)
                        
                        # Remove rows with all NaN values
                        chunk = chunk.dropna(how='all')
                        
                        # Check for and add any new columns that might be in this chunk
                        existing_columns = self.get_existing_columns(conn)
                        self.add_missing_columns(conn, chunk, existing_columns)
                        
                        # Get updated columns after potentially adding new ones
                        existing_columns = self.get_existing_columns(conn)
                        
                        # Insert data with column matching
                        self.insert_data_with_column_matching(conn, chunk, existing_columns)
                        
                        # Update progress
                        rows_imported += len(chunk)
                        self.total_rows_processed += len(chunk)
                        chunk_count = i + 1
                        self.progress.emit(f"Imported chunk {chunk_count}: {rows_imported} rows total...")
                        
                        # Commit intermediate transaction every 10 chunks for SQLite
                        if not self.is_duckdb and chunk_count % 10 == 0:
                            conn.commit()
                            conn.execute("BEGIN TRANSACTION")
                            self.progress.emit("Intermediate commit completed")
                
                # Commit the final transaction for SQLite
                if not self.is_duckdb and not self.is_cancelled:
                    try:
                        conn.commit()
                        self.progress.emit("Final commit completed")
                    except Exception as e:
                        self.error.emit(f"Error committing final transaction: {str(e)}")
                        conn.rollback()
                
                # Close the database connection
                conn.close()
                self.progress.emit("Database connection closed")
                
                if not self.is_cancelled:
                    self.success.emit(f"Successfully imported {self.file_path} into table '{table_name}'. Total rows: {self.total_rows_processed}")
                    
                    # Give the connection some time to fully close before emitting table_created
                    QThread.msleep(200)  # Brief delay to ensure connection is fully closed
                    # Emit signal that table was created/updated - after connection is closed
                    self.table_created.emit(db_path, table_name)
                    
            except Exception as e:
                self.error.emit(f"Error during import: {str(e)}")
                conn.close()  # Make sure to close the connection

        except Exception as e:
            self.error.emit(f"Error reading file: {str(e)}")
            
        finally:
            self.finished.emit()
            
    def read_file(self):
        """Read file in chunks to avoid memory issues"""
        # Convert Path object to string if needed
        file_path_str = str(self.file_path)
        file_ext = self.file_path.suffix.lower() if hasattr(self.file_path, 'suffix') else os.path.splitext(file_path_str)[1].lower()
        
        try:
            if file_ext == '.csv':
                # Try with default settings first
                try:
                    # Try to count total lines for progress reporting
                    try:
                        with open(file_path_str, 'r') as f:
                            total_lines = sum(1 for _ in f)
                        self.progress.emit(f"CSV file has approximately {total_lines} lines")
                    except Exception:
                        pass  # Ignore if we can't count lines
                        
                    # Process with reasonable chunk size
                    for chunk in pd.read_csv(file_path_str, chunksize=self.chunk_size):
                        # Clean column names
                        chunk.columns = self.clean_column_names(chunk.columns)
                        # Remove rows with all NaN values
                        chunk = chunk.dropna(how='all')
                        yield chunk
                except Exception as e:
                    self.progress.emit(f"Error reading CSV with default settings, trying more flexible settings: {str(e)}")
                    # Try with more flexible settings
                    for chunk in pd.read_csv(file_path_str, chunksize=self.chunk_size, 
                                          encoding='latin1', on_bad_lines='skip', 
                                          low_memory=False, dtype=str):
                        # Clean column names
                        chunk.columns = self.clean_column_names(chunk.columns)
                        # Remove rows with all NaN values
                        chunk = chunk.dropna(how='all')
                        yield chunk
                        
            elif file_ext in ['.xlsx', '.xls']:
                # For Excel, we need to read the entire file at once
                # but we can process it in chunks
                
                # First, check if the file is actually a valid file and exists
                if not os.path.exists(file_path_str):
                    raise FileNotFoundError(f"Excel file not found: {file_path_str}")
                
                # Check file size - if it's too small it's probably corrupt or empty
                file_size = os.path.getsize(file_path_str)
                if file_size < 100:  # Extremely small for an Excel file
                    raise ValueError(f"Excel file is likely corrupt or empty (size: {file_size} bytes)")
                
                # Try multiple engines with proper error handling
                df = None
                success = False
                error_messages = []
                
                # Try with openpyxl first for .xlsx
                if file_ext == '.xlsx':
                    try:
                        self.progress.emit(f"Trying to read Excel file with openpyxl...")
                        df = pd.read_excel(
                            self.file_path,
                            sheet_name=self.import_options.get('sheet_name', 0),
                            header=0 if self.import_options.get('header', True) else None,
                            engine='openpyxl'
                        )
                        success = True
                    except Exception as e:
                        error_messages.append(f"openpyxl error: {str(e)}")
                        self.progress.emit(f"Error with openpyxl: {str(e)}")
                
                # If .xls or openpyxl failed, try xlrd
                if not success:
                    try:
                        self.progress.emit(f"Trying to read Excel file with xlrd...")
                        df = pd.read_excel(
                            self.file_path,
                            sheet_name=self.import_options.get('sheet_name', 0),
                            header=0 if self.import_options.get('header', True) else None,
                            engine='xlrd'
                        )
                        success = True
                    except Exception as e:
                        error_messages.append(f"xlrd error: {str(e)}")
                        self.progress.emit(f"Error with xlrd: {str(e)}")
                
                # If still no success, try as CSV as a last resort
                if not success:
                    try:
                        self.progress.emit("Trying to read file as CSV instead...")
                        df = pd.read_csv(
                            self.file_path,
                            header=0 if self.import_options.get('header', True) else None
                        )
                        success = True
                    except Exception as e:
                        error_messages.append(f"CSV fallback error: {str(e)}")
                        self.progress.emit(f"CSV fallback failed: {str(e)}")
                
                if not success:
                    raise ValueError(f"Could not read Excel file with any available engine: {' | '.join(error_messages)}")
                
                self.progress.emit(f"Successfully read Excel file with {len(df)} rows. Processing in chunks...")
                
                # Split into chunks and return as a list
                chunks = []
                total_rows = len(df)
                
                for i in range(0, total_rows, self.chunk_size):
                    end = min(i + self.chunk_size, total_rows)
                    chunks.append(df.iloc[i:end])
                
                return chunks
            
            # For Parquet files
            elif file_ext == '.parquet':
                # Store file_path locally to avoid issues with 'self' reference in nested functions
                file_path_str = self.file_path
                chunk_size = self.chunk_size
                clean_column_names_func = self.clean_column_names
                progress_emit_func = self.progress.emit
                
                # Try using pyarrow for chunked reading
                try:
                    import pyarrow.parquet as pq
                    
                    # Open the file
                    parquet_file = pq.ParquetFile(file_path_str)
                    
                    # Get metadata
                    total_rows = parquet_file.metadata.num_rows
                    num_row_groups = parquet_file.num_row_groups
                    
                    progress_emit_func(f"Parquet file has {total_rows} rows in {num_row_groups} row groups")
                    
                    # Create a list to store chunks instead of using a generator
                    all_chunks = []
                    
                    # Process each row group
                    for i in range(num_row_groups):
                        # Read row group
                        table = parquet_file.read_row_group(i)
                        
                        # Convert to pandas DataFrame
                        chunk = table.to_pandas()
                        
                        # Clean column names
                        chunk.columns = clean_column_names_func(chunk.columns)
                        
                        # Remove rows with all NaN values
                        chunk = chunk.dropna(how='all')
                        
                        # Further split if the row group is larger than chunk size
                        total_group_rows = len(chunk)
                        
                        for j in range(0, total_group_rows, chunk_size):
                            end = min(j + chunk_size, total_group_rows)
                            all_chunks.append(chunk.iloc[j:end])
                    
                    return all_chunks
                        
                except ImportError:
                    # Fall back to pandas if pyarrow is not available
                    progress_emit_func("PyArrow not available, falling back to pandas for Parquet reading")
                    df = pd.read_parquet(file_path_str)
                    
                    # Clean column names
                    df.columns = clean_column_names_func(df.columns)
                    
                    # Remove rows with all NaN values
                    df = df.dropna(how='all')
                    
                    # Process in chunks
                    total_rows = len(df)
                    chunks = []
                    for i in range(0, total_rows, chunk_size):
                        end = min(i + chunk_size, total_rows)
                        chunks.append(df.iloc[i:end])
                    
                    return chunks
            else:
                self.progress.emit(f"Unsupported file format: {file_ext}")
                raise ValueError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            self.error.emit(f"Error reading file {self.file_path}: {str(e)}")
            return None
    
    def clean_column_names(self, columns):
        """Clean and deduplicate column names"""
        # Convert columns to list of strings
        cols = [str(col) if col is not None else f"Column_{i+1}" for i, col in enumerate(columns)]
        
        # Replace empty strings with placeholder names
        for i in range(len(cols)):
            if cols[i].strip() == '':
                cols[i] = f"Column_{i+1}"
        
        # Handle duplicate column names
        seen = {}
        for i in range(len(cols)):
            col = cols[i]
            if col in seen:
                seen[col] += 1
                cols[i] = f"{col}_{seen[col]}"
            else:
                seen[col] = 0
                
        return cols
    
    def get_existing_columns(self, conn):
        """Get existing columns from the table"""
        try:
            if self.is_duckdb:
                result = conn.execute(f"PRAGMA table_info('{self.table_name}')").fetchall()
                return [row[1] for row in result]  # Column name is at index 1
            else:
                import sqlite3
                cursor = conn.cursor()
                result = cursor.execute(f"PRAGMA table_info('{self.table_name}')").fetchall()
                return [row[1] for row in result]  # Column name is at index 1
        except Exception as e:
            self.progress.emit(f"Error getting existing columns: {str(e)}")
            return []
    
    def add_missing_columns(self, conn, df, existing_columns):
        """Add missing columns to the existing table"""
        new_columns = []
        column_types = self.analyze_column_types(df)
        
        # Create a mapping of lowercase column names to actual column names to check for case-insensitive duplicates
        existing_columns_lower = {col.lower(): col for col in existing_columns}
        
        for col in df.columns:
            # Check if column already exists (case-insensitive)
            if col.lower() not in existing_columns_lower:
                new_columns.append((col, column_types[col]))
        
        if not new_columns:
            return
            
        self.progress.emit(f"Adding {len(new_columns)} new columns to the table...")
        
        for col_name, col_type in new_columns:
            added = False
            errors = []
            
            # Try different quoting styles for column names
            quoting_styles = [
                f'"{col_name}"',  # Double quotes
                f'[{col_name}]',  # Square brackets
                f'`{col_name}`',  # Backticks
                col_name          # No quotes
            ]
            
            for quoted_col in quoting_styles:
                if added:
                    break
                    
                try:
                    if self.is_duckdb:
                        conn.execute(f"ALTER TABLE \"{self.table_name}\" ADD COLUMN {quoted_col} {col_type}")
                        added = True
                    else:
                        cursor = conn.cursor()
                        cursor.execute(f"ALTER TABLE \"{self.table_name}\" ADD COLUMN {quoted_col} {col_type}")
                        conn.commit()
                        added = True
                except Exception as e:
                    # If column already exists with a different case, skip it
                    if "duplicate column name" in str(e).lower():
                        self.progress.emit(f"Column \"{col_name}\" already exists with a different case, skipping...")
                        added = True  # Consider it added since it exists
                        break
                    else:
                        errors.append(f"Error with {quoted_col}: {str(e)}")
            
            if not added:
                # Just log the error but don't stop processing
                self.progress.emit(f"Could not add column {col_name}. Errors: {'; '.join(errors)}")
                
        self.progress.emit(f"Added new columns to the table.")
    
    def create_table_if_not_exists(self, conn, df):
        """Create table with appropriate column types if it doesn't exist"""
        column_types = self.analyze_column_types(df)
        
        # Build CREATE TABLE statement
        columns_sql = ', '.join([f'"{col}" {dtype}' for col, dtype in column_types.items()])
        create_table_sql = f"CREATE TABLE IF NOT EXISTS \"{self.table_name}\" ({columns_sql})"
        
        # Execute the statement
        if self.is_duckdb:
            conn.execute(create_table_sql)
        else:
            cursor = conn.cursor()
            cursor.execute(create_table_sql)
            conn.commit()
    
    def analyze_column_types(self, df):
        """Analyze column types to ensure proper database insertion"""
        column_types = {}
        
        for col in df.columns:
            # Check for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                if pd.api.types.is_integer_dtype(df[col]):
                    column_types[col] = 'BIGINT'
                else:
                    column_types[col] = 'DOUBLE'
            # Check for datetime columns
            elif pd.api.types.is_datetime64_dtype(df[col]):
                column_types[col] = 'TIMESTAMP'
            # Check for boolean columns
            elif pd.api.types.is_bool_dtype(df[col]):
                column_types[col] = 'BOOLEAN'
            # Default to text for other types
            else:
                column_types[col] = 'TEXT'
                
        return column_types
    
    def get_existing_columns(self, conn):
        """Get existing columns from the table"""
        try:
            if self.is_duckdb:
                result = conn.execute(f"PRAGMA table_info('{self.table_name}')").fetchall()
                return [row[1] for row in result]  # Column name is at index 1
            else:
                import sqlite3
                cursor = conn.cursor()
                result = cursor.execute(f"PRAGMA table_info('{self.table_name}')").fetchall()
                return [row[1] for row in result]  # Column name is at index 1
        except Exception as e:
            self.progress.emit(f"Error getting existing columns: {str(e)}")
            return []
    
    def add_missing_columns(self, conn, df, existing_columns):
        """Add missing columns to the existing table"""
        new_columns = []
        column_types = self.analyze_column_types(df)
        
        # Create a mapping of lowercase column names to actual column names to check for case-insensitive duplicates
        existing_columns_lower = {col.lower(): col for col in existing_columns}
        
        for col in df.columns:
            # Check if column already exists (case-insensitive)
            if col.lower() not in existing_columns_lower:
                new_columns.append((col, column_types[col]))
        
        if not new_columns:
            return
            
        self.progress.emit(f"Adding {len(new_columns)} new columns to the table...")
        
        for col_name, col_type in new_columns:
            added = False
            errors = []
            
            # Try different quoting styles for column names
            quoting_styles = [
                f'"{col_name}"',  # Double quotes
                f'[{col_name}]',  # Square brackets
                f'`{col_name}`',  # Backticks
                col_name          # No quotes
            ]
            
            for quoted_col in quoting_styles:
                if added:
                    break
                    
                try:
                    if self.is_duckdb:
                        conn.execute(f"ALTER TABLE \"{self.table_name}\" ADD COLUMN {quoted_col} {col_type}")
                        added = True
                    else:
                        cursor = conn.cursor()
                        cursor.execute(f"ALTER TABLE \"{self.table_name}\" ADD COLUMN {quoted_col} {col_type}")
                        conn.commit()
                        added = True
                except Exception as e:
                    # If column already exists with a different case, skip it
                    if "duplicate column name" in str(e).lower():
                        self.progress.emit(f"Column \"{col_name}\" already exists with a different case, skipping...")
                        added = True  # Consider it added since it exists
                        break
                    else:
                        errors.append(f"Error with {quoted_col}: {str(e)}")
            
            if not added:
                # Just log the error but don't stop processing
                self.progress.emit(f"Could not add column {col_name}. Errors: {'; '.join(errors)}")
                
        self.progress.emit(f"Added new columns to the table.")
    
    def create_table_if_not_exists(self, conn, df):
        """Create table with appropriate column types if it doesn't exist"""
        column_types = self.analyze_column_types(df)
        
        # Build CREATE TABLE statement
        columns_sql = ', '.join([f'"{col}" {dtype}' for col, dtype in column_types.items()])
        create_table_sql = f"CREATE TABLE IF NOT EXISTS \"{self.table_name}\" ({columns_sql})"
        
        # Execute the statement
        if self.is_duckdb:
            conn.execute(create_table_sql)
        else:
            cursor = conn.cursor()
            cursor.execute(create_table_sql)
            conn.commit()
    
    def read_file(self):
        """Read file in chunks to avoid memory issues"""
        # Convert Path object to string if needed
        file_path_str = str(self.file_path)
        file_ext = self.file_path.suffix.lower() if hasattr(self.file_path, 'suffix') else os.path.splitext(file_path_str)[1].lower()
        
        try:
            if file_ext == '.csv':
                # Try with default settings first
                try:
                    # Try to count total lines for progress reporting
                    try:
                        with open(file_path_str, 'r') as f:
                            total_lines = sum(1 for _ in f)
                        self.progress.emit(f"CSV file has approximately {total_lines} lines")
                    except Exception:
                        pass  # Ignore if we can't count lines
                        
                    # Process with reasonable chunk size
                    for chunk in pd.read_csv(file_path_str, chunksize=self.CHUNK_SIZE):
                        # Clean column names
                        chunk.columns = self.clean_column_names(chunk.columns)
                        # Remove rows with all NaN values
                        chunk = chunk.dropna(how='all')
                        yield chunk
                except Exception as e:
                    self.progress.emit(f"Error reading CSV with default settings, trying more flexible settings: {str(e)}")
                    # Try with more flexible settings
                    for chunk in pd.read_csv(file_path_str, chunksize=self.CHUNK_SIZE, 
                                          encoding='latin1', on_bad_lines='skip', 
                                          low_memory=False, dtype=str):
                        # Clean column names
                        chunk.columns = self.clean_column_names(chunk.columns)
                        # Remove rows with all NaN values
                        chunk = chunk.dropna(how='all')
                        yield chunk
                        
            elif file_ext in ['.xlsx', '.xls']:
                # For Excel, we need to read the entire file at once
                # but we can process it in chunks
                
                # First, check if the file is actually a valid file and exists
                if not os.path.exists(file_path_str):
                    raise FileNotFoundError(f"Excel file not found: {file_path_str}")
                
                # Check file size - if it's too small it's probably corrupt or empty
                file_size = os.path.getsize(file_path_str)
                if file_size < 100:  # Extremely small for an Excel file
                    raise ValueError(f"Excel file is likely corrupt or empty (size: {file_size} bytes)")
                
                # Try multiple engines with proper error handling
                df = None
                success = False
                error_messages = []
                
                # Try with openpyxl first for .xlsx
                if file_ext == '.xlsx':
                    try:
                        self.progress.emit(f"Trying to read Excel file with openpyxl...")
                        df = pd.read_excel(file_path_str, engine='openpyxl')
                        success = True
                    except Exception as e:
                        error_messages.append(f"openpyxl error: {str(e)}")
                        self.progress.emit(f"Error with openpyxl: {str(e)}")
                
                # If .xls or openpyxl failed, try xlrd
                if not success:
                    try:
                        self.progress.emit(f"Trying to read Excel file with xlrd...")
                        df = pd.read_excel(file_path_str, engine='xlrd')
                        success = True
                    except Exception as e:
                        error_messages.append(f"xlrd error: {str(e)}")
                        self.progress.emit(f"Error with xlrd: {str(e)}")
                
                # If still no success, try as CSV as a last resort
                if not success:
                    try:
                        self.progress.emit("Trying to read file as CSV instead...")
                        df = pd.read_csv(file_path_str)
                        success = True
                    except Exception as e:
                        error_messages.append(f"CSV fallback error: {str(e)}")
                        self.progress.emit(f"CSV fallback failed: {str(e)}")
                
                if not success:
                    raise Exception(f"Could not read Excel file with any available engine: {' | '.join(error_messages)}")
                
                # Clean column names
                df.columns = self.clean_column_names(df.columns)
                
                # Remove rows with all NaN values
                df = df.dropna(how='all')
                
                # Process in chunks
                total_rows = len(df)
                self.progress.emit(f"Successfully read Excel file with {total_rows} rows. Processing in chunks...")
                for i in range(0, total_rows, self.CHUNK_SIZE):
                    end = min(i + self.CHUNK_SIZE, total_rows)
                    yield df.iloc[i:end]
                    
            elif file_ext == '.parquet':
                # Try using pyarrow for chunked reading
                try:
                    import pyarrow.parquet as pq
                    
                    # Open the file
                    parquet_file = pq.ParquetFile(file_path_str)
                    
                    # Get metadata
                    total_rows = parquet_file.metadata.num_rows
                    num_row_groups = parquet_file.num_row_groups
                    
                    self.progress.emit(f"Parquet file has {total_rows} rows in {num_row_groups} row groups")
                    
                    # Process each row group
                    for i in range(num_row_groups):
                        # Read row group
                        table = parquet_file.read_row_group(i)
                        
                        # Convert to pandas DataFrame
                        chunk = table.to_pandas()
                        
                        # Clean column names
                        chunk.columns = self.clean_column_names(chunk.columns)
                        
                        # Remove rows with all NaN values
                        chunk = chunk.dropna(how='all')
                        
                        # Further split if the row group is larger than chunk size
                        total_group_rows = len(chunk)
                        
                        for j in range(0, total_group_rows, self.CHUNK_SIZE):
                            end = min(j + self.CHUNK_SIZE, total_group_rows)
                            yield chunk.iloc[j:end]
                        
                except ImportError:
                    # Fall back to pandas if pyarrow is not available
                    self.progress.emit("PyArrow not available, falling back to pandas for Parquet reading")
                    df = pd.read_parquet(file_path_str)
                    
                    # Clean column names
                    df.columns = self.clean_column_names(df.columns)
                    
                    # Remove rows with all NaN values
                    df = df.dropna(how='all')
                    
                    # Process in chunks
                    total_rows = len(df)
                    for i in range(0, total_rows, self.CHUNK_SIZE):
                        end = min(i + self.CHUNK_SIZE, total_rows)
                        yield df.iloc[i:end]
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
        except Exception as e:
            self.error.emit(f"Error reading file: {str(e)}")
            return None
    
    def cancel(self):
        self.is_cancelled = True
        self.progress.emit("Cancelling operation...")

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
        
        # Create New Database button with a database-like icon
        self.create_db_button = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_DriveHDIcon), "Create Database", self)
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
        
        # Add Import Single File button with a distinct icon
        self.import_file_button = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_FileLinkIcon), "Import File", self)
        self.import_file_button.setStatusTip("Import a single file (CSV, Excel, Parquet) into database")
        self.import_file_button.triggered.connect(self.import_single_file)
        self.toolbar.addAction(self.import_file_button)
        
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
            replace_table = dialog.replace_table
            
            # Create a progress dialog
            progress_dialog = QProgressDialog("Merging files...", "Cancel", 0, 0, self)
            progress_dialog.setWindowTitle("Merge Progress")
            progress_dialog.setMinimumDuration(0)
            progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            progress_dialog.setAutoClose(False)
            progress_dialog.setAutoReset(False)
            
            # Create the worker
            self.merge_worker = MergeFilesWorker(folder_path, db_path, table_name, use_existing_table, replace_table)
            
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
        """Handle signal when a table is created by import or merge process"""
        # Update the database tables list
        try:
            # First check if we're already connected to this database
            current_db = self.db_selector.currentText()
            if current_db == db_path:
                # Simply refresh the table list without switching databases
                self.query_tab.update_table_list()
            else:
                # Switch to the database where the table was created
                self.db_selector.setCurrentText(db_path)
                # This will trigger on_database_selected which updates the table list
        except Exception as e:
            QMessageBox.warning(self, "Update Error", f"Failed to update table list: {str(e)}")
            
        # Focus the query tab
        self.query_tab.setFocus()

    def import_single_file(self):
        """Open dialog to import a single file into a database"""
        # Get list of available databases
        available_dbs = []
        for i in range(self.db_selector.count()):
            db_text = self.db_selector.itemText(i)
            if db_text != "No database loaded":
                available_dbs.append(db_text)
        
        # Create and show the import file dialog
        dialog = ImportFileDialog(self, available_dbs)
        result = dialog.exec()
        
        if result == QDialog.DialogCode.Accepted:
            # Get the values from the dialog
            file_path = dialog.file_path
            db_path = dialog.db_path
            table_name = dialog.table_name
            use_existing_table = dialog.use_existing_table
            replace_table = dialog.replace_table
            
            # Create a progress dialog
            progress_dialog = QProgressDialog("Importing file...", "Cancel", 0, 0, self)
            progress_dialog.setWindowTitle("Import Progress")
            progress_dialog.setMinimumDuration(0)
            progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            progress_dialog.setAutoClose(False)
            progress_dialog.setAutoReset(False)
            
            # Create the worker
            self.import_worker = ImportFileWorker(file_path, db_path, table_name, use_existing_table, replace_table, dialog.import_options)
            
            # Create a thread to run the worker
            self.import_thread = QThread()
            self.import_worker.moveToThread(self.import_thread)
            
            # Connect signals
            self.import_worker.progress.connect(progress_dialog.setLabelText)
            self.import_worker.error.connect(lambda msg: QMessageBox.critical(self, "Error", msg))
            self.import_worker.success.connect(lambda msg: QMessageBox.information(self, "Success", msg))
            self.import_worker.finished.connect(progress_dialog.close)
            self.import_worker.finished.connect(self.import_thread.quit)
            
            # Connect table created signal to refresh tables
            self.import_worker.table_created.connect(self.on_table_created)
            
            # Connect thread signals
            self.import_thread.started.connect(self.import_worker.run)
            self.import_thread.finished.connect(self.import_thread.deleteLater)
            
            # Connect cancel button
            progress_dialog.canceled.connect(self.import_worker.cancel)
            
            # Start the thread
            progress_dialog.show()
            self.import_thread.start()
            
            # If this is a new database, add it to the selector
            if self.db_selector.findText(db_path) == -1:
                self.db_selector.addItem(db_path)
                self.db_selector.setCurrentText(db_path)
                self.db_selector.setEnabled(True)
                self.close_db_button.setEnabled(True)
                self.query_tab.switch_to_database(db_path)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
