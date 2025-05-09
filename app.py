import sys
import duckdb
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QListWidget, QTextEdit, QTableWidget,
    QTableWidgetItem, QLabel, QMessageBox, QSplitter, QHeaderView,
    QInputDialog, QProgressDialog, QMenu
)
from PyQt6.QtGui import QPalette, QColor, QAction, QSyntaxHighlighter, QTextCharFormat, QFont
from PyQt6.QtCore import Qt, QRegularExpression, QThread, QObject, pyqtSignal
import re
import pandas as pd
import csv

DARK_STYLESHEET = """
    QWidget {
        background-color: #2b2b2b;
        color: #ffffff;
        font-size: 10pt;
    }
    QMainWindow {
        background-color: #2b2b2b;
    }
    QMenuBar {
        background-color: #3c3c3c;
        color: #ffffff;
    }
    QMenuBar::item {
        background-color: #3c3c3c;
        color: #ffffff;
    }
    QMenuBar::item:selected {
        background-color: #555555;
    }
    QMenu {
        background-color: #3c3c3c;
        color: #ffffff;
        border: 1px solid #555555;
    }
    QMenu::item:selected {
        background-color: #555555;
    }
    QPushButton {
        background-color: #555555;
        color: #ffffff;
        border: 1px solid #666666;
        padding: 5px;
        min-height: 15px;
        border-radius: 4px;
    }
    QPushButton:hover {
        background-color: #666666;
    }
    QPushButton:pressed {
        background-color: #444444;
    }
    QListWidget {
        background-color: #3c3c3c;
        color: #ffffff;
        border: 1px solid #555555;
        padding: 5px;
    }
    QTextEdit {
        background-color: #3c3c3c;
        color: #ffffff;
        border: 1px solid #555555;
        padding: 5px;
    }
    QTableWidget {
        background-color: #3c3c3c;
        color: #ffffff;
        border: 1px solid #555555;
        gridline-color: #555555;
    }
    QTableWidget::item {
        padding: 5px;
    }
    QHeaderView::section {
        background-color: #555555;
        color: #ffffff;
        padding: 4px;
        border: 1px solid #666666;
    }
    QLabel {
        color: #ffffff;
    }
    QSplitter::handle {
        background-color: #555555;
        border: 1px solid #666666;
        height: 5px; /* Vertical splitter */
        width: 5px; /* Horizontal splitter */
    }
    QSplitter::handle:hover {
        background-color: #666666;
    }
    QSplitter::handle:pressed {
        background-color: #444444;
    }
    QMessageBox {
        background-color: #2b2b2b;
    }
    QMessageBox QLabel {
        color: #ffffff;
    }
    QMessageBox QPushButton {
        background-color: #555555;
        color: #ffffff;
        border: 1px solid #666666;
        padding: 5px;
        min-width: 70px;
    }
"""

class SQLHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)

        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#569cd6")) # Blue color for keywords
        keyword_format.setFontWeight(QFont.Weight.Bold)
        keywords = [
            "\\bSELECT\\b", "\\bFROM\\b", "\\bWHERE\\b", "\\bINSERT\\b", "\\bUPDATE\\b",
            "\\bDELETE\\b", "\\bCREATE\\b", "\\bALTER\\b", "\\bDROP\\b", "\\bTABLE\\b",
            "\\bVIEW\\b", "\\bINDEX\\b", "\\bON\\b", "\\bJOIN\\b", "\\bINNER\\b", "\\bLEFT\\b",
            "\\bRIGHT\\b", "\\bOUTER\\b", "\\bGROUP\\b", "\\bBY\\b", "\\bORDER\\b", "\\bLIMIT\\b",
            "\\bAS\\b", "\\bDISTINCT\\b", "\\bCOUNT\\b", "\\bSUM\\b", "\\bAVG\\b", "\\bMAX\\b", "\\bMIN\\b",
            "\\bAND\\b", "\\bOR\\b", "\\bNOT\\b", "\\bNULL\\b", "\\bIS\\b", "\\bTRUE\\b", "\\bFALSE\\b",
            "\\bLIKE\\b", "\\bIN\\b", "\\bBETWEEN\\b", "\\bCASE\\b", "\\bWHEN\\b", "\\bTHEN\\b",
            "\\bELSE\\b", "\\bEND\\b", "\\bSHOW\\b", "\\bDESCRIBE\\b", "\\bPRAGMA\\b"
        ]
        self.highlighting_rules = [(QRegularExpression(pattern, QRegularExpression.PatternOption.CaseInsensitiveOption), keyword_format) for pattern in keywords]

        # Optional: Add rules for strings and comments
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#ce9178")) # Orange/brown for strings
        self.highlighting_rules.append((QRegularExpression("'.*?'"), string_format))
        self.highlighting_rules.append((QRegularExpression("\"[^\"]*\""), string_format)) # Double quotes too

        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#6a9955")) # Green for comments
        comment_format.setFontItalic(True)
        self.highlighting_rules.append((QRegularExpression("--.*"), comment_format))

    def highlightBlock(self, text):
        for pattern, format in self.highlighting_rules:
            expression = QRegularExpression(pattern)
            it = expression.globalMatch(text)
            while it.hasNext():
                match = it.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), format)

# --- Worker for Background Tasks ---
class ImportWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    success = pyqtSignal(str)

    def __init__(self, db_conn_func, import_func, *args):
        super().__init__()
        self.db_conn_func = db_conn_func # Function to get a NEW db connection in this thread
        self.import_func = import_func
        self.args = args
        self.is_cancelled = False

    def run(self):
        db_conn = None
        try:
            # IMPORTANT: Create a new DB connection *within* the worker thread
            db_conn = self.db_conn_func()
            if db_conn is None:
                raise ConnectionError("Failed to establish database connection in worker thread.")

            print(f"Worker running import func: {self.import_func.__name__}")

            # Pass the worker reference itself for cancellation checks
            # and the thread-local connection
            all_args = (db_conn,) + self.args + (self,)

            # Execute the core import logic function with unpacked arguments
            success_message = self.import_func(*all_args)

            # Emit success if no exception occurred
            if success_message:
                self.success.emit(success_message)

        except InterruptedError as interrupt_e: # Catch cancellation specifically
            error_message = f"Import cancelled: {interrupt_e}"
            print(error_message)
            self.error.emit(error_message)
        except Exception as e:
            # Add traceback for better debugging
            import traceback
            tb_str = traceback.format_exc()
            error_message = f"Error during import: {e}\n{tb_str}"
            print(error_message)
            self.error.emit(error_message)
        finally:
            if db_conn:
                db_conn.close()
                print("Worker DB connection closed.")
            self.finished.emit()

    def cancel(self):
        print("Worker received cancel signal.")
        self.is_cancelled = True

# --- Worker for SQL Queries ---
class QueryWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    success = pyqtSignal(list, list, int) # Emit data (list of rows), headers (list of strings), and total count
    progress = pyqtSignal(int, int)  # current, total

    def __init__(self, db_conn_func, query, limit=1000, offset=0):
        super().__init__()
        self.db_conn_func = db_conn_func
        self.query = query
        self.limit = limit
        self.offset = offset
        self.is_cancelled = False
        
    def cancel(self):
        self.is_cancelled = True
        print("Query worker received cancel signal.")

    def run(self):
        db_conn = None
        try:
            db_conn = self.db_conn_func()
            if db_conn is None:
                raise ConnectionError("Failed to establish database connection in query worker thread.")

            print(f"Query Worker executing: {self.query[:100]}...")
            
            # First check if this is a SELECT query and get total count if it is
            is_select = self.query.strip().upper().startswith("SELECT")
            total_count = 0
            
            if is_select:
                # Wrap original query in a count query to get total rows
                # Use a subquery to ensure we're counting the actual result set
                count_query = f"SELECT COUNT(*) FROM ({self.query}) AS count_subquery"
                try:
                    count_result = db_conn.execute(count_query).fetchone()
                    if count_result:
                        total_count = count_result[0]
                        print(f"Total result count: {total_count}")
                except Exception as count_e:
                    print(f"Warning: Could not get count: {count_e}")
                    # Continue anyway - we'll still fetch what we can
                
                # If it's a SELECT query, add LIMIT and OFFSET for pagination
                # Only if the query doesn't already have a LIMIT clause
                if not re.search(r'\bLIMIT\b\s+\d+', self.query, re.IGNORECASE):
                    paginated_query = f"{self.query} LIMIT {self.limit} OFFSET {self.offset}"
                else:
                    paginated_query = self.query
            else:
                paginated_query = self.query

            # Execute the query (paginated for SELECT)
            result_relation = db_conn.execute(paginated_query)

            # Check if cancelled
            if self.is_cancelled:
                raise InterruptedError("Query execution was cancelled")

            data = []
            headers = []
            if result_relation.description:
                headers = [desc[0] for desc in result_relation.description]
                
                # Use fetchmany for more control and to avoid loading everything at once
                fetched_count = 0
                while True:
                    # Fetch a batch of rows (100 at a time)
                    batch = result_relation.fetchmany(100)
                    if not batch:
                        break
                        
                    data.extend(batch)
                    fetched_count += len(batch)
                    
                    # Report progress
                    if total_count > 0:
                        self.progress.emit(fetched_count, min(total_count, self.limit))
                    
                    # Check if we should stop (either cancelled or hit limit)
                    if self.is_cancelled:
                        raise InterruptedError("Query execution was cancelled")
                        
                    if fetched_count >= self.limit:
                        break
                
                print(f"Query Worker fetched {len(data)} rows.")
                self.success.emit(data, headers, total_count)
            else:
                # Non-SELECT query (no data/headers to emit, but signal success)
                print("Query Worker executed non-SELECT query.")
                # Emit empty lists to signify non-SELECT success, still include total count
                self.success.emit([], [], 0)

        except InterruptedError as interrupt_e:
            error_message = f"Query cancelled: {interrupt_e}"
            print(error_message)
            self.error.emit(error_message)
        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            error_message = f"Error executing query: {e}\n{tb_str}"
            print(error_message)
            self.error.emit(error_message)
        finally:
            if db_conn:
                db_conn.close()
                print("Query Worker DB connection closed.")
            self.finished.emit()

# --- Worker for Exports ---
class ExportWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    success = pyqtSignal(str)
    progress = pyqtSignal(int, int)  # current, total
    
    def __init__(self, export_type, data, file_path, **kwargs):
        super().__init__()
        self.export_type = export_type  # 'csv', 'excel', or 'parquet'
        self.data = data
        self.file_path = file_path
        self.kwargs = kwargs
        self.is_cancelled = False
    
    def cancel(self):
        self.is_cancelled = True
        print("Export worker received cancel signal.")
    
    def run(self):
        try:
            # Report start
            self.progress.emit(0, 100)
            
            # Export batches of data to avoid freezing
            total_rows = len(self.data)
            batch_size = 5000  # Adjust based on your needs
            
            if self.export_type == 'csv':
                self.export_to_csv()
            elif self.export_type == 'excel':
                self.export_to_excel()
            elif self.export_type == 'parquet':
                self.export_to_parquet()
            else:
                raise ValueError(f"Unknown export type: {self.export_type}")
            
            self.success.emit(f"Data exported successfully to {self.file_path}")
        
        except InterruptedError as interrupt_e:
            error_message = f"Export cancelled: {interrupt_e}"
            print(error_message)
            self.error.emit(error_message)
        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            error_message = f"Error during export: {e}\n{tb_str}"
            print(error_message)
            self.error.emit(error_message)
        finally:
            self.finished.emit()
    
    def export_to_csv(self):
        """Export data to CSV file with progress updates."""
        import csv
        
        delimiter = self.kwargs.get('delimiter', ',')
        total_rows = len(self.data)
        batch_size = 5000
        
        try:
            with open(self.file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=delimiter)
                
                # Write header
                writer.writerow(self.data.columns)
                
                # Write data in batches
                rows_written = 0
                for start_idx in range(0, total_rows, batch_size):
                    if self.is_cancelled:
                        raise InterruptedError("Export was cancelled")
                        
                    end_idx = min(start_idx + batch_size, total_rows)
                    batch = self.data.iloc[start_idx:end_idx]
                    
                    # Write each row in the batch
                    for _, row in batch.iterrows():
                        writer.writerow(row)
                    
                    rows_written += len(batch)
                    self.progress.emit(rows_written, total_rows)
                    
                    # Small sleep to keep UI responsive
                    QThread.msleep(1)
        except Exception as e:
            print(f"CSV export error: {e}")
            raise
    
    def export_to_excel(self):
        """Export data to Excel file with progress updates."""
        total_rows = len(self.data)
        
        try:
            # For very large datasets, use the openpyxl engine with batch processing
            if total_rows > 100000:
                self._export_excel_batched()
            else:
                # For smaller datasets, use the simpler approach
                self.progress.emit(10, 100)  # Starting
                self.data.to_excel(self.file_path, index=False)
                self.progress.emit(100, 100)  # Completed
        except Exception as e:
            print(f"Excel export error: {e}")
            raise
    
    def _export_excel_batched(self):
        """Export data to Excel in batches for very large datasets."""
        import openpyxl
        from openpyxl.utils.dataframe import dataframe_to_rows
        
        total_rows = len(self.data)
        batch_size = 5000
        
        # Create a new workbook
        wb = openpyxl.Workbook()
        ws = wb.active
        
        # Add headers
        self.progress.emit(1, 100)
        headers = list(self.data.columns)
        ws.append(headers)
        
        # Write data in batches
        rows_written = 0
        for start_idx in range(0, total_rows, batch_size):
            if self.is_cancelled:
                raise InterruptedError("Export was cancelled")
                
            end_idx = min(start_idx + batch_size, total_rows)
            batch = self.data.iloc[start_idx:end_idx]
            
            # Add each row from the batch to the worksheet
            for row in dataframe_to_rows(batch, index=False, header=False):
                ws.append(row)
            
            rows_written += len(batch)
            progress = int(min((rows_written / total_rows) * 100, 99))
            self.progress.emit(progress, 100)
            
            # Small sleep to keep UI responsive
            QThread.msleep(1)
        
        # Save the workbook
        wb.save(self.file_path)
        self.progress.emit(100, 100)
    
    def export_to_parquet(self):
        """Export data to Parquet file."""
        try:
            self.progress.emit(10, 100)  # Starting
            
            # Parquet is efficient and doesn't need the same batching approach
            # The pyarrow library handles large datasets well
            self.data.to_parquet(self.file_path, index=False)
            
            self.progress.emit(100, 100)  # Completed
        except Exception as e:
            print(f"Parquet export error: {e}")
            raise

class DuckDBApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DuckDB Query Tool")
        self.setGeometry(100, 100, 1200, 800)
        self.db_conn = None
        self.db_path = None
        self.recent_dbs = []
        self.max_recent_dbs = 5
        
        # Load recent databases
        self.load_recent_dbs()

        self.init_ui()
        self.apply_dark_theme()

    def load_recent_dbs(self):
        """Load list of recently opened databases from file."""
        try:
            recent_dbs_file = os.path.join(os.path.expanduser("~"), ".duckdb_recent")
            if os.path.exists(recent_dbs_file):
                with open(recent_dbs_file, "r") as f:
                    self.recent_dbs = [line.strip() for line in f.readlines() if os.path.exists(line.strip())]
                    # Only keep existing files
                    self.recent_dbs = self.recent_dbs[:self.max_recent_dbs]
        except Exception as e:
            print(f"Error loading recent databases: {e}")
            self.recent_dbs = []
            
    def save_recent_dbs(self):
        """Save list of recently opened databases to file."""
        try:
            recent_dbs_file = os.path.join(os.path.expanduser("~"), ".duckdb_recent")
            with open(recent_dbs_file, "w") as f:
                for db_path in self.recent_dbs:
                    f.write(f"{db_path}\n")
        except Exception as e:
            print(f"Error saving recent databases: {e}")
            
    def add_to_recent_dbs(self, db_path):
        """Add a database path to the recent list."""
        if db_path in self.recent_dbs:
            # Move to top of list if already exists
            self.recent_dbs.remove(db_path)
        self.recent_dbs.insert(0, db_path)
        # Keep only max number
        self.recent_dbs = self.recent_dbs[:self.max_recent_dbs]
        self.save_recent_dbs()
        self.update_recent_menu()

    def apply_dark_theme(self):
        self.setStyleSheet(DARK_STYLESHEET)

    def init_ui(self):
        # --- Menu Bar ---
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        new_action = QAction("&New Database", self)
        new_action.triggered.connect(self.create_db)
        file_menu.addAction(new_action)

        open_action = QAction("&Open Database", self)
        open_action.triggered.connect(self.open_db)
        file_menu.addAction(open_action)
        
        # Recent databases submenu
        self.recent_menu = file_menu.addMenu("Open &Recent")
        self.update_recent_menu()

        close_action = QAction("&Close Database", self)
        close_action.triggered.connect(self.close_db)
        file_menu.addAction(close_action)

        file_menu.addSeparator()

        exit_action = QAction("&Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # --- Import Menu ---
        import_menu = menubar.addMenu("&Import")

        import_csv_action = QAction("Import &CSV...", self)
        import_csv_action.triggered.connect(self.import_csv)
        import_menu.addAction(import_csv_action)

        import_parquet_action = QAction("Import &Parquet...", self)
        import_parquet_action.triggered.connect(self.import_parquet)
        import_menu.addAction(import_parquet_action)

        import_excel_action = QAction("Import &Excel...", self)
        import_excel_action.triggered.connect(self.import_excel)
        import_menu.addAction(import_excel_action)
        
        # --- Export Menu ---
        export_menu = menubar.addMenu("&Export")
        
        export_csv_action = QAction("Export as &CSV...", self)
        export_csv_action.triggered.connect(self.export_to_csv)
        export_menu.addAction(export_csv_action)
        
        export_excel_action = QAction("Export as &Excel...", self)
        export_excel_action.triggered.connect(self.export_to_excel)
        export_menu.addAction(export_excel_action)
        
        export_parquet_action = QAction("Export as &Parquet...", self)
        export_parquet_action.triggered.connect(self.export_to_parquet)
        export_menu.addAction(export_parquet_action)

        # --- Central Widget & Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Top Section (DB Info) ---
        self.db_status_label = QLabel("No database loaded.")
        main_layout.addWidget(self.db_status_label)

        # --- Main Content Area (Splitter) ---
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter, 1) # Give splitter stretch factor

        # --- Left Pane (Tables List) ---
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.addWidget(QLabel("Tables:"))
        self.table_list_widget = QListWidget()
        self.table_list_widget.currentItemChanged.connect(self.display_table_schema) 
        # Enable context menu
        self.table_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table_list_widget.customContextMenuRequested.connect(self.show_table_context_menu)
        left_layout.addWidget(self.table_list_widget)
        main_splitter.addWidget(left_pane)

        # --- Right Pane (Query and Results - Splitter) ---
        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_layout.addWidget(right_splitter)
        main_splitter.addWidget(right_pane)

        # --- Query Editor ---
        query_pane = QWidget()
        query_layout = QVBoxLayout(query_pane)
        query_layout.addWidget(QLabel("SQL Query:"))
        self.query_editor = QTextEdit()
        self.query_editor.setPlaceholderText("Enter your SQL query here...")
        # Apply SQL syntax highlighting
        self.highlighter = SQLHighlighter(self.query_editor.document())
        query_layout.addWidget(self.query_editor)
        run_query_button = QPushButton("Run Query")
        run_query_button.setToolTip("Execute the full query or just the selected text if a selection is made")
        run_query_button.clicked.connect(self.execute_query) # Connect here
        query_layout.addWidget(run_query_button)
        right_splitter.addWidget(query_pane)

        # --- Results Table ---
        results_pane = QWidget()
        results_layout = QVBoxLayout(results_pane)
        results_layout.addWidget(QLabel("Results:"))
        self.results_table = QTableWidget()
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers) # Read-only
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.results_table.verticalHeader().setVisible(False) # Hide row numbers by default
        results_layout.addWidget(self.results_table)
        right_splitter.addWidget(results_pane)

        # Adjust splitter sizes (optional initial split)
        main_splitter.setSizes([200, 1000]) # Adjust initial width split
        right_splitter.setSizes([300, 500]) # Adjust initial height split

        # Set initial focus
        self.query_editor.setFocus()

    def update_recent_menu(self):
        """Update the recent databases menu with current list."""
        self.recent_menu.clear()
        if not self.recent_dbs:
            no_recent = QAction("No Recent Databases", self)
            no_recent.setEnabled(False)
            self.recent_menu.addAction(no_recent)
            return
            
        for db_path in self.recent_dbs:
            # Create display name (just the filename)
            display_name = os.path.basename(db_path)
            recent_action = QAction(display_name, self)
            # Store the full path as data in the action
            recent_action.setData(db_path)
            recent_action.setStatusTip(db_path)
            recent_action.triggered.connect(self.open_recent_db)
            self.recent_menu.addAction(recent_action)
            
        self.recent_menu.addSeparator()
        clear_action = QAction("Clear Recent", self)
        clear_action.triggered.connect(self.clear_recent_dbs)
        self.recent_menu.addAction(clear_action)
        
    def open_recent_db(self):
        """Open a database from the recent list."""
        action = self.sender()
        if action and isinstance(action, QAction):
            db_path = action.data()
            if db_path and os.path.exists(db_path):
                self.connect_db(db_path)
            else:
                QMessageBox.warning(self, "Warning", f"Database file not found: {db_path}")
                # Remove invalid path from recent list
                if db_path in self.recent_dbs:
                    self.recent_dbs.remove(db_path)
                    self.save_recent_dbs()
                    self.update_recent_menu()
                    
    def clear_recent_dbs(self):
        """Clear the list of recent databases."""
        self.recent_dbs = []
        self.save_recent_dbs()
        self.update_recent_menu()

    def _get_new_db_connection(self):
        """Creates a new DuckDB connection (for worker threads)."""
        if not self.db_path:
            print("Error: Cannot create connection, db_path is not set.")
            return None
        try:
            # Ensure extensions needed for import are loaded if necessary
            # config = {'allow_unsigned_extensions': 'true'} # Example if needed
            return duckdb.connect(database=self.db_path, read_only=False)
        except Exception as e:
            print(f"Error creating new DB connection in worker: {e}")
            return None

    def connect_db(self, db_path):
        """Connects to the specified DuckDB database (main thread)."""
        try:
            if self.db_conn:
                self.db_conn.close()

            # Main thread connection
            self.db_conn = duckdb.connect(database=db_path, read_only=False)
            self.db_path = db_path
            self.db_status_label.setText(f"Connected to: {self.db_path}")
            self.load_tables()
            self.results_table.setRowCount(0) # Clear results
            self.results_table.setColumnCount(0)
            self.query_editor.clear()
            
            # Add to recent databases list
            self.add_to_recent_dbs(db_path)
        except Exception as e:
            QMessageBox.critical(self, "Database Error", f"Failed to connect to database:\n{e}")
            self.db_conn = None
            self.db_path = None
            self.db_status_label.setText("Connection failed.")
            self.table_list_widget.clear()

    def create_db(self):
        """Opens a dialog to create and connect to a new DuckDB file."""
        options = QFileDialog.Option.DontUseNativeDialog
        filePath, _ = QFileDialog.getSaveFileName(self, "Create New DuckDB Database", "",
                                                  "DuckDB Files (*.duckdb);;All Files (*)", options=options)
        if filePath:
            # Ensure the file has the .duckdb extension
            if not filePath.endswith('.duckdb'):
                filePath += '.duckdb'
            self.connect_db(filePath)
            # Note: connect_db already adds to recent list

    def open_db(self):
        """Opens a dialog to select and connect to an existing DuckDB file."""
        options = QFileDialog.Option.DontUseNativeDialog
        filePath, _ = QFileDialog.getOpenFileName(self, "Open DuckDB Database", "",
                                                "DuckDB Files (*.duckdb);;All Files (*)", options=options)
        if filePath:
            self.connect_db(filePath)
            # Note: connect_db already adds to recent list
            
    def close_db(self):
        """Closes the current database connection."""
        if self.db_conn:
            try:
                self.db_conn.close()
                self.db_conn = None
                self.db_path = None
                self.db_status_label.setText("No database connected")
                self.table_list_widget.clear()
                self.results_table.setRowCount(0)
                self.results_table.setColumnCount(0)
                self.query_editor.clear()
                QMessageBox.information(self, "Success", "Database connection closed successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to close database connection:\n{e}")
        else:
            QMessageBox.information(self, "Info", "No database connection is currently open.")

    def load_tables(self):
        """Loads the list of tables from the connected database into the list widget."""
        self.table_list_widget.clear()
        if not self.db_conn:
            return
        try:
            tables = self.db_conn.execute("SHOW TABLES").fetchall()
            for table in tables:
                self.table_list_widget.addItem(table[0])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not fetch tables:\n{e}")

    def display_table_schema(self, current, previous):
        """When a table is selected, populate the query editor with a basic SELECT query."""
        if current:
            table_name = current.text()
            # Basic quoting for potential spaces or special chars, might need adjustment for complex names
            quoted_table_name = f'"{table_name}"'
            query = f"SELECT * FROM {quoted_table_name} LIMIT 100;"
            self.query_editor.setPlainText(query)
        else:
            # Optionally clear the editor if no table is selected
            # self.query_editor.clear()
            pass # Keep existing query if user clicks away

    def execute_query(self):
        """Starts the background thread to execute the SQL query."""
        # Check if there's a selection in the editor
        cursor = self.query_editor.textCursor()
        selected_text = cursor.selectedText()
        
        # Use the selected text if available, otherwise use the entire content
        self.is_partial_query = False
        if selected_text:
            query = selected_text.strip()
            self.is_partial_query = True
        else:
            query = self.query_editor.toPlainText().strip()

        if not self.db_conn:
            QMessageBox.warning(self, "Warning", "No database connected.")
            return

        if not query:
            QMessageBox.warning(self, "Warning", "Query cannot be empty.")
            return
            
        # Store state for pagination
        self.current_query = query
        self.current_page = 0
        self.rows_per_page = 1000  # Default page size
        self.total_rows = 0
        
        # --- Setup Progress Dialog ---
        progress_title = "Executing selected query..." if self.is_partial_query else "Executing query..."
        self.query_progress = QProgressDialog(progress_title, "Cancel", 0, 100, self)
        self.query_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.query_progress.setMinimumDuration(300)  # Show after 300ms delay
        self.query_progress.setWindowTitle("Executing Query")
        self.query_progress.setValue(0)
        
        # --- Show Loading Indicator (e.g., change button text/disable) --- 
        # Find the button - assumes it's the last widget in query_layout
        # More robust: give the button an object name `self.run_query_button = ...` in init_ui
        run_button = self.query_editor.parent().layout().itemAt(self.query_editor.parent().layout().count() - 1).widget()
        if isinstance(run_button, QPushButton):
            self.original_button_text = run_button.text()
            run_button.setText("Running...")
            run_button.setEnabled(False)
        else:
            self.original_button_text = None
            run_button = None # Button not found correctly

        # Clear previous results immediately for visual feedback
        self.results_table.setRowCount(0)
        self.results_table.setColumnCount(0)

        # --- Setup Thread and Worker ---
        self.query_thread = QThread(self)
        self.query_worker = QueryWorker(
            self._get_new_db_connection, 
            query, 
            limit=self.rows_per_page, 
            offset=self.current_page * self.rows_per_page
        )
        self.query_worker.moveToThread(self.query_thread)

        # --- Connect Signals/Slots ---
        self.query_thread.started.connect(self.query_worker.run)
        self.query_worker.finished.connect(self.query_thread.quit)
        self.query_worker.finished.connect(self.query_worker.deleteLater)
        self.query_thread.finished.connect(self.query_thread.deleteLater)

        self.query_worker.error.connect(self._on_query_error)
        self.query_worker.success.connect(self._on_query_success)
        self.query_worker.progress.connect(self._on_query_progress)
        self.query_worker.finished.connect(self._on_query_finished)
        self.query_progress.canceled.connect(self.query_worker.cancel)

        # --- Start Thread ---
        self.query_thread.start()

    # --- Query UI Update Slots ---
    def _on_query_progress(self, current, total):
        """Updates the progress dialog during query execution."""
        if hasattr(self, 'query_progress') and self.query_progress:
            if total > 0:
                percentage = min(int((current / total) * 100), 100)
                self.query_progress.setValue(percentage)
                self.query_progress.setLabelText(f"Fetched {current} of {total} rows...")
            else:
                self.query_progress.setValue(0)
                self.query_progress.setLabelText(f"Fetched {current} rows...")

    def _on_query_success(self, data, headers, total_count):
        """Handles successful query execution in the UI thread."""
        print(f"Query Success (UI Thread): Received {len(data)} rows.")
        
        # Store total count for pagination
        self.total_rows = total_count
        
        if headers: # SELECT query with results
            # Update table with the returned data
            self._populate_results_table(data, headers)
            
            # Show pagination controls if we have a large result set
            if total_count > self.rows_per_page:
                pagination_msg = f"Showing page {self.current_page + 1} of {(total_count + self.rows_per_page - 1) // self.rows_per_page} " + \
                                 f"(rows {self.current_page * self.rows_per_page + 1}-{min((self.current_page + 1) * self.rows_per_page, total_count)} of {total_count})"
                success_type = "Selected query" if hasattr(self, 'is_partial_query') and self.is_partial_query else "Query"
                QMessageBox.information(self, "Success", f"{success_type} executed successfully.\n{pagination_msg}")
                
                # Add pagination UI if not already present
                self._ensure_pagination_controls()
            else:
                success_type = "Selected query" if hasattr(self, 'is_partial_query') and self.is_partial_query else "Query"
                QMessageBox.information(self, "Success", f"{success_type} executed successfully.\n{len(data)} rows returned.")
        else: # Non-SELECT query successful
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            success_type = "Selected query" if hasattr(self, 'is_partial_query') and self.is_partial_query else "Query"
            QMessageBox.information(self, "Success", f"{success_type} executed successfully (no results returned).")
            # Reload tables in case the schema changed (e.g., CREATE TABLE, DROP TABLE)
            self.load_tables()
    
    def _populate_results_table(self, data, headers):
        """Efficiently populate the results table with data."""
        # Temporarily turn off sorting to improve performance
        self.results_table.setSortingEnabled(False)
        
        # Clear and set up the table
        self.results_table.setRowCount(0)
        self.results_table.setColumnCount(len(headers))
        self.results_table.setHorizontalHeaderLabels(headers)

        # Pre-allocate rows
        self.results_table.setRowCount(len(data))
        
        # Use batch processing to reduce UI updates
        # Block signals while populating to avoid individual cell change events
        self.results_table.blockSignals(True)
        
        for row_idx, row_data in enumerate(data):
            for col_idx, cell_data in enumerate(row_data):
                item = QTableWidgetItem(str(cell_data) if cell_data is not None else "NULL")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.results_table.setItem(row_idx, col_idx, item)
            
        # Restore signals when done
        self.results_table.blockSignals(False)
        
        # Re-enable sorting
        self.results_table.setSortingEnabled(True)
        
        # Adjust column widths - but limit to reasonable size
        self.results_table.horizontalHeader().setMinimumSectionSize(50)
        self.results_table.horizontalHeader().setDefaultSectionSize(150)
        self.results_table.resizeColumnsToContents()
        
        # Set a max width for columns to avoid excessively wide columns
        for col in range(self.results_table.columnCount()):
            current_width = self.results_table.columnWidth(col)
            if current_width > 300:
                self.results_table.setColumnWidth(col, 300)

    def _ensure_pagination_controls(self):
        """Create or update pagination controls if needed."""
        # Check if we already have a layout with buttons below the results table
        if not hasattr(self, 'pagination_widget'):
            # Create pagination widgets
            self.pagination_widget = QWidget()
            pagination_layout = QHBoxLayout(self.pagination_widget)
            
            self.page_info_label = QLabel()
            pagination_layout.addWidget(self.page_info_label)
            
            pagination_layout.addStretch()
            
            self.prev_page_button = QPushButton("Previous Page")
            self.prev_page_button.clicked.connect(self._load_prev_page)
            pagination_layout.addWidget(self.prev_page_button)
            
            self.next_page_button = QPushButton("Next Page")
            self.next_page_button.clicked.connect(self._load_next_page)
            pagination_layout.addWidget(self.next_page_button)
            
            # Add to the layout that contains the results table
            results_parent_layout = self.results_table.parent().layout()
            results_parent_layout.addWidget(self.pagination_widget)
        
        # Update pagination status
        total_pages = (self.total_rows + self.rows_per_page - 1) // self.rows_per_page
        start_row = self.current_page * self.rows_per_page + 1
        end_row = min((self.current_page + 1) * self.rows_per_page, self.total_rows)
        
        self.page_info_label.setText(f"Page {self.current_page + 1} of {total_pages} (rows {start_row}-{end_row} of {self.total_rows})")
        
        # Enable/disable buttons based on current page
        self.prev_page_button.setEnabled(self.current_page > 0)
        self.next_page_button.setEnabled(self.current_page < total_pages - 1)
        
        # Show the pagination controls
        self.pagination_widget.setVisible(True)
    
    def _load_prev_page(self):
        """Load the previous page of results."""
        if self.current_page > 0:
            self.current_page -= 1
            self._load_current_page()
    
    def _load_next_page(self):
        """Load the next page of results."""
        total_pages = (self.total_rows + self.rows_per_page - 1) // self.rows_per_page
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self._load_current_page()
    
    def _load_current_page(self):
        """Load the current page of results."""
        if not hasattr(self, 'current_query') or not self.current_query:
            return
            
        # Show a progress dialog
        self.query_progress = QProgressDialog("Loading page...", "Cancel", 0, 100, self)
        self.query_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.query_progress.setMinimumDuration(300)
        self.query_progress.setWindowTitle("Loading Results")
        self.query_progress.setValue(0)
        
        # Setup thread and worker
        self.query_thread = QThread(self)
        self.query_worker = QueryWorker(
            self._get_new_db_connection, 
            self.current_query, 
            limit=self.rows_per_page, 
            offset=self.current_page * self.rows_per_page
        )
        self.query_worker.moveToThread(self.query_thread)
        
        # Connect signals
        self.query_thread.started.connect(self.query_worker.run)
        self.query_worker.finished.connect(self.query_thread.quit)
        self.query_worker.finished.connect(self.query_worker.deleteLater)
        self.query_thread.finished.connect(self.query_thread.deleteLater)
        
        self.query_worker.error.connect(self._on_query_error)
        self.query_worker.success.connect(self._on_page_loaded)
        self.query_worker.progress.connect(self._on_query_progress)
        self.query_worker.finished.connect(self._on_query_finished)
        self.query_progress.canceled.connect(self.query_worker.cancel)
        
        # Start thread
        self.query_thread.start()
    
    def _on_page_loaded(self, data, headers, total_count):
        """Handle loading a new page of results."""
        # Update the results table with the new page data
        self._populate_results_table(data, headers)
        
        # Update pagination controls
        self._ensure_pagination_controls()

    def _on_query_error(self, error_message):
        """Handles query errors in the UI thread."""
        print("Query Error (UI Thread):", error_message)
        self.results_table.setRowCount(0)
        self.results_table.setColumnCount(0)
        QMessageBox.critical(self, "Query Error", error_message)

    def _on_query_finished(self):
        """Cleans up after query execution in the UI thread."""
        print("Query thread finished.")
        
        # Close the progress dialog if it exists
        if hasattr(self, 'query_progress') and self.query_progress:
            self.query_progress.close()
            
        # Restore button state
        # Find the button again (or use self.run_query_button if set up)
        run_button = self.query_editor.parent().layout().itemAt(self.query_editor.parent().layout().count() - 1).widget()
        if isinstance(run_button, QPushButton) and self.original_button_text is not None:
            run_button.setText(self.original_button_text)
            run_button.setEnabled(True)
            
        # Clean up references
        self.query_thread = None
        self.query_worker = None

    # --- Import Methods ---

    def _get_import_options(self, file_path):
        """Gets import mode and table name from the user."""
        base_name = os.path.basename(file_path)
        modes = ["Create New Table", "Replace Existing Table", "Append to Existing Table"]

        # 1. Get Import Mode
        mode, ok = QInputDialog.getItem(self, "Import Mode",
                                       f"Select import mode for '{base_name}':",
                                       modes, 0, False)
        if not ok or not mode:
            return None, None # User cancelled

        # 2. Get Table Name (depends on mode)
        table_name = None
        if mode == "Create New Table":
            suggested_table_name = os.path.splitext(base_name)[0]
            suggested_table_name = "".join(c if c.isalnum() else '_' for c in suggested_table_name)

            dialog = QInputDialog(self)
            dialog.setStyleSheet(DARK_STYLESHEET)
            dialog.setWindowTitle("New Table Name")
            dialog.setLabelText("Enter name for the new table:")
            dialog.setTextValue(suggested_table_name)
            dialog.setOkButtonText("Ok")

            if dialog.exec() == QInputDialog.DialogCode.Accepted:
                table_name = dialog.textValue().strip()
                if not table_name:
                    QMessageBox.warning(self, "Input Error", "Table name cannot be empty.")
                    return None, None
                # Sanitize
                table_name = "".join(c if c.isalnum() else '_' for c in table_name)
                # Check if it already exists when creating new
                if self._table_exists(table_name):
                     QMessageBox.warning(self, "Input Error", f"Table '{table_name}' already exists. Cannot create new.")
                     return None, None
            else:
                 return None, None # User cancelled name input

        elif mode in ["Replace Existing Table", "Append to Existing Table"]:
            try:
                existing_tables = [t[0] for t in self.db_conn.execute("SHOW TABLES").fetchall()]
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not fetch existing tables: {e}")
                return None, None

            if not existing_tables:
                QMessageBox.warning(self, "Error", f"No existing tables found to {mode.lower().split(' ')[0]}. Choose 'Create New Table'.")
                return None, None

            table_name, ok = QInputDialog.getItem(self, f"Select Table to {mode.split(' ')[0]}",
                                                "Choose target table:", existing_tables, 0, False)
            if not ok or not table_name:
                return None, None # User cancelled

        return table_name, mode

    def _get_table_schema(self, table_name):
        """Gets column names and types for a given table. Returns dict {name: type}."""
        schema = {}
        try:
            # PRAGMA returns: cid, name, type, notnull, dflt_value, pk
            pragma_info = self.db_conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
            for col in pragma_info:
                schema[col[1]] = col[2] # {name: type}
            return schema
        except Exception as e:
            print(f"Error getting schema for table {table_name}: {e}")
            QMessageBox.warning(self, "Schema Error", f"Could not get schema for table '{table_name}'.\n{e}")
            return None

    def _get_source_schema_from_sql(self, read_function_sql):
        """Gets column names and types from a SELECT statement (e.g., read_csv). Returns dict {name: type}."""
        schema = {}
        try:
            # Describe the result of the read function
            # Important: Use LIMIT 0 for schema check without reading data if possible, but DESCRIBE works
            describe_query = f"DESCRIBE SELECT * FROM {read_function_sql};"
            # DESCRIBE returns: column_name, column_type, null, key, default, extra
            desc_result = self.db_conn.execute(describe_query).fetchall()
            for col in desc_result:
                schema[col[0]] = col[1] # {name: type}
            return schema
        except Exception as e:
            print(f"Error describing source SQL ({read_function_sql}): {e}")
            QMessageBox.warning(self, "Schema Error", f"Could not determine schema from source file.\n{e}")
            return None

    def _get_source_schema_from_df(self, df):
         """Gets column names and mapped DuckDB types from a pandas DataFrame. Returns dict {lower_name: (orig_name, type)}."""
         schema = {}
         # Simple Pandas dtype to DuckDB type mapping (can be expanded)
         type_mapping = {
             'int64': 'BIGINT',
             'int32': 'INTEGER',
             'float64': 'DOUBLE',
             'float32': 'FLOAT',
             'bool': 'BOOLEAN',
             'datetime64[ns]': 'TIMESTAMP',
             'timedelta64[ns]': 'INTERVAL',
             'object': 'VARCHAR' # Default for strings or mixed types
             # Add other types as needed (e.g., category, specific date/time types)
         }
         for col_name, dtype in df.dtypes.items():
             sql_type = type_mapping.get(str(dtype), 'VARCHAR') # Default to VARCHAR
             schema[str(col_name).lower()] = (str(col_name), sql_type) # Store lower_name -> (orig_name, type)
         return schema

    def _table_exists(self, table_name):
        """Checks if a table exists in the current database."""
        try:
            # Use PRAGMA for table info, more robust than simple SELECT
            result = self.db_conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
            return len(result) > 0
        except Exception as e:
            # Could be an error if the DB connection is bad, or other SQL errors
            print(f"Error checking if table exists: {e}") # Log error
            # Assume it doesn't exist or let the subsequent query fail
            return False

    def _execute_import_core(self, db_conn_worker, table_name, mode, file_path, read_function_sql, worker_ref):
        """Core import logic (runs in worker thread). Takes a DB connection."""
        file_name = os.path.basename(file_path)
        escaped_file_name = file_name.replace("'", "''")

        # We need temporary schema functions that use the worker connection
        def _worker_table_exists(name):
            try: return len(db_conn_worker.execute(f"PRAGMA table_info('{name}')").fetchall()) > 0
            except: return False # Simplistic error handling

        def _worker_get_table_schema(name):
            # Returns: {lower_case_name: (original_name, type, is_not_null, default_value, is_pk)}
            schema = {}
            try:
                # PRAGMA returns: cid, name, type, notnull (0/1), dflt_value, pk (0/non-zero)
                pragma_info = db_conn_worker.execute(f"PRAGMA table_info('{name}')").fetchall()
                for col in pragma_info:
                    lower_name = col[1].lower()
                    original_name = col[1]
                    col_type = col[2]
                    is_not_null = bool(col[3])
                    default_value = col[4]
                    is_pk = bool(col[5]) # PK is non-zero if part of the key
                    schema[lower_name] = (original_name, col_type, is_not_null, default_value, is_pk)
                return schema
            except Exception as e: print(f"Worker schema error: {e}"); return None

        def _worker_get_source_schema_sql(read_sql):
            schema = {}
            try:
                desc_result = db_conn_worker.execute(f"DESCRIBE SELECT * FROM {read_sql};").fetchall()
                # Store {lower_case_name: (original_name, type)}
                for col in desc_result:
                    schema[col[0].lower()] = (col[0], col[1])
                return schema
            except Exception as e: print(f"Worker describe error: {e}"); return None

        # --- Start actual logic ---
        table_exists = _worker_table_exists(table_name)
        query = None

        # Check for cancellation early
        if worker_ref.is_cancelled:
             raise InterruptedError("Import cancelled before execution.")

        select_with_source = f"SELECT *, '{escaped_file_name}' AS source_file FROM {read_function_sql}"

        if mode == "Create New Table":
            if table_exists:
                raise ValueError(f"Table '{table_name}' already exists.")
            query = f'CREATE TABLE "{table_name}" AS {select_with_source};'

        elif mode == "Replace Existing Table":
            # Use DROP CASCADE + CREATE AS to ensure no old constraints (like PKs) remain
            # This forcefully removes the table and anything depending on it.
            drop_query = f'DROP TABLE IF EXISTS "{table_name}" CASCADE;'
            print(f"Worker executing: {drop_query}")
            db_conn_worker.execute(drop_query)
            
            # Create the new table from source, without constraints
            query = f'CREATE TABLE "{table_name}" AS {select_with_source};'
            # No DELETE, ALTER, or extra schema checks needed here for Replace mode

        elif mode == "Append to Existing Table":
            if not table_exists:
                raise ValueError(f"Table '{table_name}' does not exist.")

            target_schema = _worker_get_table_schema(table_name)
            if target_schema is None:
                raise ConnectionError(f"Could not get target schema for '{table_name}'.")

            source_schema = _worker_get_source_schema_sql(read_function_sql)
            if source_schema is None:
                 raise ConnectionError(f"Could not determine source schema from '{file_name}'.")

            # --- Check for missing required columns --- 
            missing_required = []
            for lower_target_col, (orig_target_col, _, is_not_null, default_val, is_pk) in target_schema.items():
                if is_not_null and default_val is None and lower_target_col not in source_schema:
                     if lower_target_col != 'source_file':
                        missing_required.append(orig_target_col)
            
            # Raise error if ANY required column is missing
            if missing_required:
                raise ValueError(f"Target table '{table_name}' requires column(s) not found in source file '{file_name}': {', '.join(missing_required)}")
            # --- End check --- 

            # Add source_file to the expected source schema (using lowercase key)
            source_schema['source_file'] = ('source_file', 'VARCHAR')

            alter_statements = []
            # Compare using lowercase keys
            for lower_col_name, (orig_col_name, col_type) in source_schema.items():
                if lower_col_name not in target_schema:
                     # Use original case name for ALTER TABLE
                    alter_statements.append(f'ALTER TABLE "{table_name}" ADD COLUMN "{orig_col_name}" {col_type};')

            # Check for cancellation before potentially long ALTER/INSERT
            if worker_ref.is_cancelled: raise InterruptedError("Import cancelled before schema change/insert.")

            if alter_statements:
                print(f"Worker applying schema changes: {alter_statements}")
                for alter_query in alter_statements:
                    db_conn_worker.execute(alter_query)

            # Get the actual source columns for explicit column mapping, using original case
            source_cols_result = db_conn_worker.execute(f"SELECT * FROM {read_function_sql} LIMIT 0").description
            if source_cols_result:
                orig_source_cols = [desc[0] for desc in source_cols_result] + ['source_file']
                cols_str = ', '.join([f'"{col}"' for col in orig_source_cols])
                query = f'INSERT INTO "{table_name}" ({cols_str}) ({select_with_source});'
            else:
                # Fallback if can't get columns
                query = f'INSERT INTO "{table_name}" ({select_with_source});'

        if query:
            print(f"Worker executing: {query}")
            # DuckDB execute can still take time here, cancellation check isn't perfect
            db_conn_worker.execute(query)
            # Return success details
            return f"Data from '{file_name}' imported successfully into table '{table_name}' (Mode: {mode})."
        else:
            raise ValueError("Invalid import mode determined in worker.")

    def _execute_excel_import_core(self, db_conn_worker, table_name, mode, file_path, sheet_name, worker_ref):
        """Core import logic for Excel (runs in worker thread)."""
        try:
            import pandas as pd # Import within the function if not globally available
        except ImportError:
            raise ImportError("Pandas/openpyxl not found in worker thread environment.")

        file_name = os.path.basename(file_path)
        escaped_file_name = file_name.replace("'", "''")
        temp_view_name = f"__{table_name}_excel_worker_view"

        # Helper schema functions using worker connection
        def _worker_table_exists(name):
            try: return len(db_conn_worker.execute(f"PRAGMA table_info('{name}')").fetchall()) > 0
            except: return False

        def _worker_get_table_schema(name):
            # Returns: {lower_case_name: (original_name, type, is_not_null, default_value, is_pk)}
            schema = {}
            try:
                # PRAGMA returns: cid, name, type, notnull (0/1), dflt_value, pk (0/non-zero)
                pragma_info = db_conn_worker.execute(f"PRAGMA table_info('{name}')").fetchall()
                for col in pragma_info:
                    lower_name = col[1].lower()
                    original_name = col[1]
                    col_type = col[2]
                    is_not_null = bool(col[3])
                    default_value = col[4]
                    is_pk = bool(col[5]) # PK is non-zero if part of the key
                    schema[lower_name] = (original_name, col_type, is_not_null, default_value, is_pk)
                return schema
            except Exception as e: print(f"Worker schema error: {e}"); return None

        try:
            # --- Pandas Read (potentially slow) ---
            print(f"Worker reading Excel: {file_path} Sheet: {sheet_name}")
            # NOTE: Cancellation during pd.read_excel is not directly supported here.
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            df['source_file'] = escaped_file_name # Add source file column

            if worker_ref.is_cancelled:
                raise InterruptedError("Import cancelled after Excel read.")

            # --- DuckDB Operations --- 
            db_conn_worker.register(temp_view_name, df)
            select_from_view = f"SELECT * FROM {temp_view_name}"

            table_exists = _worker_table_exists(table_name)
            query = None

            if mode == "Create New Table":
                if table_exists:
                    raise ValueError(f"Table '{table_name}' already exists.")
                query = f'CREATE TABLE "{table_name}" AS {select_from_view};'

            elif mode == "Replace Existing Table":
                # Use DROP CASCADE + CREATE AS to ensure no old constraints (like PKs) remain
                # This forcefully removes the table and anything depending on it.
                drop_query = f'DROP TABLE IF EXISTS "{table_name}" CASCADE;'
                print(f"Worker executing: {drop_query}")
                db_conn_worker.execute(drop_query)
                
                # Create the new table from source, without constraints
                query = f'CREATE TABLE "{table_name}" AS {select_from_view};'
                # No DELETE, ALTER, or extra schema checks needed here for Replace mode

            elif mode == "Append to Existing Table":
                if not table_exists:
                    raise ValueError(f"Table '{table_name}' does not exist.")

                target_schema = _worker_get_table_schema(table_name)
                if target_schema is None: raise ConnectionError(f"Could not get target schema for '{table_name}'.")

                source_schema = self._get_source_schema_from_df(df) # Schema from DF
                if source_schema is None: raise ValueError("Could not get source schema from DataFrame.")
                # source_file is already in DF schema, ensure lowercase key maps to tuple
                if 'source_file' in source_schema:
                     orig_name, type = source_schema['source_file']
                     source_schema['source_file'] = (orig_name, type)
                else: # Should have been added earlier
                    source_schema['source_file'] = ('source_file', 'VARCHAR')

                # --- Check for missing required columns --- 
                missing_required = []
                for lower_target_col, (orig_target_col, _, is_not_null, default_val, is_pk) in target_schema.items():
                    if is_not_null and default_val is None and lower_target_col not in source_schema:
                         if lower_target_col != 'source_file':
                            missing_required.append(orig_target_col)
                
                # Raise error if ANY required column is missing
                if missing_required:
                    raise ValueError(f"Target table '{table_name}' requires column(s) not found in source file '{file_name}': {', '.join(missing_required)}")
                # --- End check --- 

                alter_statements = []
                # Compare using lowercase keys
                for lower_col_name, (orig_col_name, col_type) in source_schema.items():
                    if lower_col_name not in target_schema:
                        # Use original case name for ALTER TABLE
                        alter_statements.append(f'ALTER TABLE "{table_name}" ADD COLUMN "{orig_col_name}" {col_type};')

                # Check for cancellation before potentially long ALTER/INSERT
                if worker_ref.is_cancelled: raise InterruptedError("Import cancelled before schema change/insert.")

                if alter_statements:
                    print(f"Worker applying schema changes: {alter_statements}")
                    for alter_query in alter_statements:
                        db_conn_worker.execute(alter_query)

                # Modified INSERT query to explicitly list columns (original case)
                df_columns = [col_info[0] for col_info in source_schema.values()] # Get original names
                source_cols = ', '.join([f'"{col}"' for col in df_columns])
                query = f'INSERT INTO "{table_name}" ({source_cols}) {select_from_view};'

            if query:
                print(f"Worker executing: {query}")
                db_conn_worker.execute(query)
                # Return success details
                return f"Data from '{file_name}' (Sheet: {sheet_name}) imported successfully into table '{table_name}' (Mode: {mode})."
            else:
                raise ValueError("Invalid import mode determined in worker.")

        finally:
            # Ensure view is unregistered
            try: db_conn_worker.unregister(temp_view_name)
            except: pass # Ignore errors during cleanup

    # --- Updated UI Trigger --- 
    def _start_import_thread(self, import_func_core, file_path, *extra_args):
        """Generic function to setup and start the import worker thread."""
        table_name, mode = self._get_import_options(file_path)
        if not table_name or not mode:
            return # User cancelled options

        # Special handling for Excel sheet name (must be done in UI thread)
        sheet_name = None
        if import_func_core == self._execute_excel_import_core:
            try:
                import pandas as pd
                xls = pd.ExcelFile(file_path)
                sheet_names = xls.sheet_names
                if not sheet_names:
                    QMessageBox.warning(self, "Excel Read Error", "No sheets found in the Excel file.")
                    return
                
                # Always show sheet selection dialog, even for single sheet
                dialog = QInputDialog(self)
                dialog.setStyleSheet(DARK_STYLESHEET)
                sheet_title = "Select Excel Sheet"
                if len(sheet_names) == 1:
                    sheet_title += f" (only 1 sheet available)"
                else:
                    sheet_title += f" ({len(sheet_names)} sheets available)"
                    
                sheet_name_selected, ok = dialog.getItem(self, sheet_title, 
                                                       "Which sheet do you want to import?:", 
                                                       sheet_names, 0, False)
                if not ok or not sheet_name_selected:
                    return # User cancelled sheet selection
                sheet_name = sheet_name_selected
                # Add sheet_name to the arguments passed to the worker
                extra_args = (sheet_name,) + extra_args
            except ImportError:
                 QMessageBox.critical(self, "Import Error", "Pandas/openpyxl needed to select Excel sheet.")
                 return
            except Exception as e:
                 QMessageBox.critical(self, "Excel Read Error", f"Could not read sheets from Excel file for selection:\n{e}")
                 return

        # --- Setup Progress Dialog --- 
        progress_text = f"Importing {os.path.basename(file_path)}..."
        if sheet_name:
            progress_text = f"Importing {os.path.basename(file_path)} (Sheet: {sheet_name})..."
        self.progress = QProgressDialog(progress_text, "Cancel", 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.setWindowTitle("Importing Data")
        self.progress.setValue(0)
        self.progress.show()

        # --- Setup Thread and Worker ---
        self.import_thread = QThread(self)
        all_args = (table_name, mode, file_path) + extra_args
        self.import_worker = ImportWorker(self._get_new_db_connection, import_func_core, *all_args)
        self.import_worker.moveToThread(self.import_thread)

        # --- Connect Signals/Slots ---
        self.import_thread.started.connect(self.import_worker.run)
        self.import_worker.finished.connect(self.import_thread.quit)
        self.import_worker.finished.connect(self.import_worker.deleteLater)
        self.import_thread.finished.connect(self.import_thread.deleteLater)

        self.import_worker.error.connect(self._on_import_error)
        self.import_worker.success.connect(self._on_import_success)
        self.import_worker.finished.connect(self._on_import_finished)
        self.progress.canceled.connect(self.import_worker.cancel)

        # --- Start Thread ---
        self.import_thread.start()

    # --- UI Update Slots ---
    def _on_import_success(self, message):
        print("Import Success (UI Thread):", message)
        if self.progress: self.progress.close()
        QMessageBox.information(self, "Success", message)
        self.load_tables() # Refresh table list in UI thread

    def _on_import_error(self, error_message):
        print("Import Error (UI Thread):", error_message)
        if self.progress: self.progress.close()
        QMessageBox.critical(self, "Import Error", error_message)
        # Optionally reload tables even on error?
        # self.load_tables()

    def _on_import_finished(self):
        print("Import thread finished.")
        # Ensure progress dialog is closed
        if hasattr(self, 'progress') and self.progress and self.progress.isVisible():
            self.progress.close()
        # Clean up references (optional, depends on PyQt version/gc)
        self.import_thread = None
        self.import_worker = None

    def import_csv(self):
        """Imports data from a CSV file into a table with options (uses thread)."""
        if not self.db_conn:
            QMessageBox.warning(self, "Warning", "Please connect to a database first.")
            return

        options = QFileDialog.Option.DontUseNativeDialog
        filePath, _ = QFileDialog.getOpenFileName(self, "Import CSV File", "",
                                                "CSV Files (*.csv *.tsv);;All Files (*)", options=options)
        if filePath:
            # Ask for delimiter
            delimiters = [("Auto-detect (default)", None), ("Comma (,)", ","), ("Tab (\\t)", "\t"), 
                         ("Semicolon (;)", ";"), ("Pipe (|)", "|"), ("Other", "custom")]
            
            delimiter_items = [f"{name}" for name, _ in delimiters]
            delimiter_choice, ok = QInputDialog.getItem(self, "CSV Delimiter", 
                                                     "Select the delimiter used in the CSV file:", 
                                                     delimiter_items, 0, False)
            if not ok:
                return  # User cancelled
            
            # Get the delimiter value
            selected_delimiter = None
            for i, (name, value) in enumerate(delimiters):
                if delimiter_items[i] == delimiter_choice:
                    selected_delimiter = value
                    break
            
            # If "Other" was selected, ask for the custom delimiter
            if selected_delimiter == "custom":
                custom_delimiter, ok = QInputDialog.getText(self, "Custom Delimiter", 
                                                        "Enter the custom delimiter character:")
                if not ok or not custom_delimiter:
                    return  # User cancelled or entered empty delimiter
                selected_delimiter = custom_delimiter
            
            # Escape backslashes in filepath for SQL
            escaped_filePath = filePath.replace('\\', '\\\\')
            
            # Build the read_csv function with the appropriate delimiter
            if selected_delimiter is None:
                # Auto-detect with read_csv_auto
                read_function = f"read_csv_auto('{escaped_filePath}')"
            else:
                # Use specific delimiter with read_csv
                escaped_delimiter = selected_delimiter.replace("'", "''")
                read_function = f"read_csv('{escaped_filePath}', delim='{escaped_delimiter}', header=true, auto_detect=true)"
            
            # Call the starter function which sets up the thread
            self._start_import_thread(self._execute_import_core, filePath, read_function)

    def import_parquet(self):
        """Imports data from a Parquet file into a table with options (uses thread)."""
        if not self.db_conn:
            QMessageBox.warning(self, "Warning", "Please connect to a database first.")
            return

        options = QFileDialog.Option.DontUseNativeDialog
        filePath, _ = QFileDialog.getOpenFileName(self, "Import Parquet File", "",
                                                "Parquet Files (*.parquet);;All Files (*)", options=options)
        if filePath:
            # Escape backslashes in filepath for SQL
            escaped_filePath = filePath.replace('\\', '\\\\')
            read_function = f"read_parquet('{escaped_filePath}')"
            self._start_import_thread(self._execute_import_core, filePath, read_function)

    def import_excel(self):
        """Imports data from an Excel file (uses thread)."""
        if not self.db_conn:
            QMessageBox.warning(self, "Warning", "Please connect to a database first.")
            return
        # Check for pandas/openpyxl in main thread first for early feedback
        try:
            import pandas as pd
            import openpyxl
        except ImportError:
             QMessageBox.critical(self, "Import Error", "Libraries `pandas` and `openpyxl` are required for Excel import.\nPlease install them (e.g., `pip install pandas openpyxl`).")
             return

        options = QFileDialog.Option.DontUseNativeDialog
        filePath, _ = QFileDialog.getOpenFileName(self, "Import Excel File", "",
                                                "Excel Files (*.xlsx *.xls);;All Files (*)", options=options)
        if filePath:
            # No extra args needed here initially, sheet name is handled in _start_import_thread
            self._start_import_thread(self._execute_excel_import_core, filePath)

    def show_table_context_menu(self, pos):
        """Shows the context menu for the table list."""
        item = self.table_list_widget.itemAt(pos)
        if not item: # Clicked on empty space
            return

        table_name = item.text()

        menu = QMenu()
        delete_action = QAction(f'Delete Table "{table_name}"...', self)
        delete_action.triggered.connect(lambda: self.delete_table(table_name))
        menu.addAction(delete_action)

        # Add other actions later if needed (e.g., Describe, Rename)

        # Show the menu at the cursor position
        menu.exec(self.table_list_widget.mapToGlobal(pos))

    def delete_table(self, table_name):
        """Handles deleting a table after confirmation."""
        if not self.db_conn:
            QMessageBox.warning(self, "Warning", "No database connected.")
            return

        reply = QMessageBox.question(self,
                                     "Confirm Delete",
                                     f"Are you sure you want to permanently delete the table \"{table_name}\"?\n\nThis will also remove any dependent objects (like foreign key references in other tables). This action cannot be undone.",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No) # Default to No

        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Use CASCADE to remove dependencies
                drop_query = f'DROP TABLE "{table_name}" CASCADE;'
                print(f"Executing: {drop_query}")
                self.db_conn.execute(drop_query)
                self.load_tables() # Refresh list
                QMessageBox.information(self, "Success", f"Table '{table_name}' deleted successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete table '{table_name}':\n{e}")

    # --- Export Methods ---
    
    def _get_results_as_dataframe(self):
        """Convert current query results to a pandas DataFrame."""
        if not hasattr(self, 'results_table') or self.results_table.rowCount() == 0 or self.results_table.columnCount() == 0:
            QMessageBox.warning(self, "Warning", "No results to export. Run a query first.")
            return None
            
        try:
            import pandas as pd
            
            # For very large tables, this could be slow
            # First, check if we have stored the full result set for pagination
            if hasattr(self, 'current_query') and self.current_query and hasattr(self, 'total_rows') and self.total_rows > self.rows_per_page:
                # For large result sets, directly query the database to avoid UI freezing
                try:
                    # Show progress dialog for direct database query
                    progress = QProgressDialog("Preparing data for export...", "Cancel", 0, 100, self)
                    progress.setWindowModality(Qt.WindowModality.WindowModal)
                    progress.setMinimumDuration(300)
                    progress.setValue(10)
                    
                    # Create a temporary connection and get all data directly
                    temp_conn = self._get_new_db_connection()
                    if temp_conn:
                        progress.setValue(20)
                        # Process in batches with pd.read_sql_query to avoid memory issues
                        # First, make sure we don't have LIMIT or OFFSET in the original query
                        query = self.current_query
                        # Remove any existing LIMIT clause
                        if re.search(r'\bLIMIT\b\s+\d+', query, re.IGNORECASE):
                            query = re.sub(r'\bLIMIT\b\s+\d+', '', query, flags=re.IGNORECASE)
                        # Remove any existing OFFSET clause
                        if re.search(r'\bOFFSET\b\s+\d+', query, re.IGNORECASE):
                            query = re.sub(r'\bOFFSET\b\s+\d+', '', query, flags=re.IGNORECASE)
                            
                        progress.setValue(30)
                        df = pd.read_sql_query(query, temp_conn)
                        progress.setValue(90)
                        temp_conn.close()
                        progress.setValue(100)
                        progress.close()
                        return df
                except Exception as e:
                    print(f"Could not export full result set directly: {e}")
                    # Fall back to UI table data
                    if progress:
                        progress.close()
            
            # Get column headers
            headers = []
            for col in range(self.results_table.columnCount()):
                header_item = self.results_table.horizontalHeaderItem(col)
                headers.append(header_item.text() if header_item else f"Column{col}")
            
            # Get data from table (for when direct database access fails or for smaller datasets)
            data = []
            total_rows = self.results_table.rowCount()
            
            # Show progress for larger tables
            progress = None
            if total_rows > 1000:
                progress = QProgressDialog("Reading table data...", "Cancel", 0, total_rows, self)
                progress.setWindowModality(Qt.WindowModality.WindowModal)
                progress.setMinimumDuration(300)
            
            # Process in batches
            batch_size = 1000
            for start_row in range(0, total_rows, batch_size):
                end_row = min(start_row + batch_size, total_rows)
                
                if progress:
                    progress.setValue(start_row)
                    if progress.wasCanceled():
                        if progress:
                            progress.close()
                        return None
                
                # Process this batch
                for row in range(start_row, end_row):
                    row_data = []
                    for col in range(self.results_table.columnCount()):
                        item = self.results_table.item(row, col)
                        # Handle null values
                        value = item.text() if item else None
                        # Convert "NULL" string to None
                        if value == "NULL":
                            value = None
                        row_data.append(value)
                    data.append(row_data)
            
            if progress:
                progress.close()
                
            # Create DataFrame
            df = pd.DataFrame(data, columns=headers)
            return df
            
        except ImportError:
            QMessageBox.critical(self, "Error", "Pandas library is required for exporting data.\nPlease install it with 'pip install pandas'.")
            return None
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error preparing data for export:\n{e}")
            return None
    
    def export_to_csv(self):
        """Export current query results to a CSV file."""
        df = self._get_results_as_dataframe()
        if df is None:
            return
            
        # Get file path
        options = QFileDialog.Option.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export as CSV", "", "CSV Files (*.csv);;All Files (*)", 
            options=options
        )
        
        if not file_path:
            return  # User cancelled
            
        # Add .csv extension if not present
        if not file_path.lower().endswith('.csv'):
            file_path += '.csv'
            
        # Ask for delimiter
        delimiters = [
            ("Comma (,)", ","), 
            ("Tab (\\t)", "\t"), 
            ("Semicolon (;)", ";"), 
            ("Pipe (|)", "|"),
            ("Other", "custom")
        ]
        
        delimiter_items = [name for name, _ in delimiters]
        delimiter_choice, ok = QInputDialog.getItem(
            self, "CSV Delimiter", 
            "Select the delimiter to use in the CSV file:", 
            delimiter_items, 0, False
        )
        
        if not ok:
            return  # User cancelled
            
        # Get the delimiter value
        selected_delimiter = None
        for name, value in delimiters:
            if name == delimiter_choice:
                selected_delimiter = value
                break
                
        # If "Other" was selected, ask for the custom delimiter
        if selected_delimiter == "custom":
            custom_delimiter, ok = QInputDialog.getText(
                self, "Custom Delimiter", 
                "Enter the custom delimiter character:"
            )
            if not ok or not custom_delimiter:
                return  # User cancelled or entered empty delimiter
            selected_delimiter = custom_delimiter
        
        # --- Setup Progress Dialog ---
        self.export_progress = QProgressDialog("Exporting data to CSV...", "Cancel", 0, 100, self)
        self.export_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.export_progress.setMinimumDuration(300)
        self.export_progress.setWindowTitle("Exporting Data")
        self.export_progress.setValue(0)
        
        # --- Setup Thread and Worker ---
        self.export_thread = QThread(self)
        self.export_worker = ExportWorker(
            'csv', 
            df, 
            file_path, 
            delimiter=selected_delimiter
        )
        self.export_worker.moveToThread(self.export_thread)
        
        # --- Connect Signals/Slots ---
        self.export_thread.started.connect(self.export_worker.run)
        self.export_worker.finished.connect(self.export_thread.quit)
        self.export_worker.finished.connect(self.export_worker.deleteLater)
        self.export_thread.finished.connect(self.export_thread.deleteLater)
        
        self.export_worker.error.connect(self._on_export_error)
        self.export_worker.success.connect(self._on_export_success)
        self.export_worker.progress.connect(self._on_export_progress)
        self.export_worker.finished.connect(self._on_export_finished)
        self.export_progress.canceled.connect(self.export_worker.cancel)
        
        # --- Start Thread ---
        self.export_thread.start()
    
    def export_to_excel(self):
        """Export current query results to an Excel file."""
        df = self._get_results_as_dataframe()
        if df is None:
            return
            
        # Check for required libraries
        try:
            import openpyxl
        except ImportError:
            QMessageBox.critical(self, "Error", 
                "The openpyxl library is required for Excel export.\n"
                "Please install it with 'pip install openpyxl'."
            )
            return
            
        # Get file path
        options = QFileDialog.Option.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export as Excel", "", "Excel Files (*.xlsx);;All Files (*)", 
            options=options
        )
        
        if not file_path:
            return  # User cancelled
            
        # Add .xlsx extension if not present
        if not file_path.lower().endswith('.xlsx'):
            file_path += '.xlsx'
        
        # --- Setup Progress Dialog ---
        self.export_progress = QProgressDialog("Exporting data to Excel...", "Cancel", 0, 100, self)
        self.export_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.export_progress.setMinimumDuration(300)
        self.export_progress.setWindowTitle("Exporting Data")
        self.export_progress.setValue(0)
        
        # --- Setup Thread and Worker ---
        self.export_thread = QThread(self)
        self.export_worker = ExportWorker(
            'excel', 
            df, 
            file_path
        )
        self.export_worker.moveToThread(self.export_thread)
        
        # --- Connect Signals/Slots ---
        self.export_thread.started.connect(self.export_worker.run)
        self.export_worker.finished.connect(self.export_thread.quit)
        self.export_worker.finished.connect(self.export_worker.deleteLater)
        self.export_thread.finished.connect(self.export_thread.deleteLater)
        
        self.export_worker.error.connect(self._on_export_error)
        self.export_worker.success.connect(self._on_export_success)
        self.export_worker.progress.connect(self._on_export_progress)
        self.export_worker.finished.connect(self._on_export_finished)
        self.export_progress.canceled.connect(self.export_worker.cancel)
        
        # --- Start Thread ---
        self.export_thread.start()
    
    def export_to_parquet(self):
        """Export current query results to a Parquet file."""
        df = self._get_results_as_dataframe()
        if df is None:
            return
            
        # Check for required libraries
        try:
            import pyarrow
        except ImportError:
            QMessageBox.critical(self, "Error", 
                "The pyarrow library is required for Parquet export.\n"
                "Please install it with 'pip install pyarrow'."
            )
            return
            
        # Get file path
        options = QFileDialog.Option.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export as Parquet", "", "Parquet Files (*.parquet);;All Files (*)", 
            options=options
        )
        
        if not file_path:
            return  # User cancelled
            
        # Add .parquet extension if not present
        if not file_path.lower().endswith('.parquet'):
            file_path += '.parquet'
        
        # --- Setup Progress Dialog ---
        self.export_progress = QProgressDialog("Exporting data to Parquet...", "Cancel", 0, 100, self)
        self.export_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.export_progress.setMinimumDuration(300)
        self.export_progress.setWindowTitle("Exporting Data")
        self.export_progress.setValue(0)
        
        # --- Setup Thread and Worker ---
        self.export_thread = QThread(self)
        self.export_worker = ExportWorker(
            'parquet', 
            df, 
            file_path
        )
        self.export_worker.moveToThread(self.export_thread)
        
        # --- Connect Signals/Slots ---
        self.export_thread.started.connect(self.export_worker.run)
        self.export_worker.finished.connect(self.export_thread.quit)
        self.export_worker.finished.connect(self.export_worker.deleteLater)
        self.export_thread.finished.connect(self.export_thread.deleteLater)
        
        self.export_worker.error.connect(self._on_export_error)
        self.export_worker.success.connect(self._on_export_success)
        self.export_worker.progress.connect(self._on_export_progress)
        self.export_worker.finished.connect(self._on_export_finished)
        self.export_progress.canceled.connect(self.export_worker.cancel)
        
        # --- Start Thread ---
        self.export_thread.start()
    
    # --- Export UI Update Slots ---
    def _on_export_progress(self, current, total):
        """Updates the progress dialog during export."""
        if hasattr(self, 'export_progress') and self.export_progress:
            if total > 0:
                percentage = min(int((current / total) * 100), 100)
                self.export_progress.setValue(percentage)
                self.export_progress.setLabelText(f"Exported {current} of {total} rows...")
            else:
                self.export_progress.setValue(0)
    
    def _on_export_success(self, message):
        """Handles successful export in the UI thread."""
        QMessageBox.information(self, "Success", message)
    
    def _on_export_error(self, error_message):
        """Handles export errors in the UI thread."""
        QMessageBox.critical(self, "Export Error", error_message)
    
    def _on_export_finished(self):
        """Cleans up after export in the UI thread."""
        # Close the progress dialog if it exists
        if hasattr(self, 'export_progress') and self.export_progress:
            self.export_progress.close()
            
        # Clean up references
        self.export_thread = None
        self.export_worker = None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = DuckDBApp()
    main_win.show()
    sys.exit(app.exec()) 
