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
        self.start_time = None

    def export_csv_in_chunks(self):
        total_rows = len(self.df)
        with open(self.file_path, 'w', newline='') as f:
            # Write header
            self.df.head(0).to_csv(f, index=False)
            
            for start_idx in range(0, total_rows, self.CHUNK_SIZE):
                if self.is_cancelled:
                    break
                end_idx = min(start_idx + self.CHUNK_SIZE, total_rows)
                chunk = self.df.iloc[start_idx:end_idx]
                chunk.to_csv(f, header=False, index=False, mode='a')
                progress = (end_idx / total_rows) * 100
                elapsed = (pd.Timestamp.now() - self.start_time).total_seconds()
                self.progress.emit(f"Exporting CSV: {progress:.1f}% complete... (Elapsed: {elapsed:.1f}s)")

    def export_excel_in_chunks(self):
        try:
            total_rows = len(self.df)
            writer = pd.ExcelWriter(self.file_path, engine='openpyxl')
            
            for start_idx in range(0, total_rows, self.CHUNK_SIZE):
                if self.is_cancelled:
                    writer.close()
                    return
                    
                end_idx = min(start_idx + self.CHUNK_SIZE, total_rows)
                chunk = self.df.iloc[start_idx:end_idx]
                
                try:
                    if start_idx == 0:
                        chunk.to_excel(writer, index=False, sheet_name='Sheet1')
                    else:
                        # Append to existing sheet
                        worksheet = writer.sheets['Sheet1']
                        for idx, row in enumerate(chunk.values):
                            for col_idx, value in enumerate(row):
                                worksheet.cell(row=start_idx + idx + 2, column=col_idx + 1, value=value)
                    
                    progress = (end_idx / total_rows) * 100
                    elapsed = (pd.Timestamp.now() - self.start_time).total_seconds()
                    self.progress.emit(f"Exporting Excel: {progress:.1f}% complete... (Elapsed: {elapsed:.1f}s)")
                    
                except Exception as e:
                    writer.close()
                    raise Exception(f"Error while writing Excel chunk: {str(e)}")
            
            writer.close()
        except Exception as e:
            raise Exception(f"Excel export error: {str(e)}")

    def export_parquet_in_chunks(self):
        # Parquet is already optimized for large datasets
        self.progress.emit("Exporting to Parquet format...")
        self.df.to_parquet(self.file_path, index=False)

    def run(self):
        try:
            self.start_time = pd.Timestamp.now()
            self.progress.emit(f"Starting {self.format.upper()} export...")
            
            try:
                if self.format == 'csv':
                    self.export_csv_in_chunks()
                elif self.format == 'excel':
                    self.export_excel_in_chunks()
                elif self.format == 'parquet':
                    self.export_parquet_in_chunks()
                
                if not self.is_cancelled:
                    elapsed = (pd.Timestamp.now() - self.start_time).total_seconds()
                    self.progress.emit(f"Export completed successfully in {elapsed:.1f}s")
                    self.success.emit(self.file_path)
                else:
                    self.progress.emit(f"Export cancelled")
                    self.error.emit("Export was cancelled")
                
            except Exception as e:
                self.error.emit(f"Export error: {str(e)}")
            
            self.finished.emit(self.file_path)
        except Exception as e:
            self.error.emit(f"Unexpected error: {str(e)}")
            self.finished.emit(self.file_path)

    def cancel(self):
        self.is_cancelled = True

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
                # Create and configure the export worker
                self.export_worker = ExportWorker(current_tab.current_df, file_path, format_type)
                self.export_worker.progress.connect(lambda msg: current_tab.status_label.setText(msg))
                self.export_worker.success.connect(lambda path: QMessageBox.information(self, "Success", f"File saved successfully to {path}"))
                self.export_worker.error.connect(lambda msg: QMessageBox.critical(self, "Error", f"Export failed: {msg}"))
                
                # Start the export process
                self.export_worker.run()
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
            
            # Connect to the database and get table list
            conn = duckdb.connect(database=self.current_db_path, read_only=True)
            tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()
            conn.close()
            
            # Add tables to the list widget
            for table in tables:
                self.table_list.addItem(table[0])
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
        
        form_layout = QFormLayout()
        
        # Database name
        self.name_input = QLineEdit()
        form_layout.addRow("Database Name:", self.name_input)
        
        # Database type selection
        self.type_combo = QComboBox()
        self.type_combo.addItems(["DuckDB (.duckdb)", "SQLite (.db)"])
        form_layout.addRow("Database Type:", self.type_combo)
        
        # Location selection
        location_layout = QHBoxLayout()
        self.location_input = QLineEdit()
        self.location_input.setText(str(Path.home()))
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_location)
        location_layout.addWidget(self.location_input)
        location_layout.addWidget(browse_button)
        form_layout.addRow("Save Location:", location_layout)
        
        layout.addLayout(form_layout)
        
        # Status message
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: red;")
        layout.addWidget(self.status_label)
        
        # Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.create_database)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
    
    def browse_location(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.location_input.setText(directory)
    
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
