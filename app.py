import sys
import duckdb
import pandas as pd
import os
import tempfile
import shutil
from pathlib import Path

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLineEdit, QFileDialog, QTableView, QMessageBox, QLabel, QPlainTextEdit, QCompleter,
                             QListWidget, QSplitter, QTabWidget, QProgressDialog)
from PyQt6.QtCore import QAbstractTableModel, Qt, QThread, pyqtSignal, QRegularExpression, QTimer, QObject
from PyQt6.QtGui import QTextCursor, QSyntaxHighlighter, QTextCharFormat


SQL_KEYWORDS = [
    'SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE', 'JOIN', 'INNER', 'LEFT', 'RIGHT',
    'FULL', 'ON', 'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT', 'OFFSET', 'CREATE', 'DROP', 'ALTER',
    'TABLE', 'VIEW', 'INDEX', 'DISTINCT', 'VALUES', 'INTO', 'AS', 'AND', 'OR', 'NOT', 'NULL',
    'PRIMARY', 'KEY', 'FOREIGN', 'REFERENCES', 'DEFAULT', 'CONSTRAINT', 'UNIQUE', 'CHECK',
    'AUTO_INCREMENT', 'CASCADE', 'SET', 'BETWEEN', 'LIKE', 'IN', 'EXISTS', 'ALL', 'ANY', 'SOME'
]

class SQLHighlighter(QSyntaxHighlighter):
    def __init__(self, parent):
        super().__init__(parent)
        self.highlightingRules = []

        # SQL Keywords format (red)
        keywordFormat = QTextCharFormat()
        keywordFormat.setForeground(Qt.GlobalColor.red)
        keywords = SQL_KEYWORDS
        for word in keywords:
            expression = QRegularExpression(f"\\b{word}\\b")
            expression.setPatternOptions(QRegularExpression.PatternOption.CaseInsensitiveOption)
            self.highlightingRules.append((expression, keywordFormat))

        # Single-line comment format (green)
        singleLineCommentFormat = QTextCharFormat()
        singleLineCommentFormat.setForeground(Qt.GlobalColor.darkGreen)
        self.highlightingRules.append((QRegularExpression("--[^\n]*"), singleLineCommentFormat))

        # Multi-line comment format (green)
        self.multiLineCommentFormat = QTextCharFormat()
        self.multiLineCommentFormat.setForeground(Qt.GlobalColor.darkGreen)
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
        # Auto-completion disabled as per user request
        # Removed auto-completion related methods
        
    # Default behavior remains
    pass  # No additional methods needed


class PandasModel(QAbstractTableModel):
    CHUNK_SIZE = 1000  # Number of rows to load at a time

    def __init__(self, df=pd.DataFrame(), parent=None):
        super().__init__(parent)
        self._df = df
        self._total_rows = len(df)
        self._loaded_chunks = {}
        self._column_names = list(df.columns)

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

    def run(self):
        try:
            self.start_time = pd.Timestamp.now()
            # Open a new connection to avoid threading issues
            conn = duckdb.connect(database=self.db_path, read_only=False)
            # Split the queries by semicolon
            queries = [q.strip() for q in self.query.split(';') if q.strip()]
            last_df = None
            
            def update_progress(i, q):
                elapsed = (pd.Timestamp.now() - self.start_time).total_seconds()
                self.progressUpdate.emit(f"Executing query {i+1}/{len(queries)}... (Elapsed: {elapsed:.1f}s)")
            
            for i, q in enumerate(queries):
                update_progress(i, q)
                cur = conn.execute(q)
                # Update progress after execution
                update_progress(i, q)
                # Only fetch data for SELECT queries
                if q.lower().startswith('select'):
                    last_df = cur.fetchdf()
                    # Update progress after fetching data
                    update_progress(i, q)
            conn.close()
            if last_df is None:
                # For commands that do not return data, show a success message
                last_df = pd.DataFrame({'message': ['Command executed successfully']})
            self.resultReady.emit(last_df)
        except Exception as e:
            self.errorOccurred.emit(str(e))

class QueryTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.current_df = None

    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Query editor
        self.query_edit = CodeEditor()
        self.query_edit.setPlaceholderText("Enter SQL query here...")
        layout.addWidget(self.query_edit, 2)

        # Buttons with modern styling
        button_layout = QHBoxLayout()
        
        # Run Query button with status indicator
        self.run_button_container = QWidget()
        run_button_layout = QHBoxLayout(self.run_button_container)
        run_button_layout.setContentsMargins(0, 0, 0, 0)
        
        self.run_query_button = QPushButton("Run Query")
        self.run_query_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; padding: 5px 15px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #cccccc; }"
        )
        self.run_query_button.clicked.connect(self.run_query)
        run_button_layout.addWidget(self.run_query_button)
        
        # Status label
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #666666; margin-left: 10px;")
        run_button_layout.addWidget(self.status_label)
        
        button_layout.addWidget(self.run_button_container)
        button_layout.addStretch()

        # Export buttons
        export_buttons = [
            ("Export CSV", lambda: self.export_data('csv')),
            ("Export Excel", lambda: self.export_data('excel')),
            ("Export Parquet", lambda: self.export_data('parquet'))
        ]

        for text, callback in export_buttons:
            btn = QPushButton(text)
            btn.setStyleSheet(
                "QPushButton { background-color: #008CBA; color: white; padding: 5px 15px; border-radius: 3px; }"
                "QPushButton:hover { background-color: #007399; }"
                "QPushButton:disabled { background-color: #cccccc; }"
            )
            btn.clicked.connect(callback)
            button_layout.addWidget(btn)

        layout.addLayout(button_layout)

        # Results view with styling
        self.table_view = QTableView()
        self.table_view.setStyleSheet(
            "QTableView { border: 1px solid #ddd; }"
            "QHeaderView::section { background-color: #f5f5f5; padding: 5px; border: 1px solid #ddd; }"
            "QTableView::item { padding: 5px; }"
            "QTableView::item:selected { background-color: #e7f3ff; }"
        )
        layout.addWidget(self.table_view, 3)

        # Status bar for export progress
        self.export_status = QLabel()
        self.export_status.setStyleSheet("color: #666666; padding: 5px;")
        layout.addWidget(self.export_status)

    def run_query(self):
        main_window = self.window()
        if not main_window.current_db_path:
            QMessageBox.warning(self, "Warning", "No database loaded")
            return

        tc = self.query_edit.textCursor()
        selected_text = tc.selectedText()
        query = selected_text if selected_text.strip() else self.query_edit.toPlainText().strip()
        if not query:
            QMessageBox.warning(self, "Warning", "Please enter a SQL query")
            return

        # Update UI state
        self.run_query_button.setEnabled(False)
        self.status_label.setText("Executing query...")

        # Start query execution
        self.worker = QueryWorker(main_window.current_db_path, query)
        self.worker.resultReady.connect(self.handle_query_result)
        self.worker.errorOccurred.connect(self.handle_query_error)
        self.worker.progressUpdate.connect(self.update_status)
        self.worker.finished.connect(self.query_finished)
        self.worker.start()

    def update_status(self, message):
        self.status_label.setText(message)

    def query_finished(self):
        self.run_query_button.setEnabled(True)
        self.status_label.setText("Query completed")

    def handle_query_result(self, df):
        self.current_df = df
        model = PandasModel(df)
        self.table_view.setModel(model)
        self.status_label.setText(f"Query completed - {len(df)} rows returned")

    def handle_query_error(self, error):
        self.run_query_button.setEnabled(True)
        self.status_label.setText("Query failed")
        QMessageBox.critical(self, "Query Error", f"Failed to execute query: {error}")

    def export_data(self, format):
        if self.current_df is None or self.current_df.empty:
            QMessageBox.warning(self, "Warning", "No data to export")
            return

        default_filter = "CSV Files (*.csv)" if format=='csv' else "Excel Files (*.xlsx)" if format=='excel' else "Parquet Files (*.parquet)"
        file_ext = ".csv" if format=='csv' else ".xlsx" if format=='excel' else ".parquet"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", default_filter)
        if file_path:
            if not file_path.endswith(file_ext):
                file_path += file_ext

            # Create a copy of the DataFrame to avoid thread issues
            df_copy = self.current_df.copy()
            
            # Create worker thread for export
            self.export_thread = QThread()
            self.export_worker = ExportWorker(df_copy, file_path, format)
            self.export_worker.moveToThread(self.export_thread)
            
            # Connect signals
            self.export_thread.started.connect(self.export_worker.run)
            self.export_worker.finished.connect(self.export_thread.quit)
            self.export_worker.finished.connect(self.export_worker.deleteLater)
            self.export_thread.finished.connect(self.export_thread.deleteLater)
            self.export_worker.progress.connect(self.update_export_status)
            self.export_worker.error.connect(self.handle_export_error)
            self.export_worker.success.connect(self.handle_export_success)
            
            # Start export
            self.export_status.setText(f"Exporting to {format.upper()}...")
            self.export_thread.start()

    def update_export_status(self, message):
        self.export_status.setText(message)

    def handle_export_error(self, error):
        self.export_status.setText("Export failed")
        QMessageBox.critical(self, "Export Error", f"Failed to export data: {error}")

    def handle_export_success(self, file_path):
        self.export_status.setText("Export completed successfully")
        QMessageBox.information(self, "Success", f"Data exported successfully to {file_path}")

    def cancel_query(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.status_label.setText("Query cancelled")
            self.run_query_button.setEnabled(True)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Database Viewer")
        self.connection = None
        self.current_db_path = None
        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create a splitter to hold the available tables list and the query/result area
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left side: available tables list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("Available Tables:"))
        self.table_list = QListWidget()
        self.table_list.setDisabled(True)
        left_layout.addWidget(self.table_list)
        splitter.addWidget(left_panel)

        # Right side: query tabs
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # File loading widgets
        file_layout = QHBoxLayout()
        self.file_line_edit = QLineEdit()
        self.file_line_edit.setReadOnly(True)
        load_button = QPushButton("Load Database")
        load_button.clicked.connect(self.load_database)
        file_layout.addWidget(QLabel("Database File:"))
        file_layout.addWidget(self.file_line_edit)
        file_layout.addWidget(load_button)
        right_layout.addLayout(file_layout)

        # Tab widget for queries
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)
        right_layout.addWidget(self.tab_widget)

        # New Query button
        new_query_button = QPushButton("New Query")
        new_query_button.clicked.connect(self.new_query)
        right_layout.addWidget(new_query_button)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 3)

        # Create initial tab
        self.new_query()

    def load_database(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Database File", "",
                                                   "Database Files (*.duckdb *.db);;All Files (*)")
        if file_path:
            try:
                if self.connection:
                    self.connection.close()
                self.connection = duckdb.connect(database=file_path, read_only=False)
                self.current_db_path = file_path
                self.file_line_edit.setText(file_path)

                # Update available tables
                self.update_table_list()
                
                # Show success message in current tab
                current_tab = self.tab_widget.currentWidget()
                if current_tab:
                    df_message = pd.DataFrame({'message': [f"Database loaded. Use SQL commands to query tables."]})
                    current_tab.current_df = df_message
                    model = PandasModel(df_message)
                    current_tab.table_view.setModel(model)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load database: {e}")

    def update_table_list(self):
        try:
            df_tables = self.connection.execute("SHOW TABLES;").fetchdf()
            table_names = df_tables.iloc[:,0].tolist() if not df_tables.empty else []
            self.table_list.clear()
            self.table_list.addItems(table_names)
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Failed to retrieve tables: {e}")

    def new_query(self):
        new_tab = QueryTab()
        self.tab_widget.addTab(new_tab, f"Query {self.tab_widget.count() + 1}")
        self.tab_widget.setCurrentWidget(new_tab)

    def close_tab(self, index):
        if self.tab_widget.count() > 1:  # Keep at least one tab open
            self.tab_widget.removeTab(index)
        else:
            # If it's the last tab, clear it instead of closing
            tab = self.tab_widget.widget(0)
            tab.query_edit.clear()
            tab.current_df = None
            tab.table_view.setModel(PandasModel())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1000, 600)
    window.show()
    sys.exit(app.exec())

class ExportWorker(QObject):
    finished = pyqtSignal(str)  # Signal to emit when export is complete
    progress = pyqtSignal(str)  # Signal to emit progress updates
    error = pyqtSignal(str)    # Signal to emit if an error occurs
    success = pyqtSignal(str)  # Signal to emit on successful export

    def __init__(self, df, file_path, format):
        super().__init__()
        self.df = df
        self.file_path = file_path
        self.format = format

    def run(self):
        try:
            self.progress.emit(f"Starting {self.format.upper()} export...")
            
            if self.format == 'csv':
                self.df.to_csv(self.file_path, index=False)
            elif self.format == 'excel':
                self.df.to_excel(self.file_path, index=False)
            elif self.format == 'parquet':
                self.df.to_parquet(self.file_path, index=False)
            
            self.progress.emit(f"Export completed successfully")
            self.success.emit(self.file_path)
            self.finished.emit(self.file_path)
        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit(self.file_path)

    def cancel_query(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.status_label.setText("Query cancelled")
            self.run_query_button.setEnabled(True)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Database Viewer")
        self.connection = None
        self.current_db_path = None
        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create a splitter to hold the available tables list and the query/result area
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left side: available tables list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("Available Tables:"))
        self.table_list = QListWidget()
        self.table_list.setDisabled(True)
        left_layout.addWidget(self.table_list)
        splitter.addWidget(left_panel)

        # Right side: query tabs
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # File loading widgets
        file_layout = QHBoxLayout()
        self.file_line_edit = QLineEdit()
        self.file_line_edit.setReadOnly(True)
        load_button = QPushButton("Load Database")
        load_button.clicked.connect(self.load_database)
        file_layout.addWidget(QLabel("Database File:"))
        file_layout.addWidget(self.file_line_edit)
        file_layout.addWidget(load_button)
        right_layout.addLayout(file_layout)

        # Tab widget for queries
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)
        right_layout.addWidget(self.tab_widget)

        # New Query button
        new_query_button = QPushButton("New Query")
        new_query_button.clicked.connect(self.new_query)
        right_layout.addWidget(new_query_button)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 3)

        # Create initial tab
        self.new_query()

    def load_database(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Database File", "",
                                                   "Database Files (*.duckdb *.db);;All Files (*)")
        if file_path:
            try:
                if self.connection:
                    self.connection.close()
                self.connection = duckdb.connect(database=file_path, read_only=False)
                self.current_db_path = file_path
                self.file_line_edit.setText(file_path)

                # Update available tables
                self.update_table_list()
                
                # Show success message in current tab
                current_tab = self.tab_widget.currentWidget()
                if current_tab:
                    df_message = pd.DataFrame({'message': [f"Database loaded. Use SQL commands to query tables."]})
                    current_tab.current_df = df_message
                    model = PandasModel(df_message)
                    current_tab.table_view.setModel(model)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load database: {e}")

    def update_table_list(self):
        try:
            df_tables = self.connection.execute("SHOW TABLES;").fetchdf()
            table_names = df_tables.iloc[:,0].tolist() if not df_tables.empty else []
            self.table_list.clear()
            self.table_list.addItems(table_names)
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Failed to retrieve tables: {e}")

    def new_query(self):
        new_tab = QueryTab()
        self.tab_widget.addTab(new_tab, f"Query {self.tab_widget.count() + 1}")
        self.tab_widget.setCurrentWidget(new_tab)

    def close_tab(self, index):
        if self.tab_widget.count() > 1:  # Keep at least one tab open
            self.tab_widget.removeTab(index)
        else:
            # If it's the last tab, clear it instead of closing
            tab = self.tab_widget.widget(0)
            tab.query_edit.clear()
            tab.current_df = None
            tab.table_view.setModel(PandasModel())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1000, 600)
    window.show()
    sys.exit(app.exec())
