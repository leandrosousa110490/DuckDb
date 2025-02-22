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
from PyQt6.QtCore import QAbstractTableModel, Qt, QThread, pyqtSignal, QRegularExpression, QTimer
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
            for i, q in enumerate(queries):
                elapsed = (pd.Timestamp.now() - self.start_time).total_seconds()
                self.progressUpdate.emit(f"Executing query {i+1}/{len(queries)}... (Elapsed: {elapsed:.1f}s)")
                cur = conn.execute(q)
                # Only fetch data for SELECT queries
                if q.lower().startswith('select'):
                    last_df = cur.fetchdf()
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

        # Buttons
        button_layout = QHBoxLayout()
        run_query_button = QPushButton("Run Query")
        run_query_button.clicked.connect(self.run_query)
        button_layout.addWidget(run_query_button)

        export_csv_button = QPushButton("Export CSV")
        export_csv_button.clicked.connect(lambda: self.export_data('csv'))
        button_layout.addWidget(export_csv_button)

        export_excel_button = QPushButton("Export Excel")
        export_excel_button.clicked.connect(lambda: self.export_data('excel'))
        button_layout.addWidget(export_excel_button)

        export_parquet_button = QPushButton("Export Parquet")
        export_parquet_button.clicked.connect(lambda: self.export_data('parquet'))
        button_layout.addWidget(export_parquet_button)

        layout.addLayout(button_layout)

        # Results view
        self.table_view = QTableView()
        layout.addWidget(self.table_view, 3)

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

        # Create and show progress dialog
        self.progress = QProgressDialog("Executing query...", "Cancel", 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.setAutoClose(True)
        self.progress.setAutoReset(True)
        self.progress.setMinimumDuration(0)
        self.progress.canceled.connect(self.cancel_query)

        # Start query execution
        self.worker = QueryWorker(main_window.current_db_path, query)
        self.worker.resultReady.connect(self.handle_query_result)
        self.worker.errorOccurred.connect(self.handle_query_error)
        self.worker.progressUpdate.connect(self.progress.setLabelText)
        self.worker.finished.connect(self.progress.close)
        self.worker.start()

    def handle_query_result(self, df):
        self.current_df = df
        model = PandasModel(df)
        self.table_view.setModel(model)

    def handle_query_error(self, error):
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

            # Create and show progress dialog for export
            progress = QProgressDialog(f"Exporting data to {format.upper()}...", "Cancel", 0, 0, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setAutoClose(True)
            progress.setMinimumDuration(0)

            # Create worker thread for export
            self.export_worker = QThread()
            self.export_worker.run = lambda: self.export_worker_run(file_path, format, progress)
            self.export_worker.finished.connect(progress.close)
            self.export_worker.start()

    def export_worker_run(self, file_path, format, progress):
        try:
            start_time = pd.Timestamp.now()
            progress.setLabelText(f"Exporting to {format.upper()}... (0.0s)")

            # Update progress message with timer
            timer = QTimer()
            timer.timeout.connect(lambda: progress.setLabelText(
                f"Exporting to {format.upper()}... ({(pd.Timestamp.now() - start_time).total_seconds():.1f}s)"))
            timer.start(100)

            if format == 'csv':
                self.current_df.to_csv(file_path, index=False)
            elif format == 'excel':
                self.current_df.to_excel(file_path, index=False)
            elif format == 'parquet':
                self.current_df.to_parquet(file_path, index=False)

            timer.stop()
            QMessageBox.information(self, "Success", f"Data exported successfully to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export data: {e}")

    def cancel_query(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()

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
