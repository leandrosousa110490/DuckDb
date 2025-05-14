import sys
import duckdb
import os
import json
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QListWidget, QTextEdit, QTableWidget,
    QTableWidgetItem, QLabel, QMessageBox, QSplitter, QHeaderView,
    QInputDialog, QProgressDialog, QMenu, QTabWidget, QDialog,
    QLineEdit, QFormLayout, QDialogButtonBox, QListView, QAbstractItemView,
    QCompleter, QFrame
)
from PyQt6.QtGui import (
    QPalette, QColor, QAction, QSyntaxHighlighter, QTextCharFormat, QFont,
    QTextCursor, QStandardItemModel, QStandardItem
)
from PyQt6.QtCore import Qt, QRegularExpression, QThread, QObject, pyqtSignal, QStringListModel, QRect, QSize
import re
import pandas as pd
import csv
import glob

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

LIGHT_STYLESHEET = """
    QWidget {
        background-color: #f5f5f5;
        color: #333333;
        font-size: 10pt;
    }
    QMainWindow {
        background-color: #f5f5f5;
    }
    QMenuBar {
        background-color: #e0e0e0;
        color: #333333;
    }
    QMenuBar::item {
        background-color: #e0e0e0;
        color: #333333;
    }
    QMenuBar::item:selected {
        background-color: #c0c0c0;
    }
    QMenu {
        background-color: #e0e0e0;
        color: #333333;
        border: 1px solid #c0c0c0;
    }
    QMenu::item:selected {
        background-color: #c0c0c0;
    }
    QPushButton {
        background-color: #e0e0e0;
        color: #333333;
        border: 1px solid #c0c0c0;
        padding: 5px;
        min-height: 15px;
        border-radius: 4px;
    }
    QPushButton:hover {
        background-color: #d0d0d0;
    }
    QPushButton:pressed {
        background-color: #c0c0c0;
    }
    QListWidget {
        background-color: #ffffff;
        color: #333333;
        border: 1px solid #c0c0c0;
        padding: 5px;
    }
    QTextEdit {
        background-color: #ffffff;
        color: #333333;
        border: 1px solid #c0c0c0;
        padding: 5px;
    }
    QTableWidget {
        background-color: #ffffff;
        color: #333333;
        border: 1px solid #c0c0c0;
        gridline-color: #d0d0d0;
    }
    QTableWidget::item {
        padding: 5px;
    }
    QHeaderView::section {
        background-color: #e0e0e0;
        color: #333333;
        padding: 4px;
        border: 1px solid #c0c0c0;
    }
    QLabel {
        color: #333333;
    }
    QSplitter::handle {
        background-color: #e0e0e0;
        border: 1px solid #c0c0c0;
        height: 5px; /* Vertical splitter */
        width: 5px; /* Horizontal splitter */
    }
    QSplitter::handle:hover {
        background-color: #d0d0d0;
    }
    QSplitter::handle:pressed {
        background-color: #c0c0c0;
    }
    QMessageBox {
        background-color: #f5f5f5;
    }
    QMessageBox QLabel {
        color: #333333;
    }
    QMessageBox QPushButton {
        background-color: #e0e0e0;
        color: #333333;
        border: 1px solid #c0c0c0;
        padding: 5px;
        min-width: 70px;
    }
"""

class SQLHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None, dark_theme=True):
        super().__init__(parent)
        self.dark_theme = dark_theme
        self.set_color_theme(dark_theme)

    def set_color_theme(self, dark_theme):
        """Set the color theme for syntax highlighting."""
        self.dark_theme = dark_theme
        
        # Define colors for different themes
        if dark_theme:
            # Dark theme colors
            keyword_color = QColor("#569cd6")  # Blue color for keywords
            string_color = QColor("#ce9178")   # Orange/brown for strings
            comment_color = QColor("#6a9955")  # Green for comments
        else:
            # Light theme colors
            keyword_color = QColor("#0000ff")  # Blue color for keywords
            string_color = QColor("#a31515")   # Red for strings
            comment_color = QColor("#008000")  # Green for comments

        # Set up formats with the theme colors
        self.keyword_format = QTextCharFormat()
        self.keyword_format.setForeground(keyword_color)
        self.keyword_format.setFontWeight(QFont.Weight.Bold)
        
        self.string_format = QTextCharFormat()
        self.string_format.setForeground(string_color)
        
        self.comment_format = QTextCharFormat()
        self.comment_format.setForeground(comment_color)
        self.comment_format.setFontItalic(True)
        
        # Rebuild highlighting rules
        self.update_highlighting_rules()

    def update_highlighting_rules(self):
        """Update highlighting rules with current formats."""
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
        
        self.highlighting_rules = []
        
        # Add keyword rules
        for pattern in keywords:
            self.highlighting_rules.append(
                (QRegularExpression(pattern, QRegularExpression.PatternOption.CaseInsensitiveOption), 
                 self.keyword_format)
            )
        
        # Add string rules
        self.highlighting_rules.append(
            (QRegularExpression("'.*?'"), self.string_format)
        )
        self.highlighting_rules.append(
            (QRegularExpression("\"[^\"]*\""), self.string_format)
        )
        
        # Add comment rule
        self.highlighting_rules.append(
            (QRegularExpression("--.*"), self.comment_format)
        )
        
        # Force rehighlight of the document
        self.rehighlight()

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
            
            # Prepare the query for execution
            query_to_execute = self.query.strip()
            if query_to_execute.endswith(";"):
                query_to_execute = query_to_execute[:-1].strip()
            
            if is_select:
                # Wrap original query (without semicolon) in a count query to get total rows
                count_query = f"SELECT COUNT(*) FROM ({query_to_execute}) AS count_subquery"
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
                if not re.search(r'\bLIMIT\b\s+\d+', query_to_execute, re.IGNORECASE):
                    paginated_query = f"{query_to_execute} LIMIT {self.limit} OFFSET {self.offset}"
                else:
                    paginated_query = query_to_execute
            else:
                paginated_query = query_to_execute

            # Execute the query (paginated for SELECT)
            result_relation = db_conn.execute(paginated_query + ";") # Add semicolon back for execution

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

class SaveQueryDialog(QDialog):
    def __init__(self, query_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Query")
        self.setModal(True)
        
        layout = QFormLayout(self)
        
        self.name_edit = QLineEdit()
        layout.addRow("Query Name:", self.name_edit)
        
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(100)
        layout.addRow("Description:", self.description_edit)
        
        # Preview of the query (read-only)
        self.query_preview = QTextEdit()
        self.query_preview.setPlainText(query_text)
        self.query_preview.setReadOnly(True)
        self.query_preview.setMaximumHeight(150)
        layout.addRow("Query:", self.query_preview)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)
        
        self.resize(400, 350)

class SQLTextEdit(QTextEdit):
    """A custom QTextEdit with SQL autocompletion."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.completer = None
        self.completion_prefix = "" # This stores the part of the word to be completed (e.g., "SEL" or "col")
        self.current_tables_info = {} # {table_name: [column1, column2, ...]}}
        self.keywords = []
        self.auto_parentheses = True
        self.load_keywords()
    
    def load_keywords(self):
        self.keywords = [
            "SELECT", "FROM", "WHERE", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP",
            "TABLE", "VIEW", "INDEX", "TRIGGER", "FUNCTION", "PROCEDURE", "DATABASE", "SCHEMA",
            "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "FULL", "CROSS", "ON", "USING",
            "GROUP BY", "ORDER BY", "HAVING", "LIMIT", "OFFSET", "AS", "DISTINCT", "ALL",
            "UNION", "INTERSECT", "EXCEPT", "IN", "EXISTS", "NOT", "AND", "OR", "BETWEEN",
            "LIKE", "IS NULL", "IS NOT NULL", "DESC", "ASC", "VALUES", "SET", "INTO",
            "COUNT", "SUM", "AVG", "MIN", "MAX", "CASE", "WHEN", "THEN", "ELSE", "END",
            "PRIMARY KEY", "FOREIGN KEY", "REFERENCES", "CHECK", "UNIQUE", "DEFAULT",
            "AUTO_INCREMENT", "CASCADE", "RESTRICT", "PRAGMA", "EXPLAIN", "WITH", "RECURSIVE"
        ]
    
    def set_completer(self, completer):
        if self.completer:
            self.completer.activated.disconnect()
        self.completer = completer
        if self.completer:
            self.completer.setWidget(self)
            self.completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
            self.completer.activated.connect(self.insert_completion)
    
    def update_completions_data(self, tables_info):
        self.current_tables_info = tables_info
        # print(f"[SQLTextEdit] Updated completions data: {self.current_tables_info}")

    def insert_completion(self, completion):
        if self.completer and self.completer.widget() != self:
            return
        tc = self.textCursor()
        # self.completion_prefix was set by keyPressEvent based on get_text_before_cursor...
        prefix_to_replace_len = len(self.completion_prefix) 

        current_pos = tc.position()
        tc.setPosition(current_pos - prefix_to_replace_len)
        tc.movePosition(QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.KeepAnchor, prefix_to_replace_len)
        tc.removeSelectedText()
        tc.insertText(completion)
            
        if self.auto_parentheses and completion.upper() in [
            "COUNT", "SUM", "AVG", "MIN", "MAX", "COALESCE", "IFNULL",
            "NULLIF", "CAST", "EXTRACT", "DATE_TRUNC", "REGEXP_MATCHES"
        ]:
            tc.insertText("()")
            tc.movePosition(QTextCursor.MoveOperation.Left)
        self.setTextCursor(tc)
        if self.completer: self.completer.popup().hide()
    
    def get_text_before_cursor_for_completion(self):
        tc = self.textCursor()
        pos_in_block = tc.positionInBlock()
        text_before_cursor = tc.block().text()[:pos_in_block]

        _unquoted_ident_chars = r'[a-zA-Z0-9_]'
        _unquoted_ident = f'{_unquoted_ident_chars}+'
        _quoted_ident = r'\"[^\"\\r\\n]+\"'
        _any_ident = f'(?:{_unquoted_ident}|{_quoted_ident})'

        pat_ident_dot_prefix = rf'({_any_ident})(\.)({_unquoted_ident_chars}*)$'
        match = re.search(pat_ident_dot_prefix, text_before_cursor)
        if match:
            full_prefix_matched = match.group(0) 
            # completion_trigger_prefix is the part after the dot
            completion_trigger_prefix = match.group(4) if len(match.groups()) >=4 and match.group(4) is not None else ""
            return full_prefix_matched, completion_trigger_prefix

        pat_identifier_alone = rf'({_any_ident})$'
        match = re.search(pat_identifier_alone, text_before_cursor)
        if match:
            full_prefix_matched = match.group(1)
            # For standalone identifiers, the full prefix is also the part to complete
            return full_prefix_matched, full_prefix_matched

        pat_solitary_dot = r'(\.)$'
        match = re.search(pat_solitary_dot, text_before_cursor)
        if match:
            return ".", ""

        return "", ""

    def focusInEvent(self, event):
        if self.completer:
            self.completer.setWidget(self)
        super().focusInEvent(event)
    
    def keyPressEvent(self, event):
        if self.completer and self.completer.popup().isVisible():
            key = event.key()
            if key in [Qt.Key.Key_Enter, Qt.Key.Key_Return, Qt.Key.Key_Tab]:
                current_comp = self.completer.currentCompletion()
                self.completer.popup().hide() 
                self.insert_completion(current_comp)
                return 
            elif key == Qt.Key.Key_Escape:
                self.completer.popup().hide()
                return
            elif key in [Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_PageUp, Qt.Key.Key_PageDown]:
                # Let QCompleter handle these navigation keys for its popup
                pass # Do not return, let superclass handle it after completer potentially acts
        
        if event.key() in [Qt.Key.Key_Enter, Qt.Key.Key_Return]:
            # ... (auto-indent logic remains the same)
            cursor = self.textCursor()
            block = cursor.block()
            text = block.text()
            indentation = ""
            for char in text:
                if char.isspace(): indentation += char
                else: break
            super().keyPressEvent(event)
            if indentation: self.insertPlainText(indentation)
            return
        
        if event.key() == Qt.Key.Key_ParenLeft:
            # ... (auto-pair logic remains the same)
            super().keyPressEvent(event)
            self.insertPlainText(")")
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.Left)
            self.setTextCursor(cursor)
            return
        
        super().keyPressEvent(event) # Allow QTextEdit to process the key press first
        
        # Completion Logic
        text_char = event.text() # Character actually typed, if printable
        full_prefix_text, current_trigger_prefix = self.get_text_before_cursor_for_completion()
        self.completion_prefix = current_trigger_prefix # Store for insert_completion

        print(f"[KPE] Char: '{text_char}', FP: '{full_prefix_text}', TP: '{current_trigger_prefix}'")

        # Conditions to show completer:
        # 1. Printable character typed (event.text() is not empty)
        # 2. OR it's a backspace/delete (to re-evaluate completion)
        # 3. AND (there's a trigger_prefix OR the full_prefix_text ends with a dot)
        # 4. AND it's not a modifier key press alone

        is_char_key = bool(text_char) and not event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.MetaModifier)
        is_meaningful_action = is_char_key or event.key() in [Qt.Key.Key_Backspace, Qt.Key.Key_Delete]

        if is_meaningful_action and (current_trigger_prefix or full_prefix_text.endswith(".")):
            print(f"[KPE] ==> Calling handle_completion for FP: '{full_prefix_text}', TP: '{current_trigger_prefix}'")
            self.handle_completion(full_prefix_text, current_trigger_prefix)
        else:
            if self.completer: self.completer.popup().hide()
            print(f"[KPE] ==> Hiding popup. Meaningful: {is_meaningful_action}, TriggerPrefix: '{current_trigger_prefix}', EndsWithDot: {full_prefix_text.endswith('.')}")

    def parse_aliases(self, query_text):
        # ... (parse_aliases logic remains the same)
        aliases = {}
        patterns = [
            r'\bFROM\s+((?:\"[^\"\r\n]+\"|\w+))\s+(?:AS\s+)?((?:\"[^\"\r\n]+\"|\w+))(?=[\s\r\n\(,;]|$)',
            r'\bJOIN\s+((?:\"[^\"\r\n]+\"|\w+))\s+(?:AS\s+)?((?:\"[^\"\r\n]+\"|\w+))(?=[\s\r\n\(,;]|$)'
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, query_text, re.IGNORECASE):
                table_name = match.group(1).strip('"')
                alias_name = match.group(2).strip('"')
                if alias_name.upper() not in ["ON", "USING", "WHERE", "GROUP", "ORDER", "LIMIT"]:
                     aliases[alias_name] = table_name
        return aliases
    
    def handle_completion(self, full_prefix_text, current_trigger_prefix):
        if not self.completer: return
        print(f"[HC] Handling FP: '{full_prefix_text}', TP: '{current_trigger_prefix}'")
        
        model = QStandardItemModel()
        query_text = self.toPlainText()
        aliases = self.parse_aliases(query_text)
        print(f"[HC] Aliases: {aliases}, Tables: {list(self.current_tables_info.keys())}")
        
        is_after_dot = full_prefix_text.endswith(".") and len(full_prefix_text) > 1 # e.g. "table." or "alias."
        # The part before the dot is in full_prefix_text[:-1]
        # current_trigger_prefix is ALREADY the part *after* the dot, or empty if just "table."

        if is_after_dot:
            table_or_alias_before_dot = full_prefix_text[:-1].strip('"')
            print(f"[HC] After dot. Table/Alias: '{table_or_alias_before_dot}'. Col CP: '{current_trigger_prefix}'")
            actual_table_name = None
            if table_or_alias_before_dot in aliases:
                actual_table_name = aliases[table_or_alias_before_dot]
            elif table_or_alias_before_dot in self.current_tables_info:
                actual_table_name = table_or_alias_before_dot
            
            if actual_table_name and actual_table_name in self.current_tables_info:
                for column in self.current_tables_info[actual_table_name]:
                    if not current_trigger_prefix or column.lower().startswith(current_trigger_prefix.lower()):
                        item = QStandardItem(column); item.setData("column", Qt.ItemDataRole.UserRole); model.appendRow(item)
        else:
            # Not after a dot, suggest keywords, tables, aliases based on current_trigger_prefix
            print(f"[HC] Not after dot. Suggesting based on TP: '{current_trigger_prefix}'")
            for keyword in self.keywords:
                if not current_trigger_prefix or keyword.lower().startswith(current_trigger_prefix.lower()):
                    item = QStandardItem(keyword); item.setData("keyword", Qt.ItemDataRole.UserRole); model.appendRow(item)
            for table_name in self.current_tables_info.keys():
                if not current_trigger_prefix or table_name.lower().startswith(current_trigger_prefix.lower()):
                    item = QStandardItem(table_name); item.setData("table", Qt.ItemDataRole.UserRole); model.appendRow(item)
            for alias, aliased_table_name in aliases.items():
                if not current_trigger_prefix or alias.lower().startswith(current_trigger_prefix.lower()):
                    item_display_text = f'{alias} (alias for {aliased_table_name})' 
                    item = QStandardItem(alias) # Insert the alias itself
                    # item.setData(alias, Qt.DisplayRole) # Redundant, QStandardItem constructor does this
                    item.setData(f'{alias} (alias for {aliased_table_name})', Qt.ItemDataRole.ToolTipRole) # Tooltip
                    item.setData("alias", Qt.ItemDataRole.UserRole) 
                    model.appendRow(item)

        print(f"[HC] Model row count: {model.rowCount()}")
        if model.rowCount() > 0:
            self.completer.setModel(model)
            self.completer.setCompletionPrefix(current_trigger_prefix) # Filter model by this
            cr = self.cursorRect()
            # Basic width calculation, can be improved
            popup_width = self.completer.popup().sizeHintForColumn(0) + 30 
            if self.completer.popup().verticalScrollBar().isVisible():
                 popup_width += self.completer.popup().verticalScrollBar().sizeHint().width()
            cr.setWidth(popup_width)
            self.completer.complete(cr)
            print(f"[HC] Called completer.complete(). Popup visible: {self.completer.popup().isVisible()}")
        else:
            self.completer.popup().hide()
            print(f"[HC] Model empty or no matches, hiding popup.")

class QueryTab(QWidget):
    def __init__(self, parent=None, dark_theme=True):
        super().__init__(parent)
        self.dark_theme = dark_theme
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        query_layout = QVBoxLayout()
        query_layout.addWidget(QLabel("SQL Query:"))
        
        # Use the custom SQLTextEdit with autocompletion
        self.query_editor = SQLTextEdit()
        self.query_editor.setPlaceholderText("Enter your SQL query here...")
        self.highlighter = SQLHighlighter(self.query_editor.document(), self.dark_theme)
        
        # Set up the completer
        completer = QCompleter(self)
        completer.setModelSorting(QCompleter.ModelSorting.CaseSensitivelySortedModel)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setWrapAround(False)
        completer.setMaxVisibleItems(10)
        
        # Set custom popup
        popup = completer.popup()
        popup.setObjectName("completionPopup")
        popup.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        popup.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        popup.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        popup.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Plain)
        
        self.query_editor.set_completer(completer)
        
        query_layout.addWidget(self.query_editor)
        
        button_layout = QHBoxLayout()
        self.run_query_button = QPushButton("Run Query")
        self.run_query_button.setToolTip("Execute the full query or just the selected text if a selection is made")
        button_layout.addWidget(self.run_query_button)
        
        query_layout.addLayout(button_layout)
        layout.addLayout(query_layout)
        
        # Results section
        layout.addWidget(QLabel("Results:"))
        
        # Filter bar
        filter_layout = QHBoxLayout()
        filter_label = QLabel("Filter:")
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Type to filter results...")
        self.filter_input.setClearButtonEnabled(True)  # Add a clear button inside the input field
        self.filter_input.textChanged.connect(self.filter_results)  # Connect to textChanged for dynamic filtering
        
        self.filter_count_label = QLabel("")
        
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.filter_input, 1)  # Give the filter input stretch
        filter_layout.addWidget(self.filter_count_label)
        
        layout.addLayout(filter_layout)
        
        self.results_table = QTableWidget()
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.results_table.verticalHeader().setVisible(False)
        
        # Enable context menu for results table
        self.results_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.results_table.customContextMenuRequested.connect(self.show_results_context_menu)
        
        layout.addWidget(self.results_table, 1)  # Give the results table more vertical space
        
        # Initialize filter variables
        self.all_rows_visible = True  # Track if all rows are currently visible
        
        # Set up keyboard shortcuts
        self.filter_shortcut = QAction(self)
        self.filter_shortcut.setShortcut("Ctrl+F")
        self.filter_shortcut.triggered.connect(self.focus_filter)
        self.addAction(self.filter_shortcut)
        
    def focus_filter(self):
        """Focus the filter input field."""
        self.filter_input.setFocus()
        self.filter_input.selectAll()  # Also select any existing text for easy replacement
    
    def filter_results(self):
        """Filter the results table based on the filter text."""
        filter_text = self.filter_input.text().strip().lower()
        
        # Show all rows if filter is empty
        if not filter_text:
            for row in range(self.results_table.rowCount()):
                self.results_table.setRowHidden(row, False)
            self.filter_count_label.setText("")
            self.all_rows_visible = True
            return
        
        # Count how many rows match the filter
        visible_count = 0
        total_count = self.results_table.rowCount()
        
        # Check each row
        for row in range(total_count):
            # Check if any cell in this row contains the filter text
            row_matches = False
            for col in range(self.results_table.columnCount()):
                item = self.results_table.item(row, col)
                if item and filter_text in item.text().lower():
                    row_matches = True
                    break
            
            # Show/hide row based on match
            self.results_table.setRowHidden(row, not row_matches)
            if row_matches:
                visible_count += 1
        
        # Update filter count label
        self.filter_count_label.setText(f"Showing {visible_count} of {total_count} rows")
        self.all_rows_visible = (visible_count == total_count)
    
    def show_results_context_menu(self, pos):
        """Shows the context menu for the results table cells."""
        # Get the table widget (sender)
        table = self.sender()
        if not table:
            return
            
        # Get the item at the position
        item = table.itemAt(pos)
        
        menu = QMenu()
        
        # Copy cell value action (only if a cell is clicked)
        if item:
            copy_cell_action = QAction("Copy Value", table)
            copy_cell_action.triggered.connect(lambda: self.copy_cell_value(item))
            menu.addAction(copy_cell_action)
            
            # Get row and column indices
            row = item.row()
            column = item.column()
            
            # Copy row action
            copy_row_action = QAction(f"Copy Row {row + 1}", table)
            copy_row_action.triggered.connect(lambda: self.copy_row(table, row))
            menu.addAction(copy_row_action)
            
            # Copy column action
            header_text = table.horizontalHeaderItem(column).text() if table.horizontalHeaderItem(column) else f"Column_{column}"
            copy_column_action = QAction(f"Copy Column '{header_text}'", table)
            copy_column_action.triggered.connect(lambda: self.copy_column(table, column))
            menu.addAction(copy_column_action)
        
        # Copy headers action
        copy_headers_action = QAction("Copy Headers", table)
        copy_headers_action.triggered.connect(lambda: self.copy_headers(table))
        menu.addAction(copy_headers_action)
        
        # Copy table action
        copy_table_action = QAction("Copy Table", table)
        copy_table_action.triggered.connect(lambda: self.copy_table_contents(table))
        menu.addAction(copy_table_action)
        
        # Show the menu at the cursor position
        menu.exec(table.mapToGlobal(pos))
        
    def copy_cell_value(self, item):
        """Copy the value of a single cell to the clipboard."""
        if not item:
            return
            
        # Get the cell value
        value = item.text()
        
        # Copy to clipboard
        clipboard = QApplication.clipboard()
        clipboard.setText(value)
        
    def copy_row(self, table, row):
        """Copy a single row to the clipboard."""
        if not table or row < 0 or row >= table.rowCount():
            return
            
        row_data = []
        for col in range(table.columnCount()):
            item = table.item(row, col)
            value = item.text() if item else ""
            row_data.append(value)
        
        # Copy to clipboard as tab-separated text
        clipboard = QApplication.clipboard()
        clipboard.setText("\t".join(row_data))
        
    def copy_column(self, table, column):
        """Copy a single column to the clipboard."""
        if not table or column < 0 or column >= table.columnCount():
            return
            
        column_data = []
        
        # Get the column header
        header_item = table.horizontalHeaderItem(column)
        header_text = header_item.text() if header_item else f"Column_{column}"
        
        # Add header as first item
        column_data.append(header_text)
        
        # Add column data
        for row in range(table.rowCount()):
            item = table.item(row, column)
            value = item.text() if item else ""
            column_data.append(value)
        
        # Copy to clipboard as newline-separated text
        clipboard = QApplication.clipboard()
        clipboard.setText("\n".join(column_data))
        
    def copy_headers(self, table):
        """Copy the table headers to the clipboard."""
        if not table or table.columnCount() == 0:
            return
            
        headers = []
        for col in range(table.columnCount()):
            header_item = table.horizontalHeaderItem(col)
            headers.append(header_item.text() if header_item else f"Column_{col}")
        
        # Copy to clipboard as tab-separated text
        clipboard = QApplication.clipboard()
        clipboard.setText("\t".join(headers))
        
    def copy_table_contents(self, table):
        """Copy the entire table contents to the clipboard."""
        if not table or table.rowCount() == 0 or table.columnCount() == 0:
            return
            
        rows = []
        
        # Get headers
        headers = []
        for col in range(table.columnCount()):
            header_item = table.horizontalHeaderItem(col)
            headers.append(header_item.text() if header_item else f"Column_{col}")
        
        rows.append("\t".join(headers))
        
        # Get data
        for row in range(table.rowCount()):
            row_data = []
            for col in range(table.columnCount()):
                item = table.item(row, col)
                value = item.text() if item else ""
                row_data.append(value)
            rows.append("\t".join(row_data))
        
        # Copy to clipboard as tab-separated text (works well for pasting into Excel or other tools)
        clipboard = QApplication.clipboard()
        clipboard.setText("\n".join(rows))

class DuckDBApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DuckDB Query Tool")
        self.setGeometry(100, 100, 1200, 800)
        self.db_conn = None
        self.db_path = None
        self.recent_dbs = []
        self.max_recent_dbs = 5
        self.saved_queries = []
        self.query_tabs = []
        self.current_tab_index = 0
        self.dark_theme_enabled = True  # Default to dark theme
        
        # Load recent databases, saved queries, and theme preference
        self.load_recent_dbs()
        self.load_saved_queries()
        self.load_theme_preference()

        self.init_ui()
        
        # Apply theme based on preference
        if self.dark_theme_enabled:
            self.apply_dark_theme()
        else:
            self.apply_light_theme()

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
    
    def load_saved_queries(self):
        """Load saved queries from file."""
        try:
            saved_queries_file = os.path.join(os.path.expanduser("~"), ".duckdb_queries")
            if os.path.exists(saved_queries_file):
                with open(saved_queries_file, "r") as f:
                    self.saved_queries = json.load(f)
        except Exception as e:
            print(f"Error loading saved queries: {e}")
            self.saved_queries = []
    
    def load_theme_preference(self):
        """Load theme preference from file."""
        try:
            theme_file = os.path.join(os.path.expanduser("~"), ".duckdb_theme")
            if os.path.exists(theme_file):
                with open(theme_file, "r") as f:
                    theme_data = f.read().strip()
                    self.dark_theme_enabled = theme_data.lower() == "dark"
            else:
                # Default to dark theme if no preference file exists
                self.dark_theme_enabled = True
        except Exception as e:
            print(f"Error loading theme preference: {e}")
            self.dark_theme_enabled = True  # Default to dark theme on error
    
    def save_theme_preference(self):
        """Save theme preference to file."""
        try:
            theme_file = os.path.join(os.path.expanduser("~"), ".duckdb_theme")
            with open(theme_file, "w") as f:
                f.write("dark" if self.dark_theme_enabled else "light")
        except Exception as e:
            print(f"Error saving theme preference: {e}")
    
    def save_recent_dbs(self):
        """Save list of recently opened databases to file."""
        try:
            recent_dbs_file = os.path.join(os.path.expanduser("~"), ".duckdb_recent")
            with open(recent_dbs_file, "w") as f:
                for db_path in self.recent_dbs:
                    f.write(f"{db_path}\n")
        except Exception as e:
            print(f"Error saving recent databases: {e}")
    
    def save_saved_queries(self):
        """Save queries to file."""
        try:
            saved_queries_file = os.path.join(os.path.expanduser("~"), ".duckdb_queries")
            with open(saved_queries_file, "w") as f:
                json.dump(self.saved_queries, f, indent=2)
        except Exception as e:
            print(f"Error saving queries: {e}")
            
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
        """Apply dark theme to the application."""
        self.setStyleSheet(DARK_STYLESHEET)
        self.dark_theme_enabled = True
        
        # Update theme toggle menu item
        if hasattr(self, 'theme_action'):
            self.theme_action.setText("Switch to Light Theme")
        
        # Update syntax highlighters in all query tabs
        self.update_syntax_highlighters(dark_theme=True)
        
        # Save preference
        self.save_theme_preference()
    
    def apply_light_theme(self):
        """Apply light theme to the application."""
        self.setStyleSheet(LIGHT_STYLESHEET)
        self.dark_theme_enabled = False
        
        # Update theme toggle menu item
        if hasattr(self, 'theme_action'):
            self.theme_action.setText("Switch to Dark Theme")
            
        # Update syntax highlighters in all query tabs
        self.update_syntax_highlighters(dark_theme=False)
        
        # Save preference
        self.save_theme_preference()
    
    def update_syntax_highlighters(self, dark_theme):
        """Update syntax highlighters in all query tabs to match the current theme."""
        for tab in self.query_tabs:
            if hasattr(tab, 'highlighter') and tab.highlighter:
                tab.highlighter.set_color_theme(dark_theme)
    
    def toggle_theme(self):
        """Toggle between dark and light themes."""
        if self.dark_theme_enabled:
            self.apply_light_theme()
        else:
            self.apply_dark_theme()

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
        
        bulk_excel_import_action = QAction("Bulk Import Excel Files from &Folder...", self)
        bulk_excel_import_action.triggered.connect(self.import_excel_folder)
        import_menu.addAction(bulk_excel_import_action)
        
        bulk_csv_import_action = QAction("Bulk Import CSV Files from F&older...", self)
        bulk_csv_import_action.triggered.connect(self.import_csv_folder)
        import_menu.addAction(bulk_csv_import_action)
        
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
        
        # --- View Menu ---
        view_menu = menubar.addMenu("&View")
        
        # Theme toggle action
        theme_text = "Switch to Light Theme" if self.dark_theme_enabled else "Switch to Dark Theme"
        self.theme_action = QAction(theme_text, self)
        self.theme_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(self.theme_action)
        
        # --- Query Menu ---
        query_menu = menubar.addMenu("&Query")
        
        new_tab_action = QAction("&New Query Tab", self)
        new_tab_action.triggered.connect(self.add_query_tab)
        new_tab_action.setShortcut("Ctrl+T")
        query_menu.addAction(new_tab_action)
        
        close_tab_action = QAction("&Close Current Tab", self)
        close_tab_action.triggered.connect(self.close_current_tab)
        close_tab_action.setShortcut("Ctrl+W")
        query_menu.addAction(close_tab_action)
        
        query_menu.addSeparator()
        
        save_query_action = QAction("&Save Query...", self)
        save_query_action.triggered.connect(self.save_query)
        save_query_action.setShortcut("Ctrl+S")
        query_menu.addAction(save_query_action)
        
        # Saved queries submenu
        self.saved_queries_menu = query_menu.addMenu("&Saved Queries")
        self.update_saved_queries_menu()

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

        # --- Right Pane (Tabs for Queries) ---
        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        
        # Create tab widget
        self.query_tab_widget = QTabWidget()
        self.query_tab_widget.setTabsClosable(True)
        self.query_tab_widget.tabCloseRequested.connect(self.close_tab)
        self.query_tab_widget.currentChanged.connect(self.on_tab_changed)
        right_layout.addWidget(self.query_tab_widget)
        
        # Add initial tab
        self.add_query_tab()
        
        main_splitter.addWidget(right_pane)

        # Adjust splitter sizes (optional initial split)
        main_splitter.setSizes([200, 1000]) # Adjust initial width split
        
        # Set initial focus to the query editor
        self.get_current_query_editor().setFocus()
    
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
            
            # Update autocompletion data for all tabs
            self.update_autocompletion_data()
            
            # Clear all tabs' query editors and results tables
            for tab in self.query_tabs:
                tab.results_table.setRowCount(0)
                tab.results_table.setColumnCount(0)
                tab.query_editor.clear()
            
            # Add to recent databases list
            self.add_to_recent_dbs(db_path)
        except Exception as e:
            QMessageBox.critical(self, "Database Error", f"Failed to connect to database:\n{e}")
            self.db_conn = None
            self.db_path = None
            self.db_status_label.setText("Connection failed.")
            self.table_list_widget.clear()
    
    def update_autocompletion_data(self):
        """Update the autocompletion data for all query editors."""
        if not self.db_conn:
            return
        
        try:
            # Get list of tables
            tables = [t[0] for t in self.db_conn.execute("SHOW TABLES").fetchall()]
            
            # Get columns for each table
            columns = {}
            for table in tables:
                column_info = self.db_conn.execute(f"PRAGMA table_info('{table}')").fetchall()
                columns[table] = [col[1] for col in column_info]  # col[1] is the column name
            
            # Update each query editor
            for tab in self.query_tabs:
                tab.query_editor.update_completions_data(columns)
            
        except Exception as e:
            print(f"Error updating autocompletion data: {e}")
    
    def load_tables(self):
        """Loads the list of tables from the connected database into the list widget."""
        self.table_list_widget.clear()
        if not self.db_conn:
            return
        try:
            tables = self.db_conn.execute("SHOW TABLES").fetchall()
            for table in tables:
                self.table_list_widget.addItem(table[0])
            
            # Update autocompletion data
            self.update_autocompletion_data()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not fetch tables:\n{e}")
    
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
                
                # Clear all tabs' query editors and results tables
                for tab in self.query_tabs:
                    tab.results_table.setRowCount(0)
                    tab.results_table.setColumnCount(0)
                    tab.query_editor.clear()
                
                QMessageBox.information(self, "Success", "Database connection closed successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to close database connection:\n{e}")
        else:
            QMessageBox.information(self, "Info", "No database connection is currently open.")

    def display_table_schema(self, current, previous):
        """When a table is selected, populate the query editor with a basic SELECT query."""
        if current:
            table_name = current.text()
            # Only quote the table name if it contains spaces or special characters
            if ' ' in table_name or any(c in table_name for c in '-.+/*()[]{}'):
                quoted_table_name = f'"{table_name}"'
            else:
                quoted_table_name = table_name
            
            query = f"SELECT * FROM {quoted_table_name} LIMIT 100;"
            
            # Get the current query editor and set its text
            current_editor = self.get_current_query_editor()
            if current_editor:
                current_editor.setPlainText(query)
        else:
            # Keep existing query if user clicks away
            pass

    def quote_identifier(self, identifier):
        """Quote SQL identifier (table or column name) only if needed."""
        if ' ' in identifier or any(c in identifier for c in '-.+/*()[]{}'):
            return f'"{identifier}"'
        return identifier

    def execute_query(self, tab_index=None):
        """Starts the background thread to execute the SQL query."""
        # Use the provided tab index or current tab
        if tab_index is None:
            tab_index = self.query_tab_widget.currentIndex()
        
        # Get the query tab
        query_tab = self.query_tabs[tab_index]
        query_editor = query_tab.query_editor
        results_table = query_tab.results_table
        run_button = query_tab.run_query_button
        
        # Check if there's a selection in the editor
        cursor = query_editor.textCursor()
        selected_text = cursor.selectedText()
        
        # Use the selected text if available, otherwise use the entire content
        self.is_partial_query = False
        if selected_text:
            query = selected_text.strip()
            self.is_partial_query = True
        else:
            query = query_editor.toPlainText().strip()

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
        self.current_tab_for_query = tab_index  # Remember which tab executed the query
        
        # --- Setup Progress Dialog ---
        progress_title = "Executing selected query..." if self.is_partial_query else "Executing query..."
        self.query_progress = QProgressDialog(progress_title, "Cancel", 0, 100, self)
        self.query_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.query_progress.setMinimumDuration(300)  # Show after 300ms delay
        self.query_progress.setWindowTitle("Executing Query")
        self.query_progress.setValue(0)
        
        # --- Show Loading Indicator --- 
        if run_button:
            self.original_button_text = run_button.text()
            run_button.setText("Running...")
            run_button.setEnabled(False)

        # Clear previous results immediately for visual feedback
        results_table.setRowCount(0)
        results_table.setColumnCount(0)

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
        
        # Get the tab that executed the query
        tab_index = getattr(self, 'current_tab_for_query', self.query_tab_widget.currentIndex())
        query_tab = self.query_tabs[tab_index]
        results_table = query_tab.results_table
        
        if headers: # SELECT query with results
            # Update table with the returned data
            self._populate_results_table(results_table, data, headers)
            
            # Show pagination controls if we have a large result set
            if total_count > self.rows_per_page:
                pagination_msg = f"Showing page {self.current_page + 1} of {(total_count + self.rows_per_page - 1) // self.rows_per_page} " + \
                                f"(rows {self.current_page * self.rows_per_page + 1}-{min((self.current_page + 1) * self.rows_per_page, total_count)} of {total_count})"
                success_type = "Selected query" if hasattr(self, 'is_partial_query') and self.is_partial_query else "Query"
                QMessageBox.information(self, "Success", f"{success_type} executed successfully.\n{pagination_msg}")
                
                # Add pagination UI if not already present
                self._ensure_pagination_controls(tab_index)
            else:
                success_type = "Selected query" if hasattr(self, 'is_partial_query') and self.is_partial_query else "Query"
                QMessageBox.information(self, "Success", f"{success_type} executed successfully.\n{len(data)} rows returned.")
        else: # Non-SELECT query successful
            results_table.setRowCount(0)
            results_table.setColumnCount(0)
            success_type = "Selected query" if hasattr(self, 'is_partial_query') and self.is_partial_query else "Query"
            QMessageBox.information(self, "Success", f"{success_type} executed successfully (no results returned).")
            # Reload tables in case the schema changed (e.g., CREATE TABLE, DROP TABLE)
            self.load_tables()
    
    def _populate_results_table(self, table_widget, data, headers):
        """Efficiently populate the results table with data."""
        # Temporarily turn off sorting to improve performance
        table_widget.setSortingEnabled(False)
        
        # Clear and set up the table
        table_widget.setRowCount(0)
        table_widget.setColumnCount(len(headers))
        table_widget.setHorizontalHeaderLabels(headers)

        # Pre-allocate rows
        table_widget.setRowCount(len(data))
        
        # Use batch processing to reduce UI updates
        # Block signals while populating to avoid individual cell change events
        table_widget.blockSignals(True)
        
        for row_idx, row_data in enumerate(data):
            for col_idx, cell_data in enumerate(row_data):
                item = QTableWidgetItem(str(cell_data) if cell_data is not None else "NULL")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table_widget.setItem(row_idx, col_idx, item)
            
        # Restore signals when done
        table_widget.blockSignals(False)
        
        # Re-enable sorting
        table_widget.setSortingEnabled(True)
        
        # Adjust column widths - but limit to reasonable size
        table_widget.horizontalHeader().setMinimumSectionSize(50)
        table_widget.horizontalHeader().setDefaultSectionSize(150)
        table_widget.resizeColumnsToContents()
        
        # Set a max width for columns to avoid excessively wide columns
        for col in range(table_widget.columnCount()):
            current_width = table_widget.columnWidth(col)
            if current_width > 300:
                table_widget.setColumnWidth(col, 300)
                
        # Reset filter state in the parent tab
        parent_tab = self.find_parent_tab_for_table(table_widget)
        if parent_tab:
            # Clear the filter input (which will trigger showing all rows)
            parent_tab.filter_input.clear()
            parent_tab.filter_count_label.setText("")
            parent_tab.all_rows_visible = True
    
    def find_parent_tab_for_table(self, table_widget):
        """Find the parent QueryTab that contains this table widget."""
        for tab in self.query_tabs:
            if hasattr(tab, 'results_table') and tab.results_table == table_widget:
                return tab
        return None
    
    def _ensure_pagination_controls(self, tab_index=None):
        """Create or update pagination controls if needed."""
        if tab_index is None:
            tab_index = self.query_tab_widget.currentIndex()
            
        query_tab = self.query_tabs[tab_index]
        
        # Check if we already have a layout with buttons below the results table
        if not hasattr(query_tab, 'pagination_widget'):
            # Create pagination widgets
            query_tab.pagination_widget = QWidget()
            pagination_layout = QHBoxLayout(query_tab.pagination_widget)
            
            query_tab.page_info_label = QLabel()
            pagination_layout.addWidget(query_tab.page_info_label)
            
            pagination_layout.addStretch()
            
            query_tab.prev_page_button = QPushButton("Previous Page")
            query_tab.prev_page_button.clicked.connect(lambda: self._load_prev_page(tab_index))
            pagination_layout.addWidget(query_tab.prev_page_button)
            
            query_tab.next_page_button = QPushButton("Next Page")
            query_tab.next_page_button.clicked.connect(lambda: self._load_next_page(tab_index))
            pagination_layout.addWidget(query_tab.next_page_button)
            
            # Add to the layout that contains the results table
            query_tab.layout().addWidget(query_tab.pagination_widget)
        
        # Update pagination status
        total_pages = (self.total_rows + self.rows_per_page - 1) // self.rows_per_page
        start_row = self.current_page * self.rows_per_page + 1
        end_row = min((self.current_page + 1) * self.rows_per_page, self.total_rows)
        
        query_tab.page_info_label.setText(f"Page {self.current_page + 1} of {total_pages} (rows {start_row}-{end_row} of {self.total_rows})")
        
        # Enable/disable buttons based on current page
        query_tab.prev_page_button.setEnabled(self.current_page > 0)
        query_tab.next_page_button.setEnabled(self.current_page < total_pages - 1)
        
        # Show the pagination controls
        query_tab.pagination_widget.setVisible(True)
    
    def _load_prev_page(self, tab_index=None):
        """Load the previous page of results."""
        if self.current_page > 0:
            self.current_page -= 1
            self._load_current_page(tab_index)
    
    def _load_next_page(self, tab_index=None):
        """Load the next page of results."""
        total_pages = (self.total_rows + self.rows_per_page - 1) // self.rows_per_page
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self._load_current_page(tab_index)
    
    def _load_current_page(self, tab_index=None):
        """Load the current page of results."""
        if tab_index is None:
            tab_index = self.query_tab_widget.currentIndex()
            
        self.current_tab_for_query = tab_index
        
        if not hasattr(self, 'current_query') or not self.current_query:
            return
            
        # Get the tab that will display the results
        query_tab = self.query_tabs[tab_index]
        results_table = query_tab.results_table
            
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
        self.query_worker.success.connect(lambda data, headers, total_count: 
                                         self._on_page_loaded(data, headers, total_count, tab_index))
        self.query_worker.progress.connect(self._on_query_progress)
        self.query_worker.finished.connect(self._on_query_finished)
        self.query_progress.canceled.connect(self.query_worker.cancel)
        
        # Start thread
        self.query_thread.start()
    
    def _on_page_loaded(self, data, headers, total_count, tab_index=None):
        """Handle loading a new page of results."""
        if tab_index is None:
            tab_index = self.query_tab_widget.currentIndex()
            
        query_tab = self.query_tabs[tab_index]
        results_table = query_tab.results_table
        
        # Update the results table with the new page data
        self._populate_results_table(results_table, data, headers)
        
        # Update pagination controls
        self._ensure_pagination_controls(tab_index)

    def _on_query_error(self, error_message):
        """Handles query errors in the UI thread."""
        print("Query Error (UI Thread):", error_message)
        
        # Get the tab that executed the query
        tab_index = getattr(self, 'current_tab_for_query', self.query_tab_widget.currentIndex())
        query_tab = self.query_tabs[tab_index]
        results_table = query_tab.results_table
        
        results_table.setRowCount(0)
        results_table.setColumnCount(0)
        QMessageBox.critical(self, "Query Error", error_message)

    def _on_query_finished(self):
        """Cleans up after query execution in the UI thread."""
        print("Query thread finished.")
        
        # Close the progress dialog if it exists
        if hasattr(self, 'query_progress') and self.query_progress:
            self.query_progress.close()
            
        # Get the tab that executed the query
        tab_index = getattr(self, 'current_tab_for_query', self.query_tab_widget.currentIndex())
        if 0 <= tab_index < len(self.query_tabs):
            query_tab = self.query_tabs[tab_index]
            run_button = query_tab.run_query_button
            
            # Restore button state
            if hasattr(self, 'original_button_text') and self.original_button_text is not None:
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

        # Helper function to quote identifiers only when needed
        def quote_id(identifier):
            if ' ' in identifier or any(c in identifier for c in '-.+/*()[]{}'):
                return f'"{identifier}"'
            return identifier

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
        quoted_table_name = quote_id(table_name)

        if mode == "Create New Table":
            if table_exists:
                raise ValueError(f"Table '{table_name}' already exists.")
            query = f'CREATE TABLE {quoted_table_name} AS {select_with_source};'

        elif mode == "Replace Existing Table":
            # Use DROP CASCADE + CREATE AS to ensure no old constraints (like PKs) remain
            # This forcefully removes the table and anything depending on it.
            drop_query = f'DROP TABLE IF EXISTS {quoted_table_name} CASCADE;'
            print(f"Worker executing: {drop_query}")
            db_conn_worker.execute(drop_query)
            
            # Create the new table from source, without constraints
            query = f'CREATE TABLE {quoted_table_name} AS {select_with_source};'
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
                     # Quote column name only if needed
                    quoted_col_name = quote_id(orig_col_name)
                    alter_statements.append(f'ALTER TABLE {quoted_table_name} ADD COLUMN {quoted_col_name} {col_type};')

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
                # Quote column names only when needed
                cols_str = ', '.join([quote_id(col) for col in orig_source_cols])
                query = f'INSERT INTO {quoted_table_name} ({cols_str}) ({select_with_source});'
            else:
                # Fallback if can't get columns
                query = f'INSERT INTO {quoted_table_name} ({select_with_source});'

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

        # Helper function to quote identifiers only when needed
        def quote_id(identifier):
            if ' ' in identifier or any(c in identifier for c in '-.+/*()[]{}'):
                return f'"{identifier}"'
            return identifier

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
            quoted_table_name = quote_id(table_name)

            if mode == "Create New Table":
                if table_exists:
                    raise ValueError(f"Table '{table_name}' already exists.")
                query = f'CREATE TABLE {quoted_table_name} AS {select_from_view};'

            elif mode == "Replace Existing Table":
                # Use DROP CASCADE + CREATE AS to ensure no old constraints (like PKs) remain
                # This forcefully removes the table and anything depending on it.
                drop_query = f'DROP TABLE IF EXISTS {quoted_table_name} CASCADE;'
                print(f"Worker executing: {drop_query}")
                db_conn_worker.execute(drop_query)
                
                # Create the new table from source, without constraints
                query = f'CREATE TABLE {quoted_table_name} AS {select_from_view};'
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
                        # Quote column name only if needed
                        quoted_col_name = quote_id(orig_col_name)
                        alter_statements.append(f'ALTER TABLE {quoted_table_name} ADD COLUMN {quoted_col_name} {col_type};')

                # Check for cancellation before potentially long ALTER/INSERT
                if worker_ref.is_cancelled: raise InterruptedError("Import cancelled before schema change/insert.")

                if alter_statements:
                    print(f"Worker applying schema changes: {alter_statements}")
                    for alter_query in alter_statements:
                        db_conn_worker.execute(alter_query)

                # Modified INSERT query to explicitly list columns (original case)
                df_columns = [col_info[0] for col_info in source_schema.values()] # Get original names
                # Quote column names only when needed
                source_cols = ', '.join([quote_id(col) for col in df_columns])
                query = f'INSERT INTO {quoted_table_name} ({source_cols}) {select_from_view};'

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
        try:
            if hasattr(self, 'progress') and self.progress is not None:
                if self.progress.isVisible():
                    self.progress.close()
        except RuntimeError:
            # Handle case where dialog was already deleted
            pass
        QMessageBox.information(self, "Success", message)
        self.load_tables() # Refresh table list in UI thread

    def _on_import_error(self, error_message):
        print("Import Error (UI Thread):", error_message)
        try:
            if hasattr(self, 'progress') and self.progress is not None:
                if self.progress.isVisible():
                    self.progress.close()
        except RuntimeError:
            # Handle case where dialog was already deleted
            pass
        QMessageBox.critical(self, "Import Error", error_message)
        # Optionally reload tables even on error?
        # self.load_tables()

    def _safe_update_progress(self, value):
        """Safely update progress dialog value, handling cases where dialog might be closed/deleted."""
        try:
            if hasattr(self, 'progress') and self.progress is not None and self.progress.isVisible():
                self.progress.setValue(value)
        except RuntimeError:
            # Handle case where dialog was already deleted
            pass
            
    def _on_import_finished(self):
        """Clean up resources after import thread is finished."""
        print("Import thread finished.")
        # Ensure progress dialog is closed
        if hasattr(self, 'progress') and self.progress is not None:
            try:
                if self.progress.isVisible():
                    self.progress.close()
            except RuntimeError:
                # Handle case where dialog was already deleted
                pass
            self.progress = None
            
        # Clean up references properly to prevent thread leak
        if hasattr(self, 'import_thread') and self.import_thread is not None:
            try:
                # Important: wait for thread to actually finish before removing references
                if self.import_thread.isRunning():
                    self.import_thread.wait(3000)  # Wait up to 3 seconds for thread to finish
            except Exception as e:
                print(f"Error during thread cleanup: {e}")
                
        # Now clear the references
        self.import_worker = None
        self.import_thread = None

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
        """Import data from an Excel file into a DuckDB table."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Excel File", "", "Excel Files (*.xlsx *.xls *.xlsm)"
        )
        if not file_path:
            return
        
        # Get sheet names for further selection
        try:
            sheet_names = pd.ExcelFile(file_path).sheet_names
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read Excel file: {str(e)}")
            return
            
        if not sheet_names:
            QMessageBox.warning(self, "Warning", "No sheets found in the Excel file.")
            return
            
        # Let user select a sheet
        sheet_name, ok = QInputDialog.getItem(
            self, "Select Sheet", "Choose a sheet to import:", sheet_names, 0, False
        )
        if not ok or not sheet_name:
            return
            
        # Start the import thread
        self._start_import_thread(self._execute_excel_import_core, file_path, sheet_name)

    def import_excel_folder(self):
        """Import multiple Excel files from a folder and append to a single table."""
        # Check if we have a database connection
        if not self.db_conn:
            QMessageBox.warning(self, "Warning", "Please connect to a database first.")
            return
            
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder with Excel Files")
        if not folder_path:
            return
            
        # Find all Excel files in the folder
        excel_files = []
        for ext in ['.xlsx', '.xls', '.xlsm']:
            excel_files.extend([f for f in glob.glob(os.path.join(folder_path, f'*{ext}'))])
        
        if not excel_files:
            QMessageBox.warning(self, "Warning", "No Excel files found in the folder.")
            return
            
        # Get sheet names from the first file
        try:
            first_file_sheets = pd.ExcelFile(excel_files[0]).sheet_names
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read first Excel file: {str(e)}")
            return
            
        if not first_file_sheets:
            QMessageBox.warning(self, "Warning", "No sheets found in the first Excel file.")
            return
            
        # Let user select a sheet
        sheet_name, ok = QInputDialog.getItem(
            self, "Select Sheet", "Choose a sheet to import from all files:", 
            first_file_sheets, 0, False
        )
        if not ok or not sheet_name:
            return
            
        # Ask for table operation mode
        options = ["Create new table", "Add to existing table", "Replace existing table"]
        mode, ok = QInputDialog.getItem(
            self, "Table Operation", 
            "How would you like to handle the target table:", 
            options, 0, False
        )
        if not ok:
            return
            
        # Get table name based on the selected mode
        if mode == options[0]:  # Create new
            table_name, ok = QInputDialog.getText(
                self, "Table Name", "Enter name for the new table:"
            )
            if not ok or not table_name:
                return
                
            # Check if table exists
            if self._table_exists(table_name):
                confirm = QMessageBox.question(
                    self, "Table Exists", 
                    f"Table '{table_name}' already exists. Replace it?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if confirm == QMessageBox.StandardButton.No:
                    return
                    
        elif mode in [options[1], options[2]]:  # Add to or Replace existing
            # Get list of existing tables
            conn = self.db_conn
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
            table_names = [table[0] for table in tables]
            
            if not table_names:
                QMessageBox.warning(self, "Warning", "No tables exist in the database to modify.")
                return
                
            table_name, ok = QInputDialog.getItem(
                self, "Select Table", 
                "Choose the target table:", 
                table_names, 0, False
            )
            if not ok or not table_name:
                return
        
        # Ask for column handling strategy
        col_options = ["Common columns only", "All columns (fill missing with NULL)"]
        col_strategy, ok = QInputDialog.getItem(
            self, "Column Strategy", 
            "How to handle different column structures:", 
            col_options, 1, False  # Default to all columns for more flexibility
        )
        if not ok:
            return
            
        use_common_only = col_strategy == col_options[0]
        
        # Create a progress dialog
        self.progress = QProgressDialog("Preparing to import Excel files...", "Cancel", 0, 100, self)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.show()
        
        # Clean up any existing threads/workers to prevent memory leaks
        if hasattr(self, 'import_thread') and self.import_thread is not None:
            if self.import_thread.isRunning():
                self.import_thread.quit()
                self.import_thread.wait(1000)  # Wait up to 1 second for thread to finish
            self.import_thread = None
        if hasattr(self, 'import_worker'):
            self.import_worker = None
            
        # Create a worker to run in a separate thread
        worker_thread = QThread()
        worker = BulkExcelImportWorker(
            self._get_new_db_connection,
            excel_files,
            table_name,
            sheet_name,
            mode,
            use_common_only
        )
        
        # Connect signals
        worker.progress.connect(self.progress.setValue)
        worker.error.connect(self._on_import_error)
        worker.success.connect(self._on_import_success)
        worker.finished.connect(worker_thread.quit)
        
        worker.finished.connect(self._on_import_finished)
        
        # Start the worker
        worker.moveToThread(worker_thread)
        worker_thread.started.connect(worker.run)
        worker_thread.finished.connect(worker.deleteLater)
        worker_thread.finished.connect(worker_thread.deleteLater)
        
        # Store references to prevent garbage collection
        self.import_worker = worker
        self.import_thread = worker_thread
        
        # Start the thread
        worker_thread.start()

    def import_csv_folder(self):
        """Import multiple CSV files from a folder and append to a single table."""
        # Check if we have a database connection
        if not self.db_conn:
            QMessageBox.warning(self, "Warning", "Please connect to a database first.")
            return
            
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder with CSV Files")
        if not folder_path:
            return
            
        # Find all CSV files in the folder
        csv_files = []
        for ext in ['.csv', '.tsv', '.txt']:
            csv_files.extend([f for f in glob.glob(os.path.join(folder_path, f'*{ext}'))])
        
        if not csv_files:
            QMessageBox.warning(self, "Warning", "No CSV files found in the folder.")
            return
        
        # Ask for delimiter
        delimiters = [
            ("Auto-detect (default)", None), 
            ("Comma (,)", ","), 
            ("Tab (\\t)", "\t"), 
            ("Semicolon (;)", ";"), 
            ("Pipe (|)", "|"), 
            ("Other", "custom")
        ]
        
        delimiter_items = [f"{name}" for name, _ in delimiters]
        delimiter_choice, ok = QInputDialog.getItem(
            self, "CSV Delimiter", 
            "Select the delimiter used in the CSV files:", 
            delimiter_items, 0, False
        )
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
            custom_delimiter, ok = QInputDialog.getText(
                self, "Custom Delimiter", 
                "Enter the custom delimiter character:"
            )
            if not ok or not custom_delimiter:
                return  # User cancelled or entered empty delimiter
            selected_delimiter = custom_delimiter
            
        # Ask for table operation mode
        options = ["Create new table", "Add to existing table", "Replace existing table"]
        mode, ok = QInputDialog.getItem(
            self, "Table Operation", 
            "How would you like to handle the target table:", 
            options, 0, False
        )
        if not ok:
            return
            
        # Get table name based on the selected mode
        if mode == options[0]:  # Create new
            table_name, ok = QInputDialog.getText(
                self, "Table Name", "Enter name for the new table:"
            )
            if not ok or not table_name:
                return
                
            # Check if table exists
            if self._table_exists(table_name):
                confirm = QMessageBox.question(
                    self, "Table Exists", 
                    f"Table '{table_name}' already exists. Replace it?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if confirm == QMessageBox.StandardButton.No:
                    return
                    
        elif mode in [options[1], options[2]]:  # Add to or Replace existing
            # Get list of existing tables
            conn = self.db_conn
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
            table_names = [table[0] for table in tables]
            
            if not table_names:
                QMessageBox.warning(self, "Warning", "No tables exist in the database to modify.")
                return
                
            table_name, ok = QInputDialog.getItem(
                self, "Select Table", 
                "Choose the target table:", 
                table_names, 0, False
            )
            if not ok or not table_name:
                return
        
        # Ask for column handling strategy
        col_options = ["Common columns only", "All columns (fill missing with NULL)"]
        col_strategy, ok = QInputDialog.getItem(
            self, "Column Strategy", 
            "How to handle different column structures:", 
            col_options, 1, False  # Default to all columns for more flexibility
        )
        if not ok:
            return
            
        use_common_only = col_strategy == col_options[0]
        
        # Create a progress dialog
        self.progress = QProgressDialog("Preparing to import CSV files...", "Cancel", 0, 100, self)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.show()
        
        # Clean up any existing threads/workers to prevent memory leaks
        if hasattr(self, 'import_thread') and self.import_thread is not None:
            if self.import_thread.isRunning():
                self.import_thread.quit()
                self.import_thread.wait(1000)  # Wait up to 1 second for thread to finish
            self.import_thread = None
        if hasattr(self, 'import_worker'):
            self.import_worker = None
            
        # Create a worker to run in a separate thread
        worker_thread = QThread()
        worker = BulkCSVImportWorker(
            self._get_new_db_connection,
            csv_files,
            table_name,
            selected_delimiter,
            mode,
            use_common_only
        )
        
        # Connect signals
        worker.progress.connect(self.progress.setValue)
        worker.error.connect(self._on_import_error)
        worker.success.connect(self._on_import_success)
        worker.finished.connect(worker_thread.quit)
        
        worker.finished.connect(self._on_import_finished)
        
        # Start the worker
        worker.moveToThread(worker_thread)
        worker_thread.started.connect(worker.run)
        worker_thread.finished.connect(worker.deleteLater)
        worker_thread.finished.connect(worker_thread.deleteLater)
        
        # Store references to prevent garbage collection
        self.import_worker = worker
        self.import_thread = worker_thread
        
        # Start the thread
        worker_thread.start()

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
                quoted_table_name = self.quote_identifier(table_name)
                drop_query = f'DROP TABLE {quoted_table_name} CASCADE;'
                print(f"Executing: {drop_query}")
                self.db_conn.execute(drop_query)
                self.load_tables() # Refresh list
                QMessageBox.information(self, "Success", f"Table '{table_name}' deleted successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete table '{table_name}':\n{e}")

    # --- Export Methods ---
    
    def _get_results_as_dataframe(self):
        """Get the current results as a pandas dataframe."""
        current_tab = self.get_current_query_tab()
        if not current_tab or not hasattr(current_tab, 'results_table'):
            return None
            
        results_table = current_tab.results_table
        
        if results_table.rowCount() == 0 or results_table.columnCount() == 0:
            return None
        
        # Get column headers
        headers = []
        for col in range(results_table.columnCount()):
            header_item = results_table.horizontalHeaderItem(col)
            headers.append(header_item.text() if header_item else f"Column_{col}")
        
        # Get data
        data = []
        for row in range(results_table.rowCount()):
            row_data = []
            for col in range(results_table.columnCount()):
                item = results_table.item(row, col)
                value = item.text() if item else ""
                # Handle NULL values
                row_data.append(None if value == "NULL" else value)
            data.append(row_data)
        
        # Create dataframe
        df = pd.DataFrame(data, columns=headers)
        return df
    
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

    def save_query(self):
        """Save the current query as a named snippet."""
        current_editor = self.get_current_query_editor()
        if not current_editor:
            return
            
        query_text = current_editor.toPlainText().strip()
        if not query_text:
            QMessageBox.warning(self, "Warning", "Query cannot be empty.")
            return
            
        # Show save dialog
        dialog = SaveQueryDialog(query_text, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            query_name = dialog.name_edit.text().strip()
            query_desc = dialog.description_edit.toPlainText().strip()
            
            if not query_name:
                QMessageBox.warning(self, "Warning", "Query name cannot be empty.")
                return
                
            # Check if name already exists
            for query in self.saved_queries:
                if query['name'] == query_name:
                    confirm = QMessageBox.question(
                        self, "Confirm Overwrite", 
                        f"A query named '{query_name}' already exists. Overwrite?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    if confirm == QMessageBox.StandardButton.No:
                        return
                    # Remove the existing query
                    self.saved_queries.remove(query)
                    break
                    
            # Add the query to the saved list
            self.saved_queries.append({
                'name': query_name,
                'description': query_desc,
                'query': query_text
            })
            
            # Sort by name
            self.saved_queries.sort(key=lambda q: q['name'].lower())
            
            # Save to file
            self.save_saved_queries()
            
            # Update menu
            self.update_saved_queries_menu()
            
            QMessageBox.information(self, "Success", f"Query '{query_name}' saved successfully.")
    
    def load_saved_query(self):
        """Load a saved query into the current editor."""
        action = self.sender()
        if action and isinstance(action, QAction):
            idx = action.data()
            if idx is not None and 0 <= idx < len(self.saved_queries):
                query_info = self.saved_queries[idx]
                
                current_editor = self.get_current_query_editor()
                if current_editor:
                    # Check if editor has content
                    if current_editor.toPlainText().strip():
                        # Ask if user wants to replace or append
                        options = QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
                        response = QMessageBox.question(
                            self, "Load Query", 
                            f"Current editor has content. Replace it with '{query_info['name']}'?\n\n"
                            f"Yes - Replace content\n"
                            f"No - Append to current content\n"
                            f"Cancel - Do nothing",
                            options
                        )
                        
                        if response == QMessageBox.StandardButton.Cancel:
                            return
                        elif response == QMessageBox.StandardButton.Yes:
                            # Replace editor content
                            current_editor.setPlainText(query_info['query'])
                        else:  # No - append
                            # Add a newline if needed
                            current_text = current_editor.toPlainText()
                            if current_text and not current_text.endswith('\n'):
                                current_text += '\n\n'
                            # Append the query
                            current_editor.setPlainText(current_text + query_info['query'])
                    else:
                        # Empty editor, just set the content
                        current_editor.setPlainText(query_info['query'])
                    
                    # Set focus to the editor
                    current_editor.setFocus()
    
    def manage_saved_queries(self):
        """Open dialog to manage saved queries."""
        # This would be a more complex dialog with a list widget
        # For now, we'll implement a simple version that lets users delete queries
        if not self.saved_queries:
            QMessageBox.information(self, "Info", "No saved queries to manage.")
            return
            
        items = [query['name'] for query in self.saved_queries]
        item, ok = QInputDialog.getItem(
            self, "Delete Saved Query", 
            "Select a query to delete:", 
            items, 0, False
        )
        
        if ok and item:
            # Find the index of the query with this name
            for idx, query in enumerate(self.saved_queries):
                if query['name'] == item:
                    confirm = QMessageBox.question(
                        self, "Confirm Delete", 
                        f"Are you sure you want to delete the query '{item}'?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    if confirm == QMessageBox.StandardButton.Yes:
                        self.saved_queries.pop(idx)
                        self.save_saved_queries()
                        self.update_saved_queries_menu()
                        QMessageBox.information(self, "Success", f"Query '{item}' deleted.")
                    break

    def add_query_tab(self):
        """Add a new query tab."""
        new_tab = QueryTab(dark_theme=self.dark_theme_enabled)
        tab_count = self.query_tab_widget.count() + 1
        tab_title = f"Query {tab_count}"
        
        index = self.query_tab_widget.addTab(new_tab, tab_title)
        
        # Connect the run button to our execute_query method
        new_tab.run_query_button.clicked.connect(lambda: self.execute_query(tab_index=index))
        
        # Update autocompletion data for the new tab
        if self.db_conn:
            self.update_autocompletion_data()
        
        # Store reference to the tab
        self.query_tabs.append(new_tab)
        
        # Switch to the new tab
        self.query_tab_widget.setCurrentIndex(index)
        
        return new_tab
    
    def close_tab(self, index):
        """Close a tab by index."""
        if self.query_tab_widget.count() > 1:  # Don't close the last tab
            widget = self.query_tab_widget.widget(index)
            self.query_tab_widget.removeTab(index)
            self.query_tabs.pop(index)
            widget.deleteLater()
    
    def close_current_tab(self):
        """Close the currently active tab."""
        current_index = self.query_tab_widget.currentIndex()
        self.close_tab(current_index)
    
    def on_tab_changed(self, index):
        """Handle tab change events."""
        self.current_tab_index = index
    
    def get_current_query_tab(self):
        """Get the current query tab widget."""
        return self.query_tab_widget.currentWidget()
    
    def get_current_query_editor(self):
        """Get the query editor in the current tab."""
        current_tab = self.get_current_query_tab()
        if current_tab:
            return current_tab.query_editor
        return None
    
    def get_current_results_table(self):
        """Get the results table in the current tab."""
        current_tab = self.get_current_query_tab()
        if current_tab:
            return current_tab.results_table
        return None

    def update_saved_queries_menu(self):
        """Update the saved queries menu."""
        self.saved_queries_menu.clear()
        
        if not self.saved_queries:
            no_saved = QAction("No Saved Queries", self)
            no_saved.setEnabled(False)
            self.saved_queries_menu.addAction(no_saved)
            return
            
        for idx, query_info in enumerate(self.saved_queries):
            query_action = QAction(query_info['name'], self)
            query_action.setToolTip(query_info['description'])
            query_action.setData(idx)  # Store the index in the action
            query_action.triggered.connect(self.load_saved_query)
            self.saved_queries_menu.addAction(query_action)
            
        self.saved_queries_menu.addSeparator()
        manage_action = QAction("Manage Saved Queries...", self)
        manage_action.triggered.connect(self.manage_saved_queries)
        self.saved_queries_menu.addAction(manage_action)


class BulkExcelImportWorker(QObject):
    """Worker class for importing multiple Excel files in a background thread."""
    finished = pyqtSignal()
    error = pyqtSignal(str)
    success = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, db_conn_func, file_paths, table_name, sheet_name, operation_mode, use_common_only):
        super().__init__()
        self.db_conn_func = db_conn_func
        self.file_paths = file_paths
        self.table_name = table_name
        self.sheet_name = sheet_name
        self.operation_mode = operation_mode  # "Create new table", "Add to existing table", "Replace existing table"
        self.use_common_only = use_common_only
        self.is_cancelled = False
        
    def run(self):
        try:
            conn = self.db_conn_func()
            
            # Filter files that contain the requested sheet
            valid_files = []
            self.progress.emit(5)
            
            for i, file_path in enumerate(self.file_paths):
                if self.is_cancelled:
                    return
                
                try:
                    sheet_names = pd.ExcelFile(file_path).sheet_names
                    if self.sheet_name in sheet_names:
                        valid_files.append(file_path)
                except Exception as e:
                    print(f"Warning: Could not read sheet names from {file_path}: {str(e)}")
                    
                # Update progress (10% of total for file validation)
                self.progress.emit(5 + int((i / len(self.file_paths)) * 5))
            
            if not valid_files:
                self.error.emit(f"No files contain the specified sheet '{self.sheet_name}'.")
                self.finished.emit()
                return
                
            # Step 1: Analyze schemas from all valid files
            all_columns = set()
            common_columns = None
            file_schemas = {}
            
            self.progress.emit(10)
            total_valid_files = len(valid_files)
            
            for i, file_path in enumerate(valid_files):
                if self.is_cancelled:
                    return
                
                # Read Excel file header only to get schema
                try:
                    df_sample = pd.read_excel(file_path, sheet_name=self.sheet_name, nrows=1)
                    columns = set(df_sample.columns)
                    file_schemas[file_path] = columns
                    
                    # Track all possible columns
                    all_columns.update(columns)
                    
                    # Track common columns across all files
                    if common_columns is None:
                        common_columns = columns
                    else:
                        common_columns &= columns
                except Exception as e:
                    print(f"Warning: Error reading schema from {file_path}: {str(e)}")
                
                # Update progress (10% for schema analysis)
                self.progress.emit(10 + int((i / total_valid_files) * 10))
            
            # Handle case where there are no common columns
            if self.use_common_only and not common_columns:
                self.error.emit("No common columns found across Excel files. Try using 'All columns' option.")
                self.finished.emit()
                return
                
            # Use either common columns or all columns based on user choice
            columns_to_use = list(common_columns if self.use_common_only else all_columns)
            
            # Handle table creation based on operation mode
            if self.operation_mode == "Replace existing table":
                conn.execute(f'DROP TABLE IF EXISTS "{self.table_name}"')
                
            if self.operation_mode in ["Create new table", "Replace existing table"]:
                # Create schema strings for SQL
                schema_parts = []
                for col in sorted(columns_to_use):
                    schema_parts.append(f'"{col}" VARCHAR')
                
                schema_sql = ", ".join(schema_parts)
                conn.execute(f'CREATE TABLE IF NOT EXISTS "{self.table_name}" ({schema_sql})')
            
            # For "Add to existing table", verify the table exists
            elif self.operation_mode == "Add to existing table":
                tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", 
                                     (self.table_name,)).fetchall()
                if not tables:
                    self.error.emit(f"Table '{self.table_name}' does not exist.")
                    self.finished.emit()
                    return
                
                # Get existing columns for this table
                existing_columns = conn.execute(f'PRAGMA table_info("{self.table_name}")').fetchall()
                existing_column_names = [col[1] for col in existing_columns]
                
                # Add any missing columns to the table
                for col in columns_to_use:
                    if col not in existing_column_names:
                        conn.execute(f'ALTER TABLE "{self.table_name}" ADD COLUMN "{col}" VARCHAR')
                        
            self.progress.emit(20)
            
            # Step 2: Process each file - using pandas instead of read_excel_auto
            for i, file_path in enumerate(valid_files):
                if self.is_cancelled:
                    return
                
                try:
                    temp_table = f"temp_import_{i}"
                    try:
                        # Read the Excel file using pandas
                        df = pd.read_excel(file_path, sheet_name=self.sheet_name)
                        
                        if self.use_common_only:
                            # Filter to only include common columns
                            df = df[list(common_columns)]
                        else:
                            # For all columns, add missing columns as NULL
                            for col in all_columns:
                                if col not in df.columns:
                                    df[col] = None
                        
                        # Use duckdb's DataFrame registration
                        conn.register(temp_table, df)
                        
                        # Insert into main table
                        target_cols = ", ".join([f'"{col}"' for col in sorted(df.columns)])
                        source_cols = ", ".join([f'"{col}"' for col in sorted(df.columns)])
                        
                        conn.execute(f'INSERT INTO "{self.table_name}" ({target_cols}) SELECT {source_cols} FROM {temp_table};')
                    finally:
                        # Unregister the temporary table even if an error occurred
                        try:
                            conn.unregister(temp_table)
                        except:
                            pass  # Ignore errors during unregistration
                    
                except Exception as e:
                    self.error.emit(f"Error processing {file_path}: {str(e)}")
                    self.finished.emit()
                    return
                
                # Update progress (70% allocated for file processing)
                self.progress.emit(20 + int((i / total_valid_files) * 70))
            
            # Get row count
            row_count = conn.execute(f'SELECT COUNT(*) FROM "{self.table_name}"').fetchone()[0]
            file_count = len(valid_files)
            
            # Success message
            self.progress.emit(100)
            self.success.emit(f"Successfully imported {file_count} Excel files into table '{self.table_name}' ({row_count} rows total)")
            
        except Exception as e:
            self.error.emit(f"Error during bulk Excel import: {str(e)}")
        finally:
            self.finished.emit()
            
    def cancel(self):
        self.is_cancelled = True


class BulkCSVImportWorker(QObject):
    """Worker class for importing multiple CSV files in a background thread."""
    finished = pyqtSignal()
    error = pyqtSignal(str)
    success = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, db_conn_func, file_paths, table_name, delimiter, operation_mode, use_common_only):
        super().__init__()
        self.db_conn_func = db_conn_func
        self.file_paths = file_paths
        self.table_name = table_name
        self.delimiter = delimiter
        self.operation_mode = operation_mode  # "Create new table", "Add to existing table", "Replace existing table"
        self.use_common_only = use_common_only
        self.is_cancelled = False
        
    def run(self):
        try:
            conn = self.db_conn_func()
            
            # Step 1: Analyze schemas from all valid files
            all_columns = set()
            common_columns = None
            file_schemas = {}
            
            self.progress.emit(5)
            total_files = len(self.file_paths)
            
            for i, file_path in enumerate(self.file_paths):
                if self.is_cancelled:
                    return
                
                # Read CSV file header only to get schema
                try:
                    # Determine appropriate read function based on delimiter
                    if self.delimiter is None:
                        # Auto-detect
                        df_sample = pd.read_csv(file_path, nrows=1)
                    else:
                        df_sample = pd.read_csv(file_path, delimiter=self.delimiter, nrows=1)
                    
                    columns = set(df_sample.columns)
                    file_schemas[file_path] = columns
                    
                    # Track all possible columns
                    all_columns.update(columns)
                    
                    # Track common columns across all files
                    if common_columns is None:
                        common_columns = columns
                    else:
                        common_columns &= columns
                except Exception as e:
                    print(f"Warning: Error reading schema from {file_path}: {str(e)}")
                
                # Update progress (15% for schema analysis)
                self.progress.emit(5 + int((i / total_files) * 15))
            
            # Handle case where there are no common columns
            if self.use_common_only and not common_columns:
                self.error.emit("No common columns found across CSV files. Try using 'All columns' option.")
                self.finished.emit()
                return
                
            # Use either common columns or all columns based on user choice
            columns_to_use = list(common_columns if self.use_common_only else all_columns)
            
            # Handle table creation based on operation mode
            if self.operation_mode == "Replace existing table":
                conn.execute(f'DROP TABLE IF EXISTS "{self.table_name}"')
                
            if self.operation_mode in ["Create new table", "Replace existing table"]:
                # Create schema strings for SQL
                schema_parts = []
                for col in sorted(columns_to_use):
                    schema_parts.append(f'"{col}" VARCHAR')
                
                schema_sql = ", ".join(schema_parts)
                conn.execute(f'CREATE TABLE IF NOT EXISTS "{self.table_name}" ({schema_sql})')
            
            # For "Add to existing table", verify the table exists
            elif self.operation_mode == "Add to existing table":
                tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", 
                                     (self.table_name,)).fetchall()
                if not tables:
                    self.error.emit(f"Table '{self.table_name}' does not exist.")
                    self.finished.emit()
                    return
                
                # Get existing columns for this table
                existing_columns = conn.execute(f'PRAGMA table_info("{self.table_name}")').fetchall()
                existing_column_names = [col[1] for col in existing_columns]
                
                # Add any missing columns to the table
                for col in columns_to_use:
                    if col not in existing_column_names:
                        conn.execute(f'ALTER TABLE "{self.table_name}" ADD COLUMN "{col}" VARCHAR')
                        
            self.progress.emit(20)
            
            # Step 2: Process each file using pandas and DuckDB
            for i, file_path in enumerate(self.file_paths):
                if self.is_cancelled:
                    return
                
                try:
                    temp_table = f"temp_import_{i}"
                    try:
                        # Read the CSV file based on delimiter
                        if self.delimiter is None:
                            # Auto-detect
                            df = pd.read_csv(file_path)
                        else:
                            df = pd.read_csv(file_path, delimiter=self.delimiter)
                        
                        # Process columns based on strategy
                        if self.use_common_only:
                            # Filter to only include common columns
                            df = df[list(common_columns)]
                        else:
                            # For all columns, add missing columns as NULL
                            for col in all_columns:
                                if col not in df.columns:
                                    df[col] = None
                        
                        # Use duckdb's DataFrame registration
                        conn.register(temp_table, df)
                        
                        # Insert into main table
                        target_cols = ", ".join([f'"{col}"' for col in sorted(df.columns)])
                        source_cols = ", ".join([f'"{col}"' for col in sorted(df.columns)])
                        
                        conn.execute(f'INSERT INTO "{self.table_name}" ({target_cols}) SELECT {source_cols} FROM {temp_table};')
                    finally:
                        # Unregister the temporary table even if an error occurred
                        try:
                            conn.unregister(temp_table)
                        except:
                            pass  # Ignore errors during unregistration
                    
                except Exception as e:
                    self.error.emit(f"Error processing {file_path}: {str(e)}")
                    self.finished.emit()
                    return
                
                # Update progress (70% allocated for file processing)
                self.progress.emit(20 + int((i / total_files) * 70))
            
            # Get row count
            row_count = conn.execute(f'SELECT COUNT(*) FROM "{self.table_name}"').fetchone()[0]
            file_count = len(self.file_paths)
            
            # Success message
            self.progress.emit(100)
            self.success.emit(f"Successfully imported {file_count} CSV files into table '{self.table_name}' ({row_count} rows total)")
            
        except Exception as e:
            self.error.emit(f"Error during bulk CSV import: {str(e)}")
        finally:
            self.finished.emit()
            
    def cancel(self):
        self.is_cancelled = True


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = DuckDBApp()
    main_win.show()
    sys.exit(app.exec()) 


