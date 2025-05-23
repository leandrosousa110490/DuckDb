import sys
import os
import duckdb
import pandas as pd
import re
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QLabel, QLineEdit, QTextEdit, QTableWidget, QTableWidgetItem,
    QMessageBox, QComboBox, QHeaderView, QMenu, QApplication
)
from PyQt6.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor, QFont
from PyQt6.QtCore import Qt, QRegularExpression

# Assuming SQLHighlighter is in app.py or a utility module accessible here
# For now, let's put a simplified version here or assume it's imported.
# from app import SQLHighlighter # Or your actual SQLHighlighter location

class SimpleSQLHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighting_rules = []

        keyword_format = QTextCharFormat()
        keyword_format.setForeground(Qt.GlobalColor.blue)
        keyword_format.setFontWeight(QFont.Weight.Bold)
        keywords = [
            "\\bSELECT\\b", "\\bFROM\\b", "\\bWHERE\\b", "\\bINSERT\\b", "\\bUPDATE\\b",
            "\\bDELETE\\b", "\\bCREATE\\b", "\\bALTER\\b", "\\bDROP\\b", "\\bTABLE\\b",
            "\\bJOIN\\b", "\\bINNER\\b", "\\bLEFT\\b", "\\bRIGHT\\b", "\\bOUTER\\b",
            "\\bGROUP\\b", "\\bBY\\b", "\\bORDER\\b", "\\bLIMIT\\b", "\\bAS\\b", "\\bON\\b",
            "\\bAND\\b", "\\bOR\\b", "\\bNOT\\b", "\\bNULL\\b", "\\bIS\\b",
            "\\bCASE\\b", "\\bWHEN\\b", "\\bTHEN\\b", "\\bELSE\\b", "\\bEND\\b",
            # DuckDB read functions
            "\\bread_csv_auto\\b", "\\bread_parquet\\b", "\\bread_json_auto\\b", "\\bread_excel\\b"
        ]
        for pattern in keywords:
            self.highlighting_rules.append((QRegularExpression(pattern, QRegularExpression.PatternOption.CaseInsensitiveOption), keyword_format))

        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#ce9178")) # Orange-like for strings
        self.highlighting_rules.append((QRegularExpression("'.*?'"), string_format))
        self.highlighting_rules.append((QRegularExpression("\".*?\""), string_format)) # For "table_alias"

        comment_format = QTextCharFormat()
        comment_format.setForeground(Qt.GlobalColor.darkGreen)
        comment_format.setFontItalic(True)
        self.highlighting_rules.append((QRegularExpression("--.*"), comment_format))

    def highlightBlock(self, text):
        for pattern, format_rule in self.highlighting_rules:
            expression = QRegularExpression(pattern)
            it = expression.globalMatch(text)
            while it.hasNext():
                match = it.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), format_rule)

class QueryFileDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Query File")
        self.setGeometry(150, 150, 900, 700)
        self.setMinimumSize(600, 400)

        self.file_path = None
        self.current_df = None # For Excel/PDF data
        self.sheet_names = []
        self.db_conn = None # In-memory DuckDB connection
        self.currently_registered_excel_sheet_name = None # Tracks if an Excel sheet is active as "Data"

        self.init_ui()
        self._connect_db()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # File Selection Area
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected.")
        file_layout.addWidget(self.file_label, 1)
        
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(self.browse_button)
        main_layout.addLayout(file_layout)

        # Sheet Selector (for Excel)
        self.sheet_selector_label = QLabel("Sheet:")
        self.sheet_selector = QComboBox()
        self.sheet_selector.currentIndexChanged.connect(self.sheet_changed)
        self.sheet_selector_label.setVisible(False)
        self.sheet_selector.setVisible(False)
        
        sheet_layout = QHBoxLayout()
        sheet_layout.addWidget(self.sheet_selector_label)
        sheet_layout.addWidget(self.sheet_selector)
        main_layout.addLayout(sheet_layout)

        # SQL Query Editor
        main_layout.addWidget(QLabel("SQL Query (use 'Data' as the table name for the loaded file/sheet):"))
        self.query_editor = QTextEdit()
        self.query_editor.setPlaceholderText("Example: SELECT * FROM Data WHERE column_name > 10")
        self.highlighter = SimpleSQLHighlighter(self.query_editor.document())
        main_layout.addWidget(self.query_editor, 1) # Give query editor stretch

        # Run Query Button
        self.run_query_button = QPushButton("Run Query")
        self.run_query_button.clicked.connect(self.run_query)
        main_layout.addWidget(self.run_query_button)

        # Results Area
        main_layout.addWidget(QLabel("Results:"))
        self.results_table = QTableWidget()
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.results_table.customContextMenuRequested.connect(self._show_results_table_context_menu)
        # Style selected cells to have a light blue background
        self.results_table.setStyleSheet("QTableWidget::item:selected { background-color: #DCEBFF; color: black; }")
        main_layout.addWidget(self.results_table, 2) # Give results table more stretch

        # Download buttons
        download_layout = QHBoxLayout()
        self.download_csv_button = QPushButton("Download as CSV")
        self.download_csv_button.clicked.connect(self._download_results_as_csv)
        download_layout.addWidget(self.download_csv_button)

        self.download_excel_button = QPushButton("Download as Excel")
        self.download_excel_button.clicked.connect(self._download_results_as_excel)
        download_layout.addWidget(self.download_excel_button)
        main_layout.addLayout(download_layout)

        # Status bar (optional, for messages)
        self.status_label = QLabel("")
        main_layout.addWidget(self.status_label)

    def _connect_db(self):
        try:
            self.db_conn = duckdb.connect(database=':memory:', read_only=False)
            self.status_label.setText("In-memory DB ready.")
        except Exception as e:
            QMessageBox.critical(self, "DB Error", f"Failed to initialize in-memory DuckDB: {e}")
            self.status_label.setText("DB connection failed.")

    def browse_file(self):
        options = QFileDialog.Option.DontUseNativeDialog
        filePath, _ = QFileDialog.getOpenFileName(
            self, "Select File to Query", "",
            "Data Files (*.csv *.parquet *.json *.xlsx *.xls *.xlsm);;CSV Files (*.csv);;Parquet Files (*.parquet);;JSON Files (*.json);;Excel Files (*.xlsx *.xls *.xlsm);;All Files (*)",
            options=options
        )
        if filePath:
            self.file_path = filePath
            self.file_label.setText(os.path.basename(filePath))
            self.query_editor.clear()
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            self.current_df = None
            self.sheet_selector.clear()
            self.sheet_selector_label.setVisible(False)
            self.sheet_selector.setVisible(False)

            file_ext = os.path.splitext(filePath)[1].lower()
            
            if file_ext in ['.xlsx', '.xls', '.xlsm']:
                self.load_excel_sheets()
            elif file_ext == '.pdf':
                 QMessageBox.information(self, "PDF Querying", 
                                        "Querying PDF files directly for tables is complex. "
                                        "For now, please convert PDF tables to CSV or Excel first using a dedicated tool. "
                                        "Future versions might include basic PDF table extraction.")
                 self.file_path = None # Reset as we are not handling PDF directly yet
                 self.file_label.setText("No file selected (PDF not directly queryable yet).")

            else: # CSV, Parquet, JSON
                # If "Data" was registered (e.g. from a previous Excel load), unregister it.
                if hasattr(self, 'currently_registered_excel_sheet_name') and self.currently_registered_excel_sheet_name:
                    try:
                        self.db_conn.unregister("Data")
                        print(f"QueryFileDialog: Unregistered 'Data' (Excel sheet: {self.currently_registered_excel_sheet_name}).")
                    except Exception as e_unreg:
                        print(f"QueryFileDialog: Error unregistering 'Data': {e_unreg}")
                
                self.query_editor.setPlaceholderText(f"Example: SELECT * FROM Data WHERE ...")
                self._trigger_auto_query() # Auto-query for CSV/Parquet/JSON


    def load_excel_sheets(self):
        if not self.file_path:
            return
        try:
            xls = pd.ExcelFile(self.file_path)
            self.sheet_names = xls.sheet_names
            if not self.sheet_names:
                QMessageBox.warning(self, "Excel Error", "No sheets found in the Excel file.")
                return

            self.sheet_selector.addItems(self.sheet_names)
            self.sheet_selector_label.setVisible(True)
            self.sheet_selector.setVisible(True)
            if self.sheet_names:
                self.sheet_changed(0) # Trigger load for the first sheet
                # Placeholder text is now set in sheet_changed

        except Exception as e:
            QMessageBox.critical(self, "Excel Read Error", f"Could not read Excel file: {e}")
            self.file_path = None
            self.file_label.setText("Error reading Excel. Select another file.")
            self.sheet_selector_label.setVisible(False)
            self.sheet_selector.setVisible(False)
            self.status_label.setText(f"Error loading sheet '{self.sheet_names[0]}'.")
            self.currently_registered_excel_sheet_name = None # Clear tracker on error
            self.query_editor.setPlainText("") # Clear query on error
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            
    def sheet_changed(self, index):
        if not self.file_path or not self.sheet_names or index < 0 or index >= len(self.sheet_names):
            return
        
        sheet_name = self.sheet_names[index]
        fixed_alias = "Data" # The consistent alias we will use

        if self.db_conn:
            try:
                # Unregister the fixed alias if it was previously used (for another sheet or file type)
                # This ensures a clean state for the "Data" alias.
                try:
                    self.db_conn.unregister(fixed_alias)
                    print(f"Unregistered previous alias '{fixed_alias}' before loading new sheet.")
                    self.currently_registered_excel_sheet_name = None # Clear tracker
                except Exception: # DuckDBError if not registered, which is fine
                    pass 

                self.current_df = pd.read_excel(self.file_path, sheet_name=sheet_name)
                self.db_conn.register(fixed_alias, self.current_df) # Register as "Data"
                self.currently_registered_excel_sheet_name = sheet_name # Track which *actual* sheet is aliased to "Data"
                
                self.status_label.setText(f"Sheet '{sheet_name}' loaded. Query it as '{fixed_alias}'.")
                self.query_editor.setPlaceholderText(f"Example: SELECT * FROM {fixed_alias} WHERE ...")
                self._trigger_auto_query() # Run auto-query after sheet is loaded
            except Exception as e:
                QMessageBox.critical(self, "Sheet Load Error", f"Could not load sheet '{sheet_name}': {e}")
                self.status_label.setText(f"Error loading sheet '{sheet_name}'.")
                self.currently_registered_excel_sheet_name = None # Clear tracker on error
                self.query_editor.setPlainText("") # Clear query on error
                self.results_table.setRowCount(0)
                self.results_table.setColumnCount(0)

    def _trigger_auto_query(self):
        if not self.file_path and not self.currently_registered_excel_sheet_name:
            # No file/sheet loaded that can be aliased as "Data" yet
            return 
        
        default_query = "SELECT * FROM Data LIMIT 100;"
        self.query_editor.setPlainText(default_query)
        # Small delay or QTimer.singleShot might be good if run_query relies on UI fully settling,
        # but usually direct call is fine if setup is complete.
        self.run_query()

    def run_query(self):
        if not self.db_conn:
            QMessageBox.warning(self, "DB Error", "Database connection is not available.")
            return

        query = self.query_editor.toPlainText().strip()
        if not query:
            QMessageBox.warning(self, "Input Error", "Query cannot be empty.")
            return
        
        fixed_alias = "Data"
        final_query = query
        file_ext = ""
        temp_view_created_for_query = False

        if self.file_path:
            file_ext = os.path.splitext(self.file_path)[1].lower()

        is_excel_sheet_active_as_data = (
            hasattr(self, 'currently_registered_excel_sheet_name') and 
            self.currently_registered_excel_sheet_name is not None and
            file_ext in ['.xlsx', '.xls', '.xlsm'] # Ensure it is indeed an excel file path active
        )

        # Check if the query text uses the fixed_alias (e.g., "FROM Data", "Data.column")
        # This regex looks for "Data" as a whole word, case insensitive.
        query_uses_fixed_alias = re.search(rf"\b{fixed_alias}\b", query, re.IGNORECASE)

        if is_excel_sheet_active_as_data:
            # Excel sheet is loaded and registered as "Data". 
            # The query should ideally use "Data". If not, it might be an error or an advanced query.
            if not query_uses_fixed_alias:
                # Optional: You could warn the user here if they are not querying "Data"
                # QMessageBox.information(self, "Query Hint", f"An Excel sheet is loaded as '{fixed_alias}'. Ensure your query uses this alias if intended.")
                pass # Proceed with the query as written by the user
            self.status_label.setText(f"Querying loaded Excel sheet ('{self.currently_registered_excel_sheet_name}') as '{fixed_alias}'...")
            # final_query remains as is (the user's original query)
        
        elif query_uses_fixed_alias and self.file_path and file_ext not in ['.xlsx', '.xls', '.xlsm']:
            # A CSV, Parquet, or JSON file is selected, and the query uses the "Data" alias.
            # We need to create a temporary view for this query.
            read_func_str = None
            
            # Prepare path for SQL: replace backslashes and escape single quotes
            path_for_sql = self.file_path.replace('\\', '/') # Replace backslashes with forward slashes
            path_for_sql = path_for_sql.replace("'", "''")   # Escape single quotes for SQL

            if file_ext == '.csv':
                read_func_str = f"read_csv_auto('{path_for_sql}')"
            elif file_ext == '.parquet':
                read_func_str = f"read_parquet('{path_for_sql}')"
            elif file_ext == '.json':
                read_func_str = f"read_json_auto('{path_for_sql}')"
            else:
                QMessageBox.warning(self, "File Type Error", f"Cannot create a queryable view for file type: {file_ext}")
                return

            if read_func_str:
                try:
                    # Ensure "Data" alias is clean before creating a temp view
                    # (e.g. if an Excel was previously loaded and somehow not unregistered, though sheet_changed should handle it)
                    try: self.db_conn.unregister(fixed_alias)
                    except Exception: pass
                    
                    view_creation_query = f'CREATE OR REPLACE TEMP VIEW "{fixed_alias}" AS SELECT * FROM {read_func_str};'
                    print(f"Creating temporary view: {view_creation_query}")
                    self.db_conn.execute(view_creation_query)
                    temp_view_created_for_query = True
                    self.status_label.setText(f"Querying {file_ext} file as '{fixed_alias}'...")
                    # final_query remains as is, as it already references "Data"
                except Exception as e_view:
                    QMessageBox.critical(self, "Query Setup Error", f"Failed to create temporary view '{fixed_alias}' for querying: {e_view}")
                    return
        elif not self.file_path:
            QMessageBox.warning(self, "Input Error", "Please select a file first.")
            return
        # else: A file is loaded (CSV/Parquet/JSON), but the query does NOT use the "Data" alias.
        # Or, an Excel file is loaded but the query doesn't use "Data" (covered by the first `if` block's `else` path).
        # In this case, final_query remains the user's original query. It might be a direct call to
        # read_csv_auto(), etc., or an error. Let DuckDB handle it.
        # No status label change here, as we are not sure what the user is doing.

        try:
            if not is_excel_sheet_active_as_data and not temp_view_created_for_query and query_uses_fixed_alias and self.file_path:
                 # This case implies query_uses_fixed_alias, a file is loaded, it's not excel, but temp view creation failed or was skipped.
                 # This should ideally be caught by earlier checks, but as a safeguard:
                 QMessageBox.warning(self, "Query Hint", f"Query uses '{fixed_alias}' but no active source is aliased as '{fixed_alias}'. Please check file selection or query.")
                 return

            current_status = self.status_label.text()
            if not current_status.endswith("..."): # If status wasn't set by specific logic above
                 self.status_label.setText(f"Executing query...")
            else: # Append to existing status
                self.status_label.setText(current_status)

            print(f"Executing final query: {final_query}")
            result_relation = self.db_conn.execute(final_query)
            result_data = result_relation.fetchall()
            headers = [desc[0] for desc in result_relation.description] if result_relation.description else []

            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(len(headers))
            self.results_table.setHorizontalHeaderLabels(headers)

            for row_idx, row_data_tuple in enumerate(result_data):
                self.results_table.insertRow(row_idx)
                for col_idx, cell_data_item in enumerate(row_data_tuple):
                    self.results_table.setItem(row_idx, col_idx, QTableWidgetItem(str(cell_data_item)))
            
            self.results_table.resizeColumnsToContents()
            self.status_label.setText(f"Query successful. {len(result_data)} rows returned.")

        except Exception as e:
            QMessageBox.critical(self, "Query Error", f"Error executing query: {e}")
            self.status_label.setText(f"Query failed.")
            import traceback
            traceback.print_exc()
        finally:
            if temp_view_created_for_query:
                try:
                    self.db_conn.unregister(fixed_alias)
                    print(f"Temporary view '{fixed_alias}' unregistered after query.")
                except Exception as e_unreg_temp:
                    print(f"Error unregistering temporary view '{fixed_alias}': {e_unreg_temp}")

    def _get_table_data_as_df(self):
        if self.results_table.rowCount() == 0:
            return pd.DataFrame()

        headers = []
        for col in range(self.results_table.columnCount()):
            header_item = self.results_table.horizontalHeaderItem(col)
            headers.append(header_item.text() if header_item else f"Column_{col+1}")
        
        all_row_data = []
        for row_idx in range(self.results_table.rowCount()):
            row_data = []
            for col_idx in range(self.results_table.columnCount()):
                item = self.results_table.item(row_idx, col_idx)
                row_data.append(item.text() if item else "")
            all_row_data.append(row_data)
        
        return pd.DataFrame(all_row_data, columns=headers)

    def _download_results_as_csv(self):
        if self.results_table.rowCount() == 0:
            QMessageBox.information(self, "No Data", "There is no data in the table to download.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save as CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return

        try:
            df = self._get_table_data_as_df()
            df.to_csv(file_path, index=False, encoding='utf-8')
            QMessageBox.information(self, "Success", f"Results saved to {file_path}")
            self.status_label.setText(f"Results saved as CSV: {os.path.basename(file_path)}")
        except Exception as e:
            QMessageBox.critical(self, "Error Saving CSV", f"Could not save results to CSV.\\n{e}")
            self.status_label.setText("Error saving CSV.")

    def _download_results_as_excel(self):
        if self.results_table.rowCount() == 0:
            QMessageBox.information(self, "No Data", "There is no data in the table to download.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save as Excel", "", "Excel Files (*.xlsx);;All Files (*)"
        )
        if not file_path:
            return
        
        # Ensure .xlsx extension
        if not file_path.lower().endswith('.xlsx'):
            file_path += '.xlsx'

        try:
            df = self._get_table_data_as_df()
            df.to_excel(file_path, index=False, engine='openpyxl')
            QMessageBox.information(self, "Success", f"Results saved to {file_path}")
            self.status_label.setText(f"Results saved as Excel: {os.path.basename(file_path)}")
        except Exception as e:
            QMessageBox.critical(self, "Error Saving Excel", f"Could not save results to Excel.\\n{e}")
            self.status_label.setText("Error saving Excel.")

    def _show_results_table_context_menu(self, position):
        menu = QMenu()
        copy_cell_action = menu.addAction("Copy Cell Value")
        copy_row_action = menu.addAction("Copy Row")
        copy_column_action = menu.addAction("Copy Column (with Header)")
        copy_table_action = menu.addAction("Copy Table (with Headers)")

        action = menu.exec(self.results_table.mapToGlobal(position))

        if action == copy_cell_action:
            self._copy_cell_value()
        elif action == copy_row_action:
            self._copy_row_content()
        elif action == copy_column_action:
            self._copy_column_content_with_header()
        elif action == copy_table_action:
            self._copy_table_content_with_headers()

    def _copy_cell_value(self):
        selected_items = self.results_table.selectedItems()
        if selected_items:
            clipboard = QApplication.clipboard()
            clipboard.setText(selected_items[0].text())
            self.status_label.setText("Cell value copied.")
        else:
            self.status_label.setText("No cell selected to copy.")

    def _copy_row_content(self):
        selected_indexes = self.results_table.selectedIndexes()
        if not selected_indexes:
            self.status_label.setText("No row selected to copy.")
            return

        selected_row = selected_indexes[0].row()
        row_data = []
        for col in range(self.results_table.columnCount()):
            item = self.results_table.item(selected_row, col)
            row_data.append(item.text() if item else "")
        
        clipboard = QApplication.clipboard()
        clipboard.setText("\t".join(row_data))
        self.status_label.setText(f"Row {selected_row + 1} copied.")

    def _copy_column_content_with_header(self):
        selected_indexes = self.results_table.selectedIndexes()
        if not selected_indexes:
            self.status_label.setText("No column selected to copy.")
            return

        selected_col = selected_indexes[0].column()
        header_item = self.results_table.horizontalHeaderItem(selected_col)
        header_text = header_item.text() if header_item else f"Column {selected_col + 1}"
        
        col_data = [header_text]
        for row in range(self.results_table.rowCount()):
            item = self.results_table.item(row, selected_col)
            col_data.append(item.text() if item else "")
            
        clipboard = QApplication.clipboard()
        clipboard.setText("\n".join(col_data))
        self.status_label.setText(f"Column '{header_text}' copied.")

    def _copy_table_content_with_headers(self):
        if self.results_table.rowCount() == 0:
            self.status_label.setText("Table is empty, nothing to copy.")
            return

        headers = []
        for col in range(self.results_table.columnCount()):
            header_item = self.results_table.horizontalHeaderItem(col)
            headers.append(header_item.text() if header_item else f"Column {col + 1}")
        
        table_data_str = ["\t".join(headers)]
        for row in range(self.results_table.rowCount()):
            row_data = []
            for col in range(self.results_table.columnCount()):
                item = self.results_table.item(row, col)
                row_data.append(item.text() if item else "")
            table_data_str.append("\t".join(row_data))
            
        clipboard = QApplication.clipboard()
        clipboard.setText("\n".join(table_data_str))
        self.status_label.setText("Table content with headers copied.")

    def closeEvent(self, event):
        # Clean up DuckDB connection when dialog closes
        if self.db_conn:
            try:
                # If an Excel sheet was actively registered as "Data", unregister it.
                if hasattr(self, 'currently_registered_excel_sheet_name') and self.currently_registered_excel_sheet_name:
                    try:
                        self.db_conn.unregister("Data") # Target alias is the string "Data"
                        # Correctly use the literal string 'Data' in the informational print message
                        print(f"QueryFileDialog: Unregistered 'Data' (Excel sheet: {self.currently_registered_excel_sheet_name}).")
                    except Exception as e_unreg:
                        print(f"QueryFileDialog: Error unregistering 'Data': {e_unreg}")
                
                self.db_conn.close()
                print("QueryFileDialog: In-memory DB connection closed.")
            except Exception as e:
                print(f"QueryFileDialog: Error closing DB connection: {e}")
        super().closeEvent(event)

if __name__ == '__main__':
    # This is for testing the dialog independently
    app = QApplication(sys.argv)
    dialog = QueryFileDialog()
    dialog.show()
    sys.exit(app.exec()) 