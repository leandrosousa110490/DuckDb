from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, 
    QPushButton, QLabel, QLineEdit, QMessageBox, QApplication, QFileDialog,
    QMenu
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
import pandas as pd
import os # Import the os module
import json
import datetime # Add datetime import

class ExcelViewDialog(QDialog):
    def __init__(self, file_path, sheet_name, annotations_to_load=None, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.annotations_to_load = annotations_to_load # Store annotations
        self.df = None  # To store the loaded pandas DataFrame

        self.setWindowTitle(f"Excel Viewer: {self.sheet_name} - {os.path.basename(self.file_path)}")
        self.setMinimumSize(900, 700)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint)

        self.layout = QVBoxLayout(self)

        # Filter input
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Type to filter table content...")
        self.filter_input.textChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.filter_input)
        self.layout.addLayout(filter_layout)

        # Table Widget for displaying data
        self.table_widget = QTableWidget()
        self.table_widget.setEditTriggers(QTableWidget.EditTrigger.DoubleClicked) # Allow editing on double click
        self.table_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu) # Enable context menu
        self.table_widget.customContextMenuRequested.connect(self.show_table_context_menu) # Connect slot
        # Style selected cells to have a light blue background
        self.table_widget.setStyleSheet("QTableWidget::item:selected { background-color: #DCEBFF; color: black; }")
        self.layout.addWidget(self.table_widget)

        # Download buttons for ExcelViewDialog
        view_download_layout = QHBoxLayout()
        self.download_view_csv_button = QPushButton("Download View as CSV")
        self.download_view_csv_button.clicked.connect(self._download_view_as_csv)
        view_download_layout.addWidget(self.download_view_csv_button)

        self.download_view_excel_button = QPushButton("Download View as Excel")
        self.download_view_excel_button.clicked.connect(self._download_view_as_excel)
        view_download_layout.addWidget(self.download_view_excel_button)
        self.layout.addLayout(view_download_layout) # Add this before JSON save layout

        # JSON Save Area
        save_layout = QHBoxLayout()
        save_layout.addWidget(QLabel("Save to JSON File:"))
        self.json_file_path_input = QLineEdit()
        
        safe_sheet_name = "".join(c if c.isalnum() else '_' for c in self.sheet_name)
        base_name = os.path.basename(self.file_path)
        safe_file_name = "".join(c if c.isalnum() else '_' for c in base_name.split('.')[0])
        # Suggest a .json file in the same directory as the Excel file initially
        suggested_json_path = os.path.join(os.path.dirname(self.file_path), f"annotated_{safe_file_name}_{safe_sheet_name}.json")
        self.json_file_path_input.setText(suggested_json_path)
        save_layout.addWidget(self.json_file_path_input)
        
        self.save_to_json_button = QPushButton("Save to JSON File")
        self.save_to_json_button.clicked.connect(self.save_to_json_file)
        save_layout.addWidget(self.save_to_json_button)
        self.layout.addLayout(save_layout)

        self.load_and_display_data()

    def load_and_display_data(self):
        try:
            self.df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Excel", f"Could not load sheet '{self.sheet_name}' from file.\n{e}")
            self.close()
            return

        if self.df.empty:
            QMessageBox.information(self, "Info", "The selected sheet is empty.")
            # Still show an empty table if desired, or close
            # self.close()
            # return

        # +2 for the checkbox and comment columns
        self.table_widget.setColumnCount(len(self.df.columns) + 2) 
        self.table_widget.setRowCount(len(self.df))

        headers = ["Select", "_Comment"] + self.df.columns.tolist()
        self.table_widget.setHorizontalHeaderLabels(headers)

        for row_idx, row_data in self.df.iterrows():
            # Checkbox item
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            
            initial_check_state = Qt.CheckState.Unchecked
            initial_comment_text = "" # Default empty comment

            if self.annotations_to_load and row_idx < len(self.annotations_to_load):
                annotation_for_row = self.annotations_to_load[row_idx]
                if isinstance(annotation_for_row, dict):
                    if "_IsSelected" in annotation_for_row and annotation_for_row["_IsSelected"] == True:
                        initial_check_state = Qt.CheckState.Checked
                    if "_Comment" in annotation_for_row: 
                        initial_comment_text = str(annotation_for_row["_Comment"])
            
            chk_item.setCheckState(initial_check_state)
            self.table_widget.setItem(row_idx, 0, chk_item)

            # Comment item
            comment_item = QTableWidgetItem(initial_comment_text)
            comment_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable)
            self.table_widget.setItem(row_idx, 1, comment_item)

            # Data items
            for col_idx, cell_data in enumerate(row_data):
                item = QTableWidgetItem(str(cell_data) if pd.notna(cell_data) else "")
                item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
                self.table_widget.setItem(row_idx, col_idx + 2, item) # +2 due to checkbox and comment cols
        
        self.table_widget.resizeColumnsToContents()
        self.table_widget.horizontalHeader().setStretchLastSection(True)

    def _apply_filter(self):
        filter_text = self.filter_input.text().lower().strip()

        for row_idx in range(self.table_widget.rowCount()):
            if not filter_text: # If filter is empty, show all rows
                self.table_widget.setRowHidden(row_idx, False)
                continue

            row_matches = False
            # Iterate from column 1 (skip "Select" checkbox column)
            # Column 1 is _Comment, columns 2 onwards are data from self.df
            for col_idx in range(1, self.table_widget.columnCount()): 
                item = self.table_widget.item(row_idx, col_idx)
                if item and item.text().lower().strip().__contains__(filter_text):
                    row_matches = True
                    break
            
            self.table_widget.setRowHidden(row_idx, not row_matches)

    def show_table_context_menu(self, pos):
        item = self.table_widget.itemAt(pos)
        menu = QMenu(self)
        
        if item:
            copy_cell_action = QAction("Copy Cell Value", self)
            copy_cell_action.triggered.connect(lambda: self.copy_cell_value(item))
            menu.addAction(copy_cell_action)

            # Note: row and column are 0-indexed from the QTableWidgetItem
            row = item.row()
            column = item.column()

            copy_row_action = QAction(f"Copy Row {row + 1}", self)
            copy_row_action.triggered.connect(lambda: self.copy_row(self.table_widget, row))
            menu.addAction(copy_row_action)

            # For ExcelView, column 0 is 'Select', column 1 is '_Comment'. Data starts at 2 for headers from df.
            # The actual header text for data columns is self.df.columns[column - 2] if column >=2
            # Header for column 0 is "Select", for 1 is "_Comment"
            header_text = ""
            if self.table_widget.horizontalHeaderItem(column):
                 header_text = self.table_widget.horizontalHeaderItem(column).text()
            else:
                 header_text = f"Column_{column+1}" # Fallback
            
            copy_column_action = QAction(f"Copy Column '{header_text}'", self)
            copy_column_action.triggered.connect(lambda: self.copy_column(self.table_widget, column))
            menu.addAction(copy_column_action)
        
        menu.addSeparator()

        copy_headers_action = QAction("Copy Headers Only", self)
        copy_headers_action.triggered.connect(lambda: self.copy_headers(self.table_widget))
        menu.addAction(copy_headers_action)

        copy_table_action = QAction("Copy Entire Table (with Headers)", self)
        copy_table_action.triggered.connect(lambda: self.copy_table_contents(self.table_widget))
        menu.addAction(copy_table_action)
            
        menu.exec(self.table_widget.mapToGlobal(pos))

    def copy_cell_value(self, item):
        if item:
            QApplication.clipboard().setText(item.text())

    def copy_row(self, table_widget, row):
        if not table_widget or row < 0 or row >= table_widget.rowCount():
            return
        row_data = []
        for col_idx in range(table_widget.columnCount()):
            item = table_widget.item(row, col_idx)
            # For checkbox column, copy its checked state as True/False string
            if col_idx == 0 and item and item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                row_data.append(str(item.checkState() == Qt.CheckState.Checked))
            else:
                row_data.append(item.text() if item else "")
        QApplication.clipboard().setText("\t".join(row_data))

    def copy_column(self, table_widget, column):
        if not table_widget or column < 0 or column >= table_widget.columnCount():
            return
        column_data = []
        header_item = table_widget.horizontalHeaderItem(column)
        header_text = header_item.text() if header_item else f"Column_{column + 1}"
        column_data.append(header_text)

        for row_idx in range(table_widget.rowCount()):
            item = table_widget.item(row_idx, column)
            if column == 0 and item and item.flags() & Qt.ItemFlag.ItemIsUserCheckable: # Checkbox column
                column_data.append(str(item.checkState() == Qt.CheckState.Checked))
            else:
                column_data.append(item.text() if item else "")
        QApplication.clipboard().setText("\n".join(column_data))

    def copy_headers(self, table_widget):
        if not table_widget or table_widget.columnCount() == 0:
            return
        headers_list = []
        for col_idx in range(table_widget.columnCount()):
            header_item = table_widget.horizontalHeaderItem(col_idx)
            headers_list.append(header_item.text() if header_item else f"Column_{col_idx + 1}")
        QApplication.clipboard().setText("\t".join(headers_list))

    def copy_table_contents(self, table_widget):
        if not table_widget or table_widget.rowCount() == 0 or table_widget.columnCount() == 0:
            return
        
        content_lines = []
        headers_list = []
        for col_idx in range(table_widget.columnCount()):
            header_item = table_widget.horizontalHeaderItem(col_idx)
            headers_list.append(header_item.text() if header_item else f"Column_{col_idx + 1}")
        content_lines.append("\t".join(headers_list))

        for row_idx in range(table_widget.rowCount()):
            row_data = []
            for col_idx in range(table_widget.columnCount()):
                item = table_widget.item(row_idx, col_idx)
                if col_idx == 0: # Specifically handle the 'Select' checkbox column (column index 0)
                    if item: # Check if item exists
                        # Directly use checkState, convert to "True" or "False" string
                        is_checked = (item.checkState() == Qt.CheckState.Checked)
                        row_data.append(str(is_checked))
                    else:
                        row_data.append("False") # Default to "False" if item is unexpectedly None for checkbox col
                else: # For other columns (Comment column and data columns)
                    row_data.append(item.text() if item else "")
            content_lines.append("\t".join(row_data))
        
        QApplication.clipboard().setText("\n".join(content_lines))

    def _get_view_table_data_as_df(self, include_select_col_as_bool=False):
        """Helper to get current table_widget data as a DataFrame.
           Handles visible rows only if filter is active.
           Converts 'Select' column to boolean if requested.
        """
        if self.table_widget.rowCount() == 0:
            return pd.DataFrame()

        headers = []
        for col_idx in range(self.table_widget.columnCount()):
            header_item = self.table_widget.horizontalHeaderItem(col_idx)
            headers.append(header_item.text() if header_item else f"Column_{col_idx + 1}")
        
        all_row_data = []
        for row_idx in range(self.table_widget.rowCount()):
            if self.table_widget.isRowHidden(row_idx): # Export only visible rows if filter is active
                continue
            row_data = []
            for col_idx in range(self.table_widget.columnCount()):
                item = self.table_widget.item(row_idx, col_idx)
                if col_idx == 0: # 'Select' checkbox column
                    is_checked = False # Default
                    if item:
                        is_checked = (item.checkState() == Qt.CheckState.Checked)
                    row_data.append(is_checked if include_select_col_as_bool else str(is_checked))
                else:
                    row_data.append(item.text() if item else "")
            all_row_data.append(row_data)
        
        return pd.DataFrame(all_row_data, columns=headers)

    def _download_view_as_csv(self):
        if self.table_widget.rowCount() == 0:
            QMessageBox.information(self, "No Data", "There is no data in the table to download.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save View as CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return

        try:
            # For CSV, we want the "Select" column as "True"/"False" strings
            df_to_save = self._get_view_table_data_as_df(include_select_col_as_bool=False) 
            if df_to_save.empty:
                 QMessageBox.information(self, "No Visible Data", "No visible data to download (check filter)." )
                 return
            df_to_save.to_csv(file_path, index=False, encoding='utf-8')
            QMessageBox.information(self, "Success", f"Table view saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error Saving CSV", f"Could not save table view to CSV.\\n{e}")

    def _download_view_as_excel(self):
        if self.table_widget.rowCount() == 0:
            QMessageBox.information(self, "No Data", "There is no data in the table to download.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save View as Excel", "", "Excel Files (*.xlsx);;All Files (*)"
        )
        if not file_path:
            return
        
        if not file_path.lower().endswith('.xlsx'):
            file_path += '.xlsx'

        try:
            # For Excel, it's better to have actual boolean True/False for the 'Select' column
            df_to_save = self._get_view_table_data_as_df(include_select_col_as_bool=True)
            if df_to_save.empty:
                 QMessageBox.information(self, "No Visible Data", "No visible data to download (check filter)." )
                 return
            df_to_save.to_excel(file_path, index=False, engine='openpyxl')
            QMessageBox.information(self, "Success", f"Table view saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error Saving Excel", f"Could not save table view to Excel.\\n{e}")

    def save_to_json_file(self): 
        if self.df is None:
            QMessageBox.warning(self, "No Data", "No data loaded to save.")
            return

        current_path_suggestion = self.json_file_path_input.text().strip()

        # Use QFileDialog to let user confirm/change the save path
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Annotated Data as JSON", 
            current_path_suggestion, # Default path from the QLineEdit
            "JSON Files (*.json);;All Files (*)"
        )

        if not file_path:
            return # User cancelled
        
        # Ensure it has a .json extension
        if not file_path.lower().endswith('.json'):
            file_path += '.json'
        
        # Update the line edit with the chosen path (optional)
        self.json_file_path_input.setText(file_path)

        # Create a new DataFrame with the checkbox states
        data_to_save = self.df.copy()
        checkbox_states = []
        comment_texts = [] # New list for comments
        for row_idx in range(self.table_widget.rowCount()):
            chk_item = self.table_widget.item(row_idx, 0)
            checkbox_states.append(chk_item.checkState() == Qt.CheckState.Checked)
            
            comment_item = self.table_widget.item(row_idx, 1) # Comment is in column 1
            comment_texts.append(comment_item.text() if comment_item else "") # Get text, default to empty
        
        # Prepend the selection and comment columns. Order matters for to_dict('records') if we want consistency.
        # However, since it becomes a list of dicts, the key is what matters.
        # Let's add _IsSelected first, then _Comment, then original data.
        temp_df_for_export = self.df.copy() # Start with original data
        temp_df_for_export.insert(0, "_IsSelected", checkbox_states)
        temp_df_for_export.insert(1, "_Comment", comment_texts)

        # Convert datetime columns to ISO format strings before saving to JSON
        for col in temp_df_for_export.columns:
            if pd.api.types.is_datetime64_any_dtype(temp_df_for_export[col].dtype):
                # Convert to ISO format, NaT becomes None (null in JSON)
                temp_df_for_export[col] = temp_df_for_export[col].apply(lambda x: x.isoformat() if pd.notnull(x) else None)
            elif temp_df_for_export[col].dtype == 'object':
                # For object columns, check if elements are datetime instances
                # This handles cases where a column might be mixed type but contains datetimes
                def convert_if_datetime(x):
                    if isinstance(x, (datetime.datetime, datetime.date, pd.Timestamp)):
                        return x.isoformat() if pd.notnull(x) else None
                    return x
                temp_df_for_export[col] = temp_df_for_export[col].apply(convert_if_datetime)

        try:
            # Check if file exists and ask for overwrite confirmation
            if os.path.exists(file_path):
                reply = QMessageBox.question(self, "Confirm Overwrite", 
                                             f"File '{os.path.basename(file_path)}' already exists. Overwrite it?",
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                             QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.No:
                    return # User chose not to overwrite
            
            # Prepare the final JSON structure with metadata
            output_json_structure = {
                "original_excel_file_path": self.file_path,
                "original_sheet_name": self.sheet_name,
                "annotated_data": temp_df_for_export.to_dict(orient='records') # Use the new df with comments
            }
            
            # Convert the whole structure to JSON string
            json_string_output = json.dumps(output_json_structure, indent=4)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_string_output)

            QMessageBox.information(self, "Success", 
                                   f"Annotated data saved to JSON file:\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "JSON Save Error", f"Could not save data to JSON file.\n{e}")

if __name__ == '__main__':
    app = QApplication([])
    # def dummy_db_conn_func(): # No longer needed for this dialog test
    #     import duckdb
    #     try:
    #         return duckdb.connect(database=':memory:', read_only=False)
    #     except Exception as e:
    #         print(f"Dummy DB Error: {e}")
    #         return None

    dummy_excel_file = "dummy_excel_for_viewer.xlsx"
    writer = pd.ExcelWriter(dummy_excel_file, engine='openpyxl')
    pd.DataFrame({'ColA': [1, 2, 3], 'ColB': ['X', 'Y', 'Z']}).to_excel(writer, sheet_name='Sheet1', index=False)
    pd.DataFrame({'Data1': [10.1, 20.2], 'Info': ['Test1', 'Test2']}).to_excel(writer, sheet_name='Another Sheet', index=False)
    writer.close() 

    dialog = ExcelViewDialog(dummy_excel_file, 'Sheet1') # Removed dummy_db_conn_func
    dialog.show()
    app.exec() 