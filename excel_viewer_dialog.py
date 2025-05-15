from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, 
    QPushButton, QLabel, QLineEdit, QMessageBox, QApplication, QFileDialog
)
from PyQt6.QtCore import Qt
import pandas as pd
import os # Import the os module
import json

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

        # Table Widget for displaying data
        self.table_widget = QTableWidget()
        self.table_widget.setEditTriggers(QTableWidget.EditTrigger.DoubleClicked) # Allow editing on double click
        self.layout.addWidget(self.table_widget)

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