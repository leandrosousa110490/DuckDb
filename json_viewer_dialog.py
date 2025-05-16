from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTextEdit, QPushButton, QFileDialog, QMessageBox, QApplication, 
    QTabWidget, QWidget, QTableWidget, QTableWidgetItem,
    QMenu
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
import json
import os

class JsonViewDialog(QDialog):
    def __init__(self, file_path=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("JSON Viewer")
        self.setMinimumSize(800, 600) # Increased size a bit
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint)

        self.layout = QVBoxLayout(self)

        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)

        # Tab 1: Raw JSON View
        self.raw_json_tab = QWidget()
        self.raw_json_layout = QVBoxLayout(self.raw_json_tab)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap) 
        self.raw_json_layout.addWidget(self.text_edit)
        self.tab_widget.addTab(self.raw_json_tab, "Raw JSON")

        # Tab 2: Table View
        self.table_view_tab = QWidget()
        self.table_view_layout = QVBoxLayout(self.table_view_tab)
        self.table_widget = QTableWidget()
        self.table_widget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table_widget.customContextMenuRequested.connect(self.show_table_context_menu)
        self.table_view_layout.addWidget(self.table_widget)
        self.tab_widget.addTab(self.table_view_tab, "Table View")

        if file_path:
            self.load_file(file_path)

    def load_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 1. Populate Raw JSON View
                pretty_json = json.dumps(data, indent=4)
                self.text_edit.setPlainText(pretty_json)
                self.setWindowTitle(f"JSON Viewer - {os.path.basename(file_path)}")

                # 2. Attempt to Populate Table View
                self.populate_table_view(data)

        except Exception as e:
            QMessageBox.critical(self, "Error Loading JSON", f"Could not load or parse JSON file.\n{e}")
            self.text_edit.setPlainText(f"Error loading file: {e}")
            self.table_widget.setRowCount(0)
            self.table_widget.setColumnCount(0)

    def populate_table_view(self, json_data):
        self.table_widget.setRowCount(0)
        self.table_widget.setColumnCount(0)

        data_to_display_in_table = None

        if isinstance(json_data, dict) and \
           "original_excel_file_path" in json_data and \
           "original_sheet_name" in json_data and \
           "annotated_data" in json_data and \
           isinstance(json_data["annotated_data"], list):
            data_to_display_in_table = json_data["annotated_data"]
        elif isinstance(json_data, list):
            data_to_display_in_table = json_data
        
        if data_to_display_in_table and isinstance(data_to_display_in_table, list) and len(data_to_display_in_table) > 0:
            # Ensure all items are dictionaries (or can be treated as such for table)
            # If not, we simply won't populate the table for those non-dict items.
            # We will still try to process rows that are dicts.
            
            # Aggregate all unique keys from all dictionary items to form headers
            all_keys = set()
            for item in data_to_display_in_table:
                if isinstance(item, dict):
                    all_keys.update(item.keys())
            
            if not all_keys:
                print("No dictionary keys found in the list of items to display in table.")
                return

            headers = sorted(list(all_keys)) # Sort for consistent column order
            self.table_widget.setColumnCount(len(headers))
            self.table_widget.setHorizontalHeaderLabels(headers)
            
            # Filter data to only include rows that are dictionaries
            dict_rows = [row for row in data_to_display_in_table if isinstance(row, dict)]
            self.table_widget.setRowCount(len(dict_rows))

            for row_idx, row_object in enumerate(dict_rows):
                for col_idx, header in enumerate(headers):
                    cell_value = row_object.get(header) 
                    item = QTableWidgetItem(str(cell_value) if cell_value is not None else "")
                    self.table_widget.setItem(row_idx, col_idx, item)
            
            self.table_widget.resizeColumnsToContents()
        else:
            print("JSON data is not in a format suitable for direct table view (expected list of objects, or specific annotation format).")

    def show_table_context_menu(self, pos):
        if self.sender() != self.table_widget:
            return
            
        item = self.table_widget.itemAt(pos)
        menu = QMenu(self)
        
        if item:
            copy_cell_action = QAction("Copy Cell Value", self)
            copy_cell_action.triggered.connect(lambda: self.copy_cell_value(item))
            menu.addAction(copy_cell_action)

            row = item.row()
            column = item.column()

            copy_row_action = QAction(f"Copy Row {row + 1}", self)
            copy_row_action.triggered.connect(lambda: self.copy_row(self.table_widget, row))
            menu.addAction(copy_row_action)

            header_text = self.table_widget.horizontalHeaderItem(column).text() if self.table_widget.horizontalHeaderItem(column) else f"Column_{column+1}"
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
                row_data.append(item.text() if item else "")
            content_lines.append("\t".join(row_data))
        
        QApplication.clipboard().setText("\n".join(content_lines))

if __name__ == '__main__':
    app = QApplication([])
    
    # Create dummy JSON files for testing
    dummy_json_list_file = "dummy_list_data.json"
    dummy_list_content = [{"id": 1, "name": "Alice", "age": 30}, {"id": 2, "name": "Bob", "age": 24, "city": "NY"}]
    with open(dummy_json_list_file, 'w') as f: json.dump(dummy_list_content, f, indent=4)

    dummy_annot_file = "dummy_annot_data.json"
    dummy_annot_content = {
        "original_excel_file_path": "some/path/file.xlsx",
        "original_sheet_name": "Sheet1",
        "annotated_data": [
            {"_IsSelected": True, "ColA": 1, "ColB": "X"},
            {"_IsSelected": False, "ColA": 2, "ColB": "Y"}
        ]
    }
    with open(dummy_annot_file, 'w') as f: json.dump(dummy_annot_content, f, indent=4)

    dummy_single_obj_file = "dummy_single_obj.json"
    dummy_single_obj_content = {"message": "Hello", "status": "OK"}
    with open(dummy_single_obj_file, 'w') as f: json.dump(dummy_single_obj_content, f, indent=4)

    # Test 1: Generic list of objects
    dialog_list = JsonViewDialog(file_path=dummy_json_list_file)
    dialog_list.setWindowTitle("JSON Viewer - List of Objects")
    dialog_list.show()

    # Test 2: Our specific annotation format
    dialog_annot = JsonViewDialog(file_path=dummy_annot_file)
    dialog_annot.setWindowTitle("JSON Viewer - Annotation File")
    dialog_annot.show()

    # Test 3: Single JSON object (should not populate table well)
    dialog_single = JsonViewDialog(file_path=dummy_single_obj_file)
    dialog_single.setWindowTitle("JSON Viewer - Single Object")
    dialog_single.show()

    app.exec()
    
    # Clean up dummy files
    for f_path in [dummy_json_list_file, dummy_annot_file, dummy_single_obj_file]:
        if os.path.exists(f_path):
            os.remove(f_path) 