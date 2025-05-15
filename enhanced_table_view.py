from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QTableWidget, QTableWidgetItem, QLabel, QApplication
)
from PyQt6.QtCore import Qt
import re

class EnhancedTableViewDialog(QDialog):
    def __init__(self, data, headers, parent=None):
        super().__init__(parent)
        self.data = data
        self.headers = headers
        self.setWindowTitle("Enhanced Table View")
        self.setMinimumSize(800, 600)
        # Add window flags to enable minimize and maximize buttons
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint)

        self.layout = QVBoxLayout(self)

        # Filter input
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter (e.g., column_name:value AND another_col:text):"))
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("name:John AND age:>30")
        self.filter_input.textChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.filter_input)
        self.layout.addLayout(filter_layout)

        # Results table
        self.table_widget = QTableWidget()
        self.table_widget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.layout.addWidget(self.table_widget)

        self.populate_table()

    def populate_table(self):
        self.table_widget.setRowCount(len(self.data))
        self.table_widget.setColumnCount(len(self.headers))
        self.table_widget.setHorizontalHeaderLabels(self.headers)

        for row_idx, row_data in enumerate(self.data):
            for col_idx, cell_data in enumerate(row_data):
                item = QTableWidgetItem(str(cell_data) if cell_data is not None else "NULL")
                self.table_widget.setItem(row_idx, col_idx, item)
        
        self.table_widget.resizeColumnsToContents()

    def apply_filters(self):
        filter_text = self.filter_input.text().strip()
        if not filter_text:
            for i in range(self.table_widget.rowCount()):
                self.table_widget.setRowHidden(i, False)
            return

        # Simple AND-based colon filter parser
        # Example: "name:Alice AND age:>30"
        # More complex parsing (OR, parentheses) would require a more robust parser
        active_filters = []
        try:
            filter_parts = filter_text.split(" AND ")
            for part in filter_parts:
                if ':' not in part:
                    # If a part of the filter doesn't have a colon, it could be a global search term
                    # For now, we'll treat malformed parts as invalid, or you could implement global search
                    # For simplicity, let's assume valid colon filters or skip malformed ones
                    continue 
                
                col_name, value = part.split(':', 1)
                col_name = col_name.strip()
                value = value.strip().lower() # Case-insensitive value matching
                
                if not col_name or not value:
                    continue

                try:
                    col_idx = self.headers.index(col_name)
                    active_filters.append({'col_idx': col_idx, 'value': value})
                except ValueError: # Column name not found in headers
                    # Handle error: maybe show a message to the user or ignore invalid column name
                    print(f"Warning: Column '{col_name}' not found in headers.")
                    # To make it stricter, we can hide all rows if a filter is invalid
                    for i in range(self.table_widget.rowCount()):
                        self.table_widget.setRowHidden(i, True)
                    return
        except Exception as e:
            print(f"Error parsing filter: {e}")
            # Optionally, indicate an error in the UI, or hide all rows
            for i in range(self.table_widget.rowCount()):
                self.table_widget.setRowHidden(i, True)
            return

        if not active_filters and filter_text: # If filter text exists but no valid filters parsed
            for i in range(self.table_widget.rowCount()):
                 self.table_widget.setRowHidden(i, True) # Hide all if filter is present but invalid
            return

        for row_idx in range(self.table_widget.rowCount()):
            row_visible = True
            for f in active_filters:
                item = self.table_widget.item(row_idx, f['col_idx'])
                if not item or f['value'] not in item.text().lower():
                    row_visible = False
                    break
            self.table_widget.setRowHidden(row_idx, not row_visible)

if __name__ == '__main__':
    # Example Usage (for testing this dialog independently)
    app = QApplication([])
    example_headers = ["ID", "Name", "Age", "City"]
    example_data = [
        (1, "Alice", 30, "New York"),
        (2, "Bob", 24, "Los Angeles"),
        (3, "Charlie", 35, "New York"),
        (4, "David", 29, "Chicago"),
        (5, "Eve", 22, "Los Angeles")
    ]
    dialog = EnhancedTableViewDialog(data=example_data, headers=example_headers)
    dialog.show()
    app.exec() 