from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QTableWidget, QTableWidgetItem, QLabel, QApplication, QHeaderView,
    QMenu, QPushButton, QFileDialog, QMessageBox
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt
import re
import pandas as pd
import os

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
        filter_layout.addWidget(QLabel("Filter (e.g., Name:CONTAINS:John AND Age:>:30 AND City:ISNULL):"))
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("ColName:OPERATOR:Value AND ... (Operators: >, <, >=, <=, =, !=, CONTAINS, NOTCONTAINS, STARTSWITH, ENDSWITH, ISNULL, ISNOTNULL)")
        self.filter_input.textChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.filter_input)
        self.layout.addLayout(filter_layout)

        # Results table
        self.table_widget = QTableWidget()
        self.table_widget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table_widget.customContextMenuRequested.connect(self.show_table_context_menu)
        self.layout.addWidget(self.table_widget)

        # Download buttons
        download_layout = QHBoxLayout()
        self.download_csv_button = QPushButton("Download View as CSV")
        self.download_csv_button.clicked.connect(self._download_current_view_as_csv)
        download_layout.addWidget(self.download_csv_button)

        self.download_excel_button = QPushButton("Download View as Excel")
        self.download_excel_button.clicked.connect(self._download_current_view_as_excel)
        download_layout.addWidget(self.download_excel_button)
        self.layout.addLayout(download_layout)

        self.populate_table()

    def populate_table(self):
        self.table_widget.setRowCount(0) # Clear before populating
        self.table_widget.setRowCount(len(self.data))
        self.table_widget.setColumnCount(len(self.headers))
        self.table_widget.setHorizontalHeaderLabels(self.headers)

        for row_idx, row_data in enumerate(self.data):
            for col_idx, cell_data in enumerate(row_data):
                item = QTableWidgetItem(str(cell_data) if cell_data is not None else "NULL")
                self.table_widget.setItem(row_idx, col_idx, item)
        
        self.table_widget.resizeColumnsToContents()
        self.table_widget.horizontalHeader().setStretchLastSection(True)
        self.apply_filters() # Apply initial filters if any

    def show_table_context_menu(self, pos):
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
            
        if menu.isEmpty(): # Should not be empty if we reach here due to separator and general actions
            return
            
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

    def _get_current_view_data_as_df(self):
        """Helper to get current table_widget data (visible rows) as a DataFrame."""
        if self.table_widget.rowCount() == 0:
            return pd.DataFrame()

        headers = []
        for col_idx in range(self.table_widget.columnCount()):
            header_item = self.table_widget.horizontalHeaderItem(col_idx)
            headers.append(header_item.text() if header_item else f"Column_{col_idx + 1}")
        
        all_row_data = []
        for row_idx in range(self.table_widget.rowCount()):
            if self.table_widget.isRowHidden(row_idx):
                continue # Skip hidden rows
            
            row_data = []
            for col_idx in range(self.table_widget.columnCount()):
                item = self.table_widget.item(row_idx, col_idx)
                row_data.append(item.text() if item else "")
            all_row_data.append(row_data)
        
        if not all_row_data: # If all rows were hidden
            return pd.DataFrame(columns=headers) 
            
        return pd.DataFrame(all_row_data, columns=headers)

    def _download_current_view_as_csv(self):
        df_to_save = self._get_current_view_data_as_df()
        if df_to_save.empty and self.table_widget.rowCount() > 0: # Table has data, but all are filtered out
            QMessageBox.information(self, "No Visible Data", "No data matches the current filter criteria to download.")
            return
        elif df_to_save.empty:
            QMessageBox.information(self, "No Data", "There is no data in the table to download.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Current View as CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return

        try:
            df_to_save.to_csv(file_path, index=False, encoding='utf-8')
            QMessageBox.information(self, "Success", f"Current view saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error Saving CSV", f"Could not save current view to CSV.\\n{e}")

    def _download_current_view_as_excel(self):
        df_to_save = self._get_current_view_data_as_df()
        if df_to_save.empty and self.table_widget.rowCount() > 0: # Table has data, but all are filtered out
            QMessageBox.information(self, "No Visible Data", "No data matches the current filter criteria to download.")
            return
        elif df_to_save.empty:
            QMessageBox.information(self, "No Data", "There is no data in the table to download.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Current View as Excel", "", "Excel Files (*.xlsx);;All Files (*)"
        )
        if not file_path:
            return
        
        if not file_path.lower().endswith('.xlsx'):
            file_path += '.xlsx'

        try:
            df_to_save.to_excel(file_path, index=False, engine='openpyxl')
            QMessageBox.information(self, "Success", f"Current view saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error Saving Excel", f"Could not save current view to Excel.\\n{e}")

    def _evaluate_condition(self, cell_value_str, operator, filter_value_str):
        cell_value_str_lower = cell_value_str.lower()
        filter_value_str_lower = filter_value_str.lower() if filter_value_str is not None else None

        op = operator.upper()

        if op in ('>', '<', '>=', '<=', '=', '!='):
            # Numeric or direct equality/inequality for strings if numeric fails
            try:
                num_cell_val = float(cell_value_str)
                num_filter_val = float(filter_value_str)
                if op == '>': return num_cell_val > num_filter_val
                if op == '<': return num_cell_val < num_filter_val
                if op == '>=': return num_cell_val >= num_filter_val
                if op == '<=': return num_cell_val <= num_filter_val
                if op == '=': return num_cell_val == num_filter_val
                if op == '!=': return num_cell_val != num_filter_val
            except (ValueError, TypeError):
                # Fallback to string comparison for = and !=
                if op == '=': return cell_value_str_lower == filter_value_str_lower
                if op == '!=': return cell_value_str_lower != filter_value_str_lower
                # For other comparison ops, if not numeric, it's a mismatch for this op type
                if op in ('>', '<', '>=', '<='):
                    return False 
        
        if op == 'CONTAINS':
            return filter_value_str_lower in cell_value_str_lower
        if op == 'NOTCONTAINS':
            return filter_value_str_lower not in cell_value_str_lower
        if op == 'STARTSWITH':
            return cell_value_str_lower.startswith(filter_value_str_lower)
        if op == 'ENDSWITH':
            return cell_value_str_lower.endswith(filter_value_str_lower)
        if op == 'ISNULL':
            return cell_value_str_lower == "null" or not cell_value_str.strip()
        if op == 'ISNOTNULL':
            return cell_value_str_lower != "null" and bool(cell_value_str.strip())
        
        return False # Unknown operator

    def apply_filters(self):
        filter_text = self.filter_input.text().strip()

        if not filter_text:
            for i in range(self.table_widget.rowCount()):
                self.table_widget.setRowHidden(i, False)
            return

        active_filters = []
        # Split by AND, respecting potential spaces around AND
        filter_conditions = re.split(r'\s+AND\s+', filter_text, flags=re.IGNORECASE)

        valid_filter_structure = True
        for condition_str in filter_conditions:
            if not condition_str.strip():
                continue

            # Regex to parse: "column_name:OPERATOR:value" or "column_name:OPERATOR" or "column_name:value"
            match = re.match(r'^([^:]+?)\s*:\s*([^:]+?)(?:\s*:\s*(.+))?$', condition_str.strip())
            
            col_name, op_or_val, val_if_op = None, None, None

            if match:
                col_name = match.group(1).strip()
                op_or_val = match.group(2).strip()
                val_if_op = match.group(3).strip() if match.group(3) is not None else None

                operator = ""
                value = ""

                potential_ops = ['ISNULL', 'ISNOTNULL', '>=', '<=', '!=', '>', '<', '=', 
                                 'CONTAINS', 'NOTCONTAINS', 'STARTSWITH', 'ENDSWITH']
                # Check if op_or_val is a recognized operator
                is_recognized_op = False
                for po in potential_ops:
                    if op_or_val.upper() == po:
                        operator = po
                        is_recognized_op = True
                        break
                
                if is_recognized_op:
                    if operator in ['ISNULL', 'ISNOTNULL']:
                        value = None # Value is not needed/used
                        if val_if_op is not None: # User provided a value for ISNULL/ISNOTNULL
                            print(f"Warning: Value '{val_if_op}' ignored for operator {operator} on column '{col_name}'.")
                    elif val_if_op is not None: # Operator needs a value, and it's provided
                        value = val_if_op
                    else: # Operator needs a value, but it's missing
                        print(f"Error: Operator {operator} for column '{col_name}' requires a value.")
                        valid_filter_structure = False
                        break
                else: # op_or_val is not an operator, so it must be the value, and operator is CONTAINS
                    operator = 'CONTAINS'
                    value = op_or_val # The "operator" part was actually the start of the value
                    if val_if_op is not None: # And if there was a third part, append it to the value
                        value += ":" + val_if_op

            else: # Simpler format: col_name:value (implies CONTAINS)
                parts = condition_str.split(':', 1)
                if len(parts) == 2:
                    col_name = parts[0].strip()
                    operator = 'CONTAINS'
                    value = parts[1].strip()
                else:
                    print(f"Error: Malformed filter condition '{condition_str}'. Expected format: col:op:val, col:op, or col:val")
                    valid_filter_structure = False
                    break
            
            if not col_name:
                print(f"Error: Missing column name in condition: '{condition_str}'")
                valid_filter_structure = False
                break

            try:
                col_idx = self.headers.index(col_name)
                active_filters.append({'col_idx': col_idx, 'operator': operator, 'value': value, 'raw_col_name': col_name})
            except ValueError:
                print(f"Warning: Column '{col_name}' not found in headers.")
                valid_filter_structure = False # Treat as invalid filter if column doesn't exist
                break
        
        if not valid_filter_structure:
            for i in range(self.table_widget.rowCount()): # Hide all rows if filter syntax is bad
                self.table_widget.setRowHidden(i, True)
            return

        if not active_filters and filter_text: # If filter text exists but no valid filters parsed from it
            for i in range(self.table_widget.rowCount()):
                 self.table_widget.setRowHidden(i, True)
            return

        for row_idx in range(self.table_widget.rowCount()):
            row_visible = True
            for f_dict in active_filters:
                item = self.table_widget.item(row_idx, f_dict['col_idx'])
                cell_data_str = item.text() if item else ""
                
                if not self._evaluate_condition(cell_data_str, f_dict['operator'], f_dict['value']):
                    row_visible = False
                    break
            self.table_widget.setRowHidden(row_idx, not row_visible)

if __name__ == '__main__':
    # Example Usage (for testing this dialog independently)
    app = QApplication([])
    example_headers = ["ID", "Name", "Age", "City", "Status"]
    example_data = [
        (1, "Alice", 30, "New York", "Active"),
        (2, "Bob", 24, "Los Angeles", None),
        (3, "Charlie", 35, "New York", "Inactive"),
        (4, "David", 29, "Chicago", "Active"),
        (5, "Eve", 22, "Los Angeles", "NULL"),
        (6, "Mallory", 40, "New York", ""), 
        (7, "Trent", 33, "Chicago", "Active")
    ]
    dialog = EnhancedTableViewDialog(data=example_data, headers=example_headers)
    dialog.show()
    app.exec() 