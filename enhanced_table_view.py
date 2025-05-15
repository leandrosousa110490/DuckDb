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
        filter_layout.addWidget(QLabel("Filter (e.g., Name:CONTAINS:John AND Age:>:30 AND City:ISNULL):"))
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("ColName:OPERATOR:Value AND ... (Operators: >, <, >=, <=, =, !=, CONTAINS, NOTCONTAINS, STARTSWITH, ENDSWITH, ISNULL, ISNOTNULL)")
        self.filter_input.textChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.filter_input)
        self.layout.addLayout(filter_layout)

        # Results table
        self.table_widget = QTableWidget()
        self.table_widget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.layout.addWidget(self.table_widget)

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
        self.apply_filters() # Apply initial filters if any

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