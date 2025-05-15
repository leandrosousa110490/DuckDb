from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel, QWidget, QMessageBox
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl, Qt
import plotly.graph_objects as go
import pandas as pd

class ChartViewDialog(QDialog):
    def __init__(self, data, headers, parent=None):
        super().__init__(parent)
        self.data = data
        self.headers = headers
        self.df = pd.DataFrame(self.data, columns=self.headers) # Work with pandas DataFrame

        self.setWindowTitle("Chart Builder")
        self.setMinimumSize(1000, 700)
        # Allow maximizing
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint)

        self.layout = QVBoxLayout(self)

        # Controls Area
        controls_widget = QWidget()
        self.controls_layout = QHBoxLayout(controls_widget)
        self.layout.addWidget(controls_widget)

        # Chart Type
        self.chart_type_label = QLabel("Chart Type:")
        self.controls_layout.addWidget(self.chart_type_label)
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(["Bar", "Line", "Scatter", "Pie", "Histogram"])
        self.controls_layout.addWidget(self.chart_type_combo)

        # X-Axis
        self.x_axis_label = QLabel("X-Axis:")
        self.controls_layout.addWidget(self.x_axis_label)
        self.x_axis_combo = QComboBox()
        self.x_axis_combo.addItems(self.headers)
        self.controls_layout.addWidget(self.x_axis_combo)

        # Y-Axis
        self.y_axis_label = QLabel("Y-Axis:")
        self.controls_layout.addWidget(self.y_axis_label)
        self.y_axis_combo = QComboBox()
        self.y_axis_combo.addItems(self.headers)
        self.controls_layout.addWidget(self.y_axis_combo)
        
        # Aggregation for Y-Axis (relevant for Bar, Line, Pie)
        self.y_agg_label = QLabel("Y-Axis Aggregation:")
        self.controls_layout.addWidget(self.y_agg_label)
        self.y_agg_combo = QComboBox()
        self.y_agg_combo.addItems(["None", "Sum", "Average", "Count", "Min", "Max"])
        # Hide by default, show based on chart type
        self.y_agg_label.setVisible(False)
        self.y_agg_combo.setVisible(False)
        self.controls_layout.addWidget(self.y_agg_combo)

        # Color/Group By (Optional)
        self.color_by_label = QLabel("Group/Color By (Optional):")
        self.controls_layout.addWidget(self.color_by_label)
        self.color_by_combo = QComboBox()
        self.color_by_combo.addItem("None") # Option for no color grouping
        self.color_by_combo.addItems(self.headers)
        self.controls_layout.addWidget(self.color_by_combo)
        
        # Value for Histogram (if Histogram is selected)
        self.hist_value_label = QLabel("Value (for Histogram):")
        self.controls_layout.addWidget(self.hist_value_label)
        self.hist_value_combo = QComboBox()
        self.hist_value_combo.addItems(self.headers)
        self.hist_value_label.setVisible(False)
        self.hist_value_combo.setVisible(False)
        self.controls_layout.addWidget(self.hist_value_combo)

        self.update_chart_button = QPushButton("Update Chart")
        self.update_chart_button.clicked.connect(self.render_chart)
        self.controls_layout.addWidget(self.update_chart_button)
        
        self.controls_layout.addStretch() # Push controls to the left

        # Chart Display Area
        self.web_view = QWebEngineView()
        self.layout.addWidget(self.web_view, 1) # Give web_view more stretch factor

        # Connect chart type change to show/hide relevant controls
        self.chart_type_combo.currentTextChanged.connect(self.on_chart_type_change)
        self.on_chart_type_change(self.chart_type_combo.currentText()) # Initial setup

        # Initial chart render
        self.render_chart()

    def on_chart_type_change(self, chart_type):
        is_bar_line_pie = chart_type in ["Bar", "Line", "Pie"]
        is_scatter = chart_type == "Scatter"
        is_histogram = chart_type == "Histogram"

        # Y-Axis and its aggregation
        self.y_axis_combo.setVisible(is_bar_line_pie or is_scatter)
        self.y_axis_label.setVisible(is_bar_line_pie or is_scatter)

        self.y_agg_combo.setVisible(is_bar_line_pie)
        self.y_agg_label.setVisible(is_bar_line_pie)
        
        # X-Axis (always visible for these types, but label might change for Pie)
        self.x_axis_combo.setVisible(is_bar_line_pie or is_scatter or is_histogram) # Histogram uses X for category too
        self.x_axis_label.setVisible(is_bar_line_pie or is_scatter or is_histogram)
        if chart_type == "Pie":
            self.x_axis_label.setText("Labels (Names):") # X-axis label
            self.y_axis_label.setText("Values:") # Y-axis label
        else:
            self.x_axis_label.setText("X-Axis:")
            self.y_axis_label.setText("Y-Axis:")


        # Histogram specific value column
        self.hist_value_combo.setVisible(is_histogram)
        self.hist_value_label.setVisible(is_histogram)
        if is_histogram: # For histogram, X-axis is often the value to bin, or can be a category for grouped histograms
             self.x_axis_label.setText("Value to Bin (X-Axis):") # X-axis label
             self.y_axis_combo.setVisible(False) # Y-axis is count for histogram
             self.y_axis_label.setVisible(False) # Y-axis label


    def render_chart(self):
        if self.df.empty:
            self.web_view.setHtml("<html><body>No data to display.</body></html>")
            return

        chart_type = self.chart_type_combo.currentText()
        x_col = self.x_axis_combo.currentText()
        y_col = self.y_axis_combo.currentText()
        y_agg = self.y_agg_combo.currentText()
        color_col = self.color_by_combo.currentText()
        if color_col == "None":
            color_col = None
            
        hist_val_col = self.hist_value_combo.currentText()


        fig = go.Figure()
        
        # Ensure numeric conversions for relevant columns if possible
        # This is a simplified approach; more robust type handling might be needed
        temp_df = self.df.copy()
        try:
            if x_col in temp_df.columns: temp_df[x_col] = pd.to_numeric(temp_df[x_col], errors='ignore')
            if y_col in temp_df.columns: temp_df[y_col] = pd.to_numeric(temp_df[y_col], errors='ignore')
            if color_col and color_col in temp_df.columns: temp_df[color_col] = temp_df[color_col].astype(str) # Color usually categorical
            if hist_val_col in temp_df.columns: temp_df[hist_val_col] = pd.to_numeric(temp_df[hist_val_col], errors='coerce')

        except Exception as e:
            print(f"Error during data conversion for chart: {e}")
            # Potentially show error to user

        try:
            if chart_type == "Bar" or chart_type == "Line":
                if y_agg != "None":
                    if not color_col:
                        grouped = temp_df.groupby(x_col)
                    else:
                        grouped = temp_df.groupby([x_col, color_col])
                    
                    if y_agg == "Sum": agg_data = grouped[y_col].sum()
                    elif y_agg == "Average": agg_data = grouped[y_col].mean()
                    elif y_agg == "Count": agg_data = grouped[y_col].count() # or .size() if y_col is not relevant for count
                    elif y_agg == "Min": agg_data = grouped[y_col].min()
                    elif y_agg == "Max": agg_data = grouped[y_col].max()
                    else: agg_data = temp_df.set_index(x_col)[y_col] # Should not happen with "None" check

                    agg_data = agg_data.reset_index()
                    
                    plot_func = go.Bar if chart_type == "Bar" else go.Line
                    if not color_col:
                        fig.add_trace(plot_func(x=agg_data[x_col], y=agg_data[y_col], name=y_col))
                    else:
                        for group_name, group_df in agg_data.groupby(color_col):
                            fig.add_trace(plot_func(x=group_df[x_col], y=group_df[y_col], name=str(group_name)))
                else: # No aggregation
                    if not color_col:
                        fig.add_trace(go.Bar(x=temp_df[x_col], y=temp_df[y_col], name=y_col) if chart_type == "Bar" 
                                      else go.Scatter(x=temp_df[x_col], y=temp_df[y_col], mode='lines+markers', name=y_col))
                    else:
                        for name, group in temp_df.groupby(color_col):
                             fig.add_trace(go.Bar(x=group[x_col], y=group[y_col], name=str(name)) if chart_type == "Bar" 
                                      else go.Scatter(x=group[x_col], y=group[y_col], mode='lines+markers', name=str(name)))


            elif chart_type == "Scatter":
                if not color_col:
                    fig.add_trace(go.Scatter(x=temp_df[x_col], y=temp_df[y_col], mode='markers'))
                else:
                    for name, group in temp_df.groupby(color_col):
                        fig.add_trace(go.Scatter(x=group[x_col], y=group[y_col], mode='markers', name=str(name)))
                fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
            
            elif chart_type == "Pie":
                # For Pie, X-axis is labels, Y-axis is values. Aggregation on Y-axis.
                if y_agg != "None" and y_agg != "Count": # Count doesn't make sense for pie chart values directly, usually count of x_col
                    grouped = temp_df.groupby(x_col)
                    if y_agg == "Sum": pie_data = grouped[y_col].sum()
                    elif y_agg == "Average": pie_data = grouped[y_col].mean() # Avg might be weird for pie
                    # Min/Max don't typically make sense for pie charts directly.
                    else: pie_data = temp_df.groupby(x_col)[y_col].sum() # Default to sum for safety
                    pie_data = pie_data.reset_index()
                    fig.add_trace(go.Pie(labels=pie_data[x_col], values=pie_data[y_col]))
                elif y_agg == "Count": # Count occurrences of x_col categories
                    pie_data = temp_df[x_col].value_counts().reset_index()
                    pie_data.columns = [x_col, 'count']
                    fig.add_trace(go.Pie(labels=pie_data[x_col], values=pie_data['count']))
                else: # No aggregation, direct values if y_col is numeric
                     fig.add_trace(go.Pie(labels=temp_df[x_col], values=temp_df[y_col]))


            elif chart_type == "Histogram":
                # Use hist_val_col for the values to be binned.
                # x_col can be used for color grouping (creating overlaid or stacked histograms)
                if not color_col or color_col == "None":
                    fig.add_trace(go.Histogram(x=temp_df[hist_val_col], name=hist_val_col))
                else:
                    for name, group in temp_df.groupby(color_col):
                        fig.add_trace(go.Histogram(x=group[hist_val_col], name=str(name)))
                fig.update_layout(barmode='overlay') # or 'stack'
                fig.update_traces(opacity=0.75)


            fig.update_layout(title=f"{chart_type} Chart: {y_col if chart_type != 'Histogram' else hist_val_col} by {x_col}" + (f" grouped by {color_col}" if color_col else ""))
            html = fig.to_html(full_html=False, include_plotlyjs='cdn')
            self.web_view.setHtml(html)

        except Exception as e:
            QMessageBox.critical(self, "Chart Error", f"Could not generate chart: {e}\nCheck column selections and data types.")
            self.web_view.setHtml(f"<html><body>Error generating chart: {e}</body></html>")


if __name__ == '__main__':
    # Example Usage
    app = QApplication([])
    example_headers = ["Category", "Value", "Group", "Size"]
    example_data = [
        ("A", 10, "X1", 5), ("B", 15, "X1", 8), ("A", 20, "X2", 3),
        ("C", 25, "X1", 10), ("B", 30, "X2", 6), ("A", 12, "X1", 9),
        ("C", 18, "X2", 4)
    ]
    dialog = ChartViewDialog(data=example_data, headers=example_headers)
    dialog.show()
    app.exec() 