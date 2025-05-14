import re

# Read the file content
with open('app.py', 'r', encoding='utf-8') as file:
    content = file.read()

# Fix 1: Fix indentation in _on_import_finished method
pattern = re.compile(r'''def _on_import_finished\(self\):.*?try:
                if self\.progress\.isVisible\(\):
            self\.progress\.close\(\)''', re.DOTALL)

replacement = '''def _on_import_finished(self):
        """Clean up resources after import thread is finished."""
        print("Import thread finished.")
        # Ensure progress dialog is closed
        if hasattr(self, 'progress') and self.progress is not None:
            try:
                if self.progress.isVisible():
                    self.progress.close()'''

content = pattern.sub(replacement, content)

# Fix 2: Fix the progress dialog to prevent freezing
pattern = re.compile(r'''# --- Setup Progress Dialog --- 
        progress_text = f"Importing {os\.path\.basename\(file_path\)}..."
        if sheet_name:
            progress_text = f"Importing {os\.path\.basename\(file_path\)} \(Sheet: {sheet_name}\)..."
        self\.progress = QProgressDialog\(progress_text, "Cancel", 0, 0, self\)
        self\.progress\.setWindowModality\(Qt\.WindowModality\.WindowModal\)
        self\.progress\.setWindowTitle\("Importing Data"\)
        self\.progress\.setValue\(0\)
        self\.progress\.show\(\)''')

replacement = '''# --- Setup Progress Dialog --- 
        progress_text = f"Importing {os.path.basename(file_path)}..."
        if sheet_name:
            progress_text = f"Importing {os.path.basename(file_path)} (Sheet: {sheet_name})..."
        self.progress = QProgressDialog(progress_text, "Cancel", 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModality.ApplicationModal)  # More blocking to prevent UI interaction
        self.progress.setWindowTitle("Importing Data")
        self.progress.setValue(0)
        
        # Process pending events before showing the dialog to ensure UI is responsive
        QCoreApplication.processEvents()
        self.progress.show()
        # Process events again after show to ensure dialog is displayed
        QCoreApplication.processEvents()'''

content = pattern.sub(replacement, content)

# Fix 3: Improve table replacement functionality
pattern = re.compile(r'''elif mode == "Replace Existing Table":
            # Use DROP CASCADE \+ CREATE AS to ensure no old constraints \(like PKs\) remain
            # This forcefully removes the table and anything depending on it\.
            drop_query = f'DROP TABLE IF EXISTS {quoted_table_name} CASCADE;'
            print\(f"Worker executing: {drop_query}"\)
            db_conn_worker\.execute\(drop_query\)
            
            # Create the new table from source, without constraints
            query = f'CREATE TABLE {quoted_table_name} AS {select_with_source};'
            # No DELETE, ALTER, or extra schema checks needed here for Replace mode''')

replacement = '''elif mode == "Replace Existing Table":
            # Use DROP CASCADE + CREATE AS to ensure no old constraints (like PKs) remain
            # This forcefully removes the table and anything depending on it.
            drop_query = f'DROP TABLE IF EXISTS {quoted_table_name} CASCADE;'
            print(f"Worker executing: {drop_query}")
            db_conn_worker.execute(drop_query)
            
            # Ensure transaction is committed after drop
            db_conn_worker.execute("COMMIT;")
            
            # Check for cancellation before table creation
            if worker_ref.is_cancelled:
                raise InterruptedError("Import cancelled after table drop.")
                
            # Create the new table from source, without constraints
            query = f'CREATE TABLE {quoted_table_name} AS {select_with_source};'
            # No DELETE, ALTER, or extra schema checks needed here for Replace mode'''

content = pattern.sub(replacement, content)

# Write the updated content back to the file
with open('app.py', 'w', encoding='utf-8') as file:
    file.write(content)

print("✅ Fixed indentation in _on_import_finished method")
print("✅ Improved progress dialog handling to prevent UI freezing")
print("✅ Enhanced table replacement code to fix issues when replacing existing tables") 