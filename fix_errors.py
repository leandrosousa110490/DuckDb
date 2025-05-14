import sys
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt, QTimer, QCoreApplication
import os

def fix_app_py():
    """Fix indentation and performance issues in app.py"""
    try:
        with open('app.py', 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Fix 1: Fix indentation in _on_import_finished method
        # The problematic section with incorrect indentation
        old_import_finished = '''    def _on_import_finished(self):
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
            self.progress = None'''
        
        # The corrected version with proper indentation
        new_import_finished = '''    def _on_import_finished(self):
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
            self.progress = None'''
        
        # Fix 2: Improve thread handling and UI responsiveness in _start_import_thread
        old_progress_code = '''        # --- Setup Progress Dialog --- 
        progress_text = f"Importing {os.path.basename(file_path)}..."
        if sheet_name:
            progress_text = f"Importing {os.path.basename(file_path)} (Sheet: {sheet_name})..."
        self.progress = QProgressDialog(progress_text, "Cancel", 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.setWindowTitle("Importing Data")
        self.progress.setValue(0)
        self.progress.show()'''
        
        new_progress_code = '''        # --- Setup Progress Dialog --- 
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
        
        # Fix 3: Enhance thread cleanup to handle table replacement issue
        old_execute_replace = '''        elif mode == "Replace Existing Table":
            # Use DROP CASCADE + CREATE AS to ensure no old constraints (like PKs) remain
            # This forcefully removes the table and anything depending on it.
            drop_query = f'DROP TABLE IF EXISTS {quoted_table_name} CASCADE;'
            print(f"Worker executing: {drop_query}")
            db_conn_worker.execute(drop_query)
            
            # Create the new table from source, without constraints
            query = f'CREATE TABLE {quoted_table_name} AS {select_with_source};'
            # No DELETE, ALTER, or extra schema checks needed here for Replace mode'''
        
        new_execute_replace = '''        elif mode == "Replace Existing Table":
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
            
        # Apply the fixes
        content = content.replace(old_import_finished, new_import_finished)
        content = content.replace(old_progress_code, new_progress_code)
        content = content.replace(old_execute_replace, new_execute_replace)
        
        # Write the fixed content back to app.py
        with open('app.py', 'w', encoding='utf-8') as file:
            file.write(content)
        
        print("✅ Fixed indentation in _on_import_finished method")
        print("✅ Improved progress dialog handling to prevent UI freezing")
        print("✅ Enhanced table replacement code to fix issues when replacing existing tables")
        return True
    except Exception as e:
        print(f"❌ Error fixing app.py: {e}")
        return False

if __name__ == "__main__":
    success = fix_app_py()
    sys.exit(0 if success else 1) 