with open('app.py', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Find where _on_import_finished is defined
start_line = -1
for i, line in enumerate(lines):
    if 'def _on_import_finished(self):' in line:
        start_line = i
        break

if start_line > 0:
    # Fix the indentation of the try block
    fixed_lines = []
    fixed_lines.extend(lines[:start_line + 5])  # Include method signature and up to 'if hasattr(self...'
    
    # Add the properly indented try block
    fixed_lines.append('            try:\n')
    fixed_lines.append('                if self.progress.isVisible():\n')
    fixed_lines.append('                    self.progress.close()\n')
    fixed_lines.append('            except RuntimeError:\n')
    
    # Continue with the rest of the file
    fixed_lines.extend(lines[start_line + 9:])  # Skip the old try block
    
    # Write back to the file
    with open('app.py', 'w', encoding='utf-8') as file:
        file.writelines(fixed_lines)
    
    print("✅ Fixed indentation in _on_import_finished method")
else:
    print("❌ Could not find _on_import_finished method in app.py") 