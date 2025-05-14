import re

with open('app.py', 'r', encoding='utf-8') as file:
    content = file.read()

# Fix the table replacement code for Excel import
pattern = re.compile(r'''elif mode == "Replace Existing Table":
                # Use DROP CASCADE \+ CREATE AS to ensure no old constraints \(like PKs\) remain
                # This forcefully removes the table and anything depending on it\.
                drop_query = f'DROP TABLE IF EXISTS {quoted_table_name} CASCADE;'
                print\(f"Worker executing: {drop_query}"\)
                db_conn_worker\.execute\(drop_query\)
                
                # Create the new table from source, without constraints
                query = f'CREATE TABLE {quoted_table_name} AS {select_from_view};'
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
                query = f'CREATE TABLE {quoted_table_name} AS {select_from_view};'
                # No DELETE, ALTER, or extra schema checks needed here for Replace mode'''

content = pattern.sub(replacement, content)

# Write the updated content back to the file
with open('app.py', 'w', encoding='utf-8') as file:
    file.write(content)

print("✅ Fixed Excel table replacement functionality") 