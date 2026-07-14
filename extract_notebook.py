import json
import os

notebook_path = "01_Data_Exploration.ipynb"
output_path = "extracted_code.py"

if os.path.exists(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    code_cells = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            code_cells.append(f"# === CELL ===\n{source}\n\n")
            
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(code_cells)
    print(f"Successfully extracted code to {output_path}")
else:
    print(f"Notebook {notebook_path} not found.")
