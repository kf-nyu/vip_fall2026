import os
import re
import nbformat as nbf

def convert_script(filepath, out_dir):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    nb = nbf.v4.new_notebook()
    cells = []

    # Extract the top-level docstring to become a markdown cell
    match = re.match(r'^\"\"\"(.*?)\"\"\"', content, flags=re.DOTALL)
    if match:
        docstring = match.group(1).strip()
        cells.append(nbf.v4.new_markdown_cell(f"# {os.path.basename(filepath)}\n\n{docstring}"))
        content = content[match.end():].lstrip()
    else:
        cells.append(nbf.v4.new_markdown_cell(f"# {os.path.basename(filepath)}"))
    
    colab_setup = """# ─────────────────────────────────────────────────────────────────────────────
# GOOGLE COLAB SETUP
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys

try:
    import google.colab
    IN_COLAB = True
    print("Running in Google Colab. Mounting Google Drive...")
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    IN_COLAB = False
    print("Running locally. No drive mount needed.")"""
    cells.append(nbf.v4.new_code_cell(colab_setup))
    
    # Split the script wherever there are large decorative header blocks
    # e.g.:
    # # ─────────────────────────────────────────────────────────────────────────────
    # #  DATA LOADING
    # # ─────────────────────────────────────────────────────────────────────────────
    pattern = re.compile(r'# ─{10,}\n#\s*(.*?)\n# ─{10,}')
    parts = pattern.split(content)
    
    # The first split part contains imports and constants before the first header
    if parts[0].strip():
        cells.append(nbf.v4.new_code_cell(parts[0].strip()))
        
    # Iterate through the matched header titles and their following code bodies
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        code_body = parts[i+1].strip()
        
        if header:
             cells.append(nbf.v4.new_markdown_cell(f"### {header}"))
        if code_body:
             if 'args = parser.parse_args()' in code_body:
                 code_body = code_body.replace('args = parser.parse_args()', 'args = parser.parse_args(args=[])')
             if 'Path(__file__).parent.parent' in code_body:
                 code_body = code_body.replace('Path(__file__).parent.parent', 
                 "(Path('/content/drive/MyDrive/VIP/week1_submission_package') if 'google.colab' in sys.modules else Path.cwd().parent)")
             if 'Path(__file__).resolve().parent.parent' in code_body:
                 code_body = code_body.replace('Path(__file__).resolve().parent.parent', 
                 "(Path('/content/drive/MyDrive/VIP/week1_submission_package') if 'google.colab' in sys.modules else Path.cwd().parent)")
             cells.append(nbf.v4.new_code_cell(code_body))
             
    nb.cells = cells
    out_path = os.path.join(out_dir, os.path.basename(filepath).replace('.py', '.ipynb'))
    with open(out_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Converted {os.path.basename(filepath)} -> {out_path}")

def main():
    project_dir = '/Users/kfunaki/Projects/vip'
    out_dir = os.path.join(project_dir, 'revised_submission')
    os.makedirs(out_dir, exist_ok=True)
    
    scripts = [
        'ARX/ARX_Revised.py',
        'TCN/TCN_Revised.py',
        'ELM/ELM_Revised.py',
        'FAVAR/FAVAR_Revised.py'
    ]
    
    for script in scripts:
        full_path = os.path.join(project_dir, script)
        if os.path.exists(full_path):
            convert_script(full_path, out_dir)
        else:
            print(f"Skipping not found: {full_path}")

if __name__ == '__main__':
    main()
