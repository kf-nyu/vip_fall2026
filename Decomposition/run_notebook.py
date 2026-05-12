import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

try:
    with open('Decomposition_Comparative_Analysis.ipynb', 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': './'}})
    
    with open('Executed_Decomposition.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Notebook executed successfully.")
except Exception as e:
    print(f"Error executing notebook: {e}")
