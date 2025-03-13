import re
from parser import clean_error_message
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import shutil

def execute_code_in_notebook(notebook_path: str, code: str):
    status_dictionary = {}
   
    if not os.path.exists(notebook_path) or os.stat(notebook_path).st_size == 0:
        
        nb = nbformat.v4.new_notebook()
        print("Creating a new notebook as the given one is empty or missing.")
    else:
        
        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)
        except Exception as e:
            print(f"Error reading notebook: {e}")
            return "Error: Notebook file is corrupted or not a valid JSON."
    new_cell = nbformat.v4.new_code_cell(code)
    nb.cells.append(new_cell)
    last_cell = nb.cells[-1]

    cell_id = last_cell.get("id", "unknownid")
    status_dictionary['metadata'] = cell_id
   
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    
    try:
        ep.preprocess(nb, {'metadata': {'path': '.'}})
       
        output = new_cell.get("outputs", [])
        result = []
        
        for out in output:
            if 'text' in out:
                result.append(out['text'])
            if 'traceback' in out:
                status_dictionary['code_state'] = 'Error'
                status_dictionary['code_error'] = "\n".join(out['traceback'])
                status_dictionary['code_error'] = clean_error_message(status_dictionary['code_error'])
                nb.cells.pop(-1)  
                print(f"Error in execution. Removed cell {cell_id} from notebook.")  

                return status_dictionary  
        
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        status_dictionary['code_result'] = result if result else ["Executed successfully, but no output."]
        status_dictionary['code_state'] = 'Success'
        
        return status_dictionary
    
    except Exception as e:
        nb.cells.pop(-1)  
        print(f"Execution failed. Removed cell {cell_id} from notebook.1")  # Added
        status_dictionary['code_state'] = 'Error'
        status_dictionary['code_error'] = f"{str(e)}"
        status_dictionary['code_error'] = clean_error_message(status_dictionary['code_error'])
        return status_dictionary

# output = execute_code_in_notebook("/Users/silviachen/Documents/Software/SCAagent/agent/test.ipynb", test_script)
# print(output)

def delete_notebook_blocks(notebook_path: str, cell_id: str = None):
    """
    Removes the latest code cell from the notebook, or a specific code cell by ID if provided.

    Parameters:
        notebook_path (str): Path to the Jupyter notebook.
        cell_id (str, optional): The unique ID of the cell to remove. If not provided, removes the latest code cell.

    Returns:
        dict: A status dictionary containing details of the removal operation.
    """
    status_dictionary = {}

    
    if not os.path.exists(notebook_path):
        status_dictionary['status'] = 'Error'
        status_dictionary['message'] = f"Notebook {notebook_path} does not exist."
        return status_dictionary

    
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as e:
        status_dictionary['status'] = 'Error'
        status_dictionary['message'] = f"Error reading notebook: {str(e)}"
        return status_dictionary

    
    if not nb.cells:
        status_dictionary['status'] = 'Error'
        status_dictionary['message'] = "Notebook is empty, no cells to remove."
        return status_dictionary

    removed_cell = None

    if cell_id:
        
        for i, cell in enumerate(nb.cells):
            if cell.get("id") == cell_id:
                removed_cell = nb.cells.pop(i)
                break
        if removed_cell:
            status_dictionary['status'] = 'Success'
            status_dictionary['message'] = f"Removed code cell with ID {cell_id}."
        else:
            status_dictionary['status'] = 'Error'
            status_dictionary['message'] = f"No code cell found with ID {cell_id}."
            return status_dictionary
    else:
       
        for i in range(len(nb.cells) - 1, -1, -1):
            if nb.cells[i].cell_type == "code":
                removed_cell = nb.cells.pop(i)
                break

        if removed_cell:
            status_dictionary['status'] = 'Success'
            status_dictionary['message'] = "Removed the latest code cell."
            status_dictionary['removed_id'] = removed_cell.metadata.get("id", "unknown")
        else:
            status_dictionary['status'] = 'Error'
            status_dictionary['message'] = "No code cells found to remove."
            return status_dictionary

  
    try:
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
    except Exception as e:
        status_dictionary['status'] = 'Error'
        status_dictionary['message'] = f"Failed to save the notebook: {str(e)}"
        return status_dictionary

    return status_dictionary

def extract_python_code(output_string):
    """
    Extracts Python code from a given output string.
    The function removes everything outside triple backticks and the 'python' keyword.
    """
    pattern = r"```python\n(.*?)\n```"
    match = re.findall(pattern, output_string, re.DOTALL)
    
    if match:
        return match[0].strip() 
    return ""

# print(delete_notebook_blocks("/Users/silviachen/Documents/Software/SCAagent/agent/test.ipynb", """aef4becd"""))

def update_and_execute_notebook(notebook_path: str, replacements: dict):

    if not os.path.exists(notebook_path):
        return {"status": "Error", "message": f"Notebook {notebook_path} does not exist."}
    
    # Backup original notebook in case rollback is needed
    backup_path = notebook_path + ".backup"
    shutil.copy(notebook_path, backup_path)
    
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as e:
        return {"status": "Error", "message": f"Error reading notebook: {str(e)}"}
    
    original_code = {}  # Store original code before replacing
    cell_positions = {}  # Store cell position index
    modified_cell_ids = set(replacements.keys())
    earliest_index = None
    
    # Replace code content in corresponding cells
    for i, cell in enumerate(nb.cells):
        cell_id = cell.get("id")
        if cell_id in replacements:
            original_code[cell_id] = cell.source  # Backup original code
            cell.source = replacements[cell_id]  # Replace with new code
            cell_positions[cell_id] = i
            
            if earliest_index is None or i < earliest_index:
                earliest_index = i
    
    if earliest_index is None:
        return {"status": "Error", "message": "No matching cell IDs found in the notebook."}
    
    # Execute all cells from the earliest modified cell onward
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    try:
        ep.preprocess(nb, {'metadata': {'path': '.'}}, start_cell=earliest_index)
        
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        return {"status": "Success", "message": "Notebook updated and executed successfully."}
    
    except Exception as e:
        # Rollback to original notebook on failure
        shutil.copy(backup_path, notebook_path)
        return {"status": "Error", "message": f"Execution failed. Rolled back to the previous version: {str(e)}"}
    
    finally:
        # Cleanup backup file
        os.remove(backup_path)