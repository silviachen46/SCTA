import re
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

def clean_error_message(error_message):
    # Step 1: Remove ANSI escape sequences (color codes, formatting codes)
    ansi_escape = re.compile(r'\x1b\[[0-9;]*[mK]')
    cleaned_message = ansi_escape.sub('', error_message)

    # Step 2: Remove unnecessary repeated lines or lines with just dashes and metadata
    cleaned_message = re.sub(r'[-]+ stderr [-]+\n', '', cleaned_message)  # Remove stderr marker
    cleaned_message = re.sub(r'[-]+\n', '', cleaned_message)  # Remove repeated dashes
    cleaned_message = re.sub(r'Cell In\[\d+\], line \d+\n', '', cleaned_message)  # Remove notebook line references

    # Step 3: Remove paths to internal library files (optional)
    cleaned_message = re.sub(r'File .*site-packages/.*\n', '', cleaned_message)  # Remove Python package tracebacks

    # Step 4: Remove excessive whitespace lines
    cleaned_message = re.sub(r'\n\s*\n', '\n', cleaned_message)  # Remove extra newlines

    return cleaned_message.strip()

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

import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import psutil
import gc

def clean_error_message(error_msg: str) -> str:
    return error_msg

def execute_code_in_notebook(notebook_path: str, code: str):
    status_dictionary = {}
    memory_code = (
        "%reset -f\n"  # 
        "import psutil\n"
        "process = psutil.Process()\n"
        "print(f'SCA_AGENT_MEMORY_BEFORE: {process.memory_info().rss}')\n"
        f"{code}\n"
        "print(f'SCA_AGENT_MEMORY_AFTER: {process.memory_info().rss}')\n"
    )
    
  
    try:
        if not os.path.exists(notebook_path) or os.stat(notebook_path).st_size == 0:
            nb = nbformat.v4.new_notebook()
            print("Creating new notebook.")
        else:
            with open(notebook_path, "r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)
    except Exception as e:
        print(f"Error handling notebook: {e}")
        return {"code_state": "Error", "code_error": str(e), "errored_code" : code}, None

  
    new_cell = nbformat.v4.new_code_cell(memory_code)
    nb.cells.append(new_cell)
    cell_id = new_cell.get("id", "unknown_id")
    status_dictionary['metadata'] = cell_id
    
   
    ep = ExecutePreprocessor(
        timeout=600,
        kernel_name="python3",
        allow_errors=True,
        extra_arguments=["--HistoryManager.enabled=False"]  
    )
    
    try:
       
        resources = {'metadata': {'path': '.'}}
        ep.preprocess(nb, resources)
        
      
        output = new_cell.get("outputs", [])
        memory_data = {'before': None, 'after': None}
        result_output = []
        
        for out in output:
            if 'text' in out:
                text = out['text']
                for line in text.split('\n'):
                    if 'SCA_AGENT_MEMORY_BEFORE:' in line:
                        memory_data['before'] = int(line.split(': ')[1])
                    elif 'SCA_AGENT_MEMORY_AFTER:' in line:
                        memory_data['after'] = int(line.split(': ')[1])
                result_output.append(text.strip())
            
            if 'traceback' in out:
                status_dictionary.update({
                    'code_state': 'Error',
                    'code_error': clean_error_message("\n".join(out['traceback'])),
                    "errored_code" : code
                })
                break

      
        if all(memory_data.values()):
            status_dictionary['memory_usage'] = {
                **memory_data,
                'used': memory_data['after'] - memory_data['before'],
                'message': f"Memory used: {(memory_data['after'] - memory_data['before']) / 1e6:.2f} MB"
            }
        
       
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
            
        status_dictionary.setdefault('code_state', 'Success')
        status_dictionary['code_result'] = result_output or ["No output"]
        
    except Exception as e:
        status_dictionary.update({
            'code_state': 'Error',
            'code_error': clean_error_message(str(e)),
            "errored_code" : code
        })
      
        try:
            nb.cells.remove(new_cell)
        except ValueError:
            pass
        
    finally:
        if hasattr(ep, 'kernel_manager') and ep.kernel_manager.is_alive():
            try:
                ep.kernel_manager.shutdown_kernel(now=True)
                ep.kernel_manager.cleanup_resources()
            except Exception as e:
                print(f"Kernel shutdown error: {str(e)}")
        del ep
        gc.collect()
        
    return status_dictionary  

def read_last_result(filepath: str) -> str:
    """
    Read only the most recent 'Step N Result' block from the result file.
    """
    last_block = []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Reverse iterate and collect lines until the start of the last block
    found_start = False
    for line in reversed(lines):
        if line.startswith("Step ") and "Result" in line:
            found_start = True
            last_block.append(line)
            break
        last_block.append(line)

    if not found_start:
        return ""

    # Reverse the collected block to correct order
    return "".join(reversed(last_block))
    
