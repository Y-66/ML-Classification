import os
import glob
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys

def execute_notebook(filepath):
    print(f"  Loading {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Configure the executor
    # timeout=-1 means no timeout
    ep = ExecutePreprocessor(timeout=-1, kernel_name='python3')
    
    # execute in the directory of the notebook so relative paths work
    notebook_dir = os.path.dirname(os.path.abspath(filepath))
    
    print(f"  Executing {filepath} (this may take a while)...")
    try:
        ep.preprocess(nb, {'metadata': {'path': notebook_dir}})
        print(f"  Execution successful.")
    except Exception as e:
        print(f"  Execution FAILED: {e}")
        # Return the notebook anyway, possibly partially executed
        return nb, str(e)
        
    return nb, None

def parse_notebook_content(nb, filename, error_msg=None):
    summary = []
    
    # Add title based on filename
    model_name = filename.replace('.ipynb', '')
    summary.append(f"# {model_name}\n")
    
    if error_msg:
        summary.append(f"**Warning: Notebook execution failed or incomplete.**\n> {error_msg}\n")

    cells = nb.get('cells', [])
    for cell in cells:
        cell_type = cell.get('cell_type')
        
        if cell_type == 'markdown':
            source = cell.get('source', '')
            summary.append(source + "\n")
            
        elif cell_type == 'code':
            # We treat the source code (input) as part of the report if desired, 
            # but usually for results we focus on output. 
            # User request: "所有模型的结果及其分析" (Results and Analysis). 
            # Often context from code is useful, but let's stick to outputs for brevity unless requested.
            # Actually, let's include input code in a collapsed or block? 
            # The previous script ignored input code. I'll stick to that to keep it clean, 
            # or maybe include it if it's short. Let's strictly follow "results and analysis".
            
            outputs = cell.get('outputs', [])
            for output in outputs:
                output_type = output.get('output_type')
                
                if output_type == 'stream':
                    text = output.get('text', '')
                    if text.strip():
                        summary.append("```text\n" + text + "\n```\n")
                        
                elif output_type == 'execute_result':
                    data = output.get('data', {})
                    # Prefer text/plain
                    text = data.get('text/plain', '')
                    if text.strip():
                        summary.append("```text\n" + text + "\n```\n")
                
                elif output_type == 'display_data':
                    data = output.get('data', {})
                    # Handle images (png/jpeg)
                    if 'image/png' in data:
                        # We can embed base64 image in markdown
                        # ![alt](data:image/png;base64,...)
                        # VS Code supports this.
                        base64_data = data['image/png']
                        # Remove newlines in base64
                        base64_data = base64_data.replace('\n', '')
                        summary.append(f"![Chart](data:image/png;base64,{base64_data})\n")
                    elif 'text/plain' in data:
                        text = data['text/plain']
                        summary.append(f"```text\n{text}\n```\n")
                
                elif output_type == 'error':
                    ename = output.get('ename', 'Error')
                    evalue = output.get('evalue', '')
                    traceback = output.get('traceback', [])
                    # format traceback
                    tb_text = '\n'.join(traceback)
                    summary.append(f"❌ **Error**: `{ename}`: {evalue}\n")
                    summary.append(f"<details><summary>Traceback</summary>\n\n```text\n{tb_text}\n```\n\n</details>\n")

    return "\n".join(summary)

def main():
    models_dir = 'models'
    output_file = 'Model_Results_And_Analysis.md'
    
    if not os.path.exists(models_dir):
        print(f"Directory {models_dir} not found.")
        return

    # Sort notebooks for consistent ordering
    notebooks = sorted(glob.glob(os.path.join(models_dir, '*.ipynb')))
    
    all_summaries = []
    all_summaries.append("# Classification Model Results & Analysis Summary (Auto-Run)\n")
    all_summaries.append(f"**Source Directory**: `{models_dir}/`\n")
    all_summaries.append(f"**Generated**: {os.path.basename(sys.argv[0])}\n")
    all_summaries.append("---\n")
    
    total = len(notebooks)
    # Filter only for the notebooks the user listed in workspace or all? 
    # Workspace info lists many. I'll execute all found in models/.
    
    for i, nb_path in enumerate(notebooks):
        filename = os.path.basename(nb_path)
        print(f"[{i+1}/{total}] Processing {filename}...")
        
        try:
            nb, error_msg = execute_notebook(nb_path)
            nb_summary = parse_notebook_content(nb, filename, error_msg)
            
            all_summaries.append(nb_summary)
            all_summaries.append("\n<br>\n\n---\n<br>\n") # Visual separator
            
        except ImportError:
            print("Error: nbformat or nbconvert not installed. Please install them: pip install nbformat nbconvert")
            return
        except Exception as e:
            print(f"Critical error processing {nb_path}: {e}")
            all_summaries.append(f"## {filename}\n\nCritical error processing notebook: {e}\n")
            
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_summaries))
    
    print(f"\nSummary successfully generated: {output_file}")

if __name__ == "__main__":
    main()
