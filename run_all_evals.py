import subprocess
import os
import sys
import re
from datetime import datetime
import locale

def get_system_encoding():
    """Gets the preferred encoding of the system."""
    try:
        # For Windows, get the OEM code page
        if os.name == 'nt':
            return f"cp{subprocess.check_output(['chcp'], text=True).split()[-1]}"
        return locale.getpreferredencoding()
    except Exception:
        return "utf-8"

def extract_metrics(output):
    """Extracts the evaluation metrics block from the script output."""
    # Use regex to find the metrics block
    match = re.search(r"---\sEvaluation\sMetrics\s---([\s\S]*)", output)
    if match:
        metrics_block = match.group(1).strip()
        # Clean up the block to remove extra empty lines that might exist at the end
        lines = [line for line in metrics_block.split('\n') if line.strip()]
        return "\n".join(lines)
    return None

def run_evaluation_scripts():
    """
    Runs all evaluation scripts in the 'eval' directory and saves only their 
    final metrics to a single text file.
    """
    eval_scripts = [
        "eval_baseline.py",
        "dynamic_feature_pool_pose_aware_correction.py",
        "eval_proxyfusion_dynamic_pooling.py",
        "eval_proxyfusion_quality.py"
    ]
    
    output_filename = "evaluation_results.txt"
    eval_directory = "eval"
    system_encoding = get_system_encoding()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(output_filename, "w", encoding="utf-8") as outfile:
        print(f"Starting evaluation run at {timestamp}")
        outfile.write(f"Evaluation Run Started at: {timestamp}\n")
        outfile.write("="*80 + "\n\n")

        for script_name in eval_scripts:
            script_path = os.path.join(eval_directory, script_name)
            python_executable = sys.executable
            command = [python_executable, script_path]
            
            print(f"--- Running {script_name} ---")
            
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding=system_encoding,
                    errors='ignore'
                )
                
                outfile.write(f"--- Results for {script_name} ---\n")
                
                if result.returncode == 0:
                    metrics = extract_metrics(result.stdout)
                    if metrics:
                        outfile.write(metrics + "\n")
                        print(f"--- Finished {script_name} successfully ---")
                    else:
                        outfile.write("Metrics block not found in the output.\n")
                        print(f"--- {script_name} finished, but metrics could not be extracted ---")
                else:
                    # If there was an error, write the full error log for debugging
                    outfile.write(f"Script failed with return code {result.returncode}.\n")
                    outfile.write("\n--- STDOUT ---\n")
                    outfile.write(result.stdout)
                    outfile.write("\n--- STDERR ---\n")
                    outfile.write(result.stderr)
                    print(f"--- {script_name} finished with errors (see {output_filename} for details) ---")

            except Exception as e:
                error_message = f"An unexpected error occurred: {e}\n"
                print(error_message)
                outfile.write(f"--- Results for {script_name} ---\n")
                outfile.write(error_message)

            outfile.write("\n" + "="*80 + "\n\n")

    print(f"\nAll evaluation scripts have been run. Cleaned results are saved in '{output_filename}'.")

if __name__ == "__main__":
    run_evaluation_scripts()
