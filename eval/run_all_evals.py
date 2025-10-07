import subprocess
import os
from datetime import datetime

def run_evaluation_scripts():
    """
    Runs all evaluation scripts in the 'eval' directory and saves their output to a single text file.
    """
    eval_scripts = [
        "eval_baseline.py",
        "dynamic_feature_pool_pose_aware_correction.py",
        "eval_proxyfusion_dynamic_pooling.py",
        "eval_proxyfusion_quality.py"
    ]
    
    output_filename = "evaluation_results.txt"
    eval_directory = "eval"
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(output_filename, "w", encoding="utf-8") as outfile:
        print(f"Starting evaluation run at {timestamp}")
        outfile.write(f"Evaluation Run Started at: {timestamp}\n")
        outfile.write("="*80 + "\n\n")

        for script_name in eval_scripts:
            script_path = os.path.join(eval_directory, script_name)
            command = ["python", script_path]
            
            print(f"--- Running {script_name} ---")
            outfile.write(f"--- Results for {script_name} ---\n\n")
            
            try:
                # Execute the script
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    check=True  # Raise an exception if the script fails
                )
                
                # Write stdout to the file
                outfile.write(result.stdout)
                print(f"--- Finished {script_name} ---")

            except FileNotFoundError:
                error_message = f"Error: The script at {script_path} was not found.\n"
                print(error_message)
                outfile.write(error_message)
            except subprocess.CalledProcessError as e:
                # If the script returns a non-zero exit code, it's an error
                error_message = (
                    f"Error running {script_name}.\n"
                    f"Return Code: {e.returncode}\n"
                    f"Stdout:\n{e.stdout}\n"
                    f"Stderr:\n{e.stderr}\n"
                )
                print(error_message)
                outfile.write(error_message)
            except Exception as e:
                error_message = f"An unexpected error occurred while running {script_name}: {e}\n"
                print(error_message)
                outfile.write(error_message)

            outfile.write("\n" + "="*80 + "\n\n")

    print(f"\nAll evaluation scripts have been run. Results are saved in '{output_filename}'.")

if __name__ == "__main__":
    run_evaluation_scripts()
