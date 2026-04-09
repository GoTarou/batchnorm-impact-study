import sys
import os
import re
import traceback
import subprocess

# Prevent PyTorch from hanging in background Windows process
os.environ["OMP_NUM_THREADS"] = "1"

try:
    import experiment
    print("Starting optimizer experiments. This may take 15-20 minutes...")
    
    # Run the 4 optimizers
    results = experiment.run_optimizer_experiment()

    # Move to root to update REPORT.md
    os.chdir("..")
    
    with open("REPORT.md", "r", encoding="utf-8") as f:
        content = f.read()

    # Define how to find and replace the table rows
    # The table in REPORT.md looks like:
    # | SGD | 0.05 | Linear decay | — | — |
    # | SGD + Momentum | 0.05 | Linear decay | — | — |
    # | SGD + Nesterov | 0.05 | Linear decay | — | — |
    # | Adam | 0.001 | None | — | — |

    for r in results:
        model_name = r['model']
        acc = f"{r['accuracy']*100:.2f}"
        epochs = str(r['epochs_ran'])
        
        # Regex to find the row starting with "| {model_name} |"
        # and ending with two "— | — |" or similar placeholders
        pattern = re.compile(r"(\|\s*" + re.escape(model_name) + r"\s*\|.*?\|.*?\|)\s*[—\-\?]+\s*\|\s*[—\-\?]+\s*\|")
        
        # Replace the placeholders with actual accuracy and epochs
        replacement = r"\g<1> " + acc + r" | " + epochs + r" |"
        
        content = pattern.sub(replacement, content)

    # Save REPORT.md
    with open("REPORT.md", "w", encoding="utf-8") as f:
        f.write(content)
        
    print("Updated REPORT.md successfully.")
    
    # Commit and push
    subprocess.run(["git", "add", "images/", "REPORT.md"], check=True)
    subprocess.run(["git", "commit", "-m", "Auto-update: Generate Experiment 5 images and table results"], check=True)
    subprocess.run(["git", "push"], check=True)
    
    print("Successfully pushed to GitHub!")
    
except Exception as e:
    print(f"Error occurred: {e}")
    traceback.print_exc()

# Keep window open for a bit if run explicitly
import time
time.sleep(10)
