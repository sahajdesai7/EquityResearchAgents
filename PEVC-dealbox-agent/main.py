import subprocess
import sys
import time
from pathlib import Path

# --- CONFIGURATION ---
# Define the project root relative to this script
# Assumes main.py is in the root 'PEVC-dealbox-agent' folder
PROJECT_ROOT = Path(__file__).resolve().parent
MODULES_DIR = PROJECT_ROOT / "modules"

# Define the sequence of scripts to run
PIPELINE = [
    {
        "name": "Crawler",
        "file": MODULES_DIR / "crawl_smergers.py",
        "desc": "Crawling Smergers for raw data..."
    },
    {
        "name": "Cleaner",
        "file": MODULES_DIR / "clean_smergers_llm.py",
        "desc": "Standardizing data using Groq (LLM)..."
    },
    {
        "name": "Filter",
        "file": MODULES_DIR / "filter_dealbox.py",
        "desc": "Applying investment logic and geographic screening..."
    }
]

def run_pipeline():
    print("🚀 Starting PEVC Dealbox Pipeline...\n")
    
    total_start = time.time()

    for step in PIPELINE:
        script_path = step["file"]
        
        # Verify script exists
        if not script_path.exists():
            print(f"❌ Error: Could not find {step['name']} module at {script_path}")
            sys.exit(1)

        print(f"--- Step: {step['name']} ---")
        print(f"{step['desc']}")
        
        try:
            # Run the script and wait for it to finish
            # check=True raises a CalledProcessError if the script fails (non-zero exit code)
            result = subprocess.run(
                [sys.executable, str(script_path)], 
                check=True,
                text=True
            )
            print(f"✅ {step['name']} completed successfully.\n")
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Pipeline failed at step: {step['name']}")
            print(f"Error details: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Unexpected error occurred: {e}")
            sys.exit(1)

    total_time = time.time() - total_start
    print(f"🎉 Pipeline finished successfully in {total_time:.2f} seconds.")
    print(f"📂 Check the 'json' folder for results.")

if __name__ == "__main__":
    run_pipeline()