import time
import sys
import logging
import traceback
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from processing.clean import clean_data
from processing.features import feature_pipeline
from model.train import train_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("pipeline")

def print_banner(text, color_code="36"):
    """Prints a styled banner to the console."""
    border = "=" * 65
    # Use ANSI escape codes for local terminal colour if possible
    print(f"\033[{color_code}m{border}\033[0m")
    print(f"\033[{color_code}m{text.center(65)}\033[0m")
    print(f"\033[{color_code}m{border}\033[0m")

def run_pipeline():
    start_time = time.time()
    
    print_banner("CS2 PREDICTIVE ENGINE: FULL PIPELINE", "1;35") # Bold Magenta
    
    try:
        # --- PHASE 1: CLEANING ---
        print(f"\n\033[1;33m[PHASE 1] CLEANING\033[0m")
        print("Scrubbing raw HLTV and PandaScore data, normalising team names...")
        clean_data()
        
        # --- PHASE 2: FEATURE ENGINEERING ---
        print(f"\n\033[1;33m[PHASE 2] FEATURE ENGINEERING\033[0m")
        print("Calculating temporal differentials, win streaks, and map comfort...")
        feature_pipeline()
        
        # --- PHASE 3: MODEL TRAINING ---
        print(f"\n\033[1;33m[PHASE 3] TRAINING\033[0m")
        print("Optimising PyTorch model and exporting training state...")
        train_model()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print_banner("PIPELINE COMPLETED SUCCESSFULLY", "1;32") # Bold Green
        print(f"\033[1mTotal Execution Time:\033[0m {duration:.2f} seconds")
        print(f"\033[1mArtifacts Generated:\033[0m")
        print(f"  - data/processed/clean_maps.parquet")
        print(f"  - data/processed/features.parquet")
        print(f"  - data/checkpoints/best_mvp_model.pt")
        print(f"  - data/checkpoints/scaler.pkl")
        print(f"  - data/training_state.json")
        print("\033[1;32m" + "="*65 + "\033[0m\n")

    except Exception as e:
        print_banner("PIPELINE FAILED", "1;31") # Bold Red
        logger.error(f"Error during pipeline execution: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Check if run from root
    if not (project_root / "config.py").exists():
        print("Error: pipeline.py must be run from the project root directory.")
        sys.exit(1)
        
    run_pipeline()
