
import os
import sys
import optuna
import shutil

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Override constants for testing
import src.optuna_search as search
search.N_TRIALS = 1
search.EPOCHS_PER_TRIAL = 1

def test_optuna_smoke():
    print("=== Optuna Search Smoke Test ===")
    try:
        search.run_optimization()
        print("Smoke test passed.")
    except Exception as e:
        print(f"Smoke test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_optuna_smoke()
