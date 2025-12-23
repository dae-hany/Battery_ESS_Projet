
import os
import sys
import shutil

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.main import ExperimentConfig, run_pipeline, LSTMDeepSurv, DeepSurvTrainer, get_dataloaders, DataConfig
import torch

def test_fast_pipeline():
    print("=== Fast Pipeline Verification ===")
    
    # 1. Setup Config for FAST run
    exp_config = ExperimentConfig(
        epochs=2,
        batch_size=4,
        hidden_dim=16,
        results_dir="results_test",
        plots_dir="plots_test"
    )
    
    # Data Config (use small subset logic effectively by just running normal load but small epochs)
    # We can't easily slice data loading without modifying code, but 14 batteries is small.
    # We will just run 2 epochs.
    
    results_dir = os.path.join(exp_config.base_dir, exp_config.results_dir)
    plots_dir = os.path.join(exp_config.base_dir, exp_config.plots_dir)
    if os.path.exists(results_dir): shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    device = 'cpu' # Force CPU 
    
    print("\n[Step 1] Loading Data...")
    data_config = DataConfig(
        base_dir=exp_config.base_dir, 
        window_size=exp_config.window_size,
        feature_set_name='dynamic'
    )
    
    train_loader, val_loader = get_dataloaders(data_config, batch_size=exp_config.batch_size)
    
    model = LSTMDeepSurv(
        input_dim=4,
        hidden_dim=exp_config.hidden_dim,
        num_layers=1,
        dropout=0.0
    )
    
    print("\n[Step 3] Starting Training (2 Epochs)...")
    trainer = DeepSurvTrainer(
        model=model,
        device=device,
        lr=exp_config.learning_rate,
        patience=2
    )
    
    history = trainer.fit(train_loader, val_loader, epochs=exp_config.epochs, verbose=1)
    
    print("\n[Step 4] Checking Results...")
    assert len(history['train_loss']) == 2
    print("Optimization finished.")
    
    # Cleanup
    if os.path.exists(results_dir): shutil.rmtree(results_dir)
    # if os.path.exists(plots_dir): shutil.rmtree(plots_dir) 
    
    print("=== Verification Passed ===")

if __name__ == "__main__":
    test_fast_pipeline()
