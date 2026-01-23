import pandas as pd
import matplotlib.pyplot as plt

def analyze_distributions():
    df = pd.read_csv('NASA_Augmented_EIS_Dataset.csv')
    
    # Filter only original data (no augmentation) for clarity
    df = df[df['Augmentation_Type'] == 'Original']
    
    features = ['Rs', 'Rct', 'AUC_X', 'Relative_Rs']
    
    train_ids = ['B0005', 'B0006', 'B0007']
    test_id = 'B0018'
    
    train_df = df[df['Battery_ID'].isin(train_ids)]
    test_df = df[df['Battery_ID'] == test_id]
    
    print("--- Feature Statistics (Train vs Test) ---")
    for f in features:
        mean_train = train_df[f].mean()
        std_train = train_df[f].std()
        mean_test = test_df[f].mean()
        std_test = test_df[f].std()
        
        print(f"\nFeature: {f}")
        print(f"  Train: Mean={mean_train:.4f}, Std={std_train:.4f}, Range=[{train_df[f].min():.4f}, {train_df[f].max():.4f}]")
        print(f"  Test : Mean={mean_test:.4f}, Std={std_test:.4f}, Range=[{test_df[f].min():.4f}, {test_df[f].max():.4f}]")
        
        # Check Z-score of test mean relative to train
        z_score = (mean_test - mean_train) / std_train
        print(f"  Test Mean Z-Score: {z_score:.2f}")

if __name__ == "__main__":
    analyze_distributions()
