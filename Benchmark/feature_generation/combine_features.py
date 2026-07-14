import os
import numpy as np
import pandas as pd

# ================= 1. Define list of 16 ESM score columns to extract =================
esm_cols = [
    'esm1v_t33_650M_UR90S_1', 'esm1v_t33_650M_UR90S_2', 'esm1v_t33_650M_UR90S_3',
    'esm1v_t33_650M_UR90S_4', 'esm1v_t33_650M_UR90S_5', 'esm2_t33_650M_UR50D',
    'esm2_t36_3B_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t12_35M_UR50D',
    'esm2_t6_8M_UR50D', 'esm1_t6_43M_UR50S', 'esm1_t34_670M_UR50D',
    'esm1_t34_670M_UR50S', 'esm1_t34_670M_UR100', 'esm1_t12_85M_UR50S',
    'esm1b_t33_650M_UR50S'
]

# Four base features to be processed
feature_names = ["aaindex", "onehot", "prott5", "georgiev"]


# ================= 2. Concatenation Function Definition =================
def append_esm_to_features(csv_path, split_prefix):
    """
    csv_path: Path to the corresponding CSV file (e.g., "train_set_N96.csv").
    split_prefix: Naming prefix for the .npy files (e.g., "train" or "test").
    """
    print(f"\n[{split_prefix.upper()}] Processing split {csv_path}...")

    # 1. Read CSV and extract the 16 ESM score columns
    df = pd.read_csv(csv_path)
    # Extract ESM scores and convert to a float32 numpy array, Shape: (N, 16)
    esm_features = df[esm_cols].values.astype(np.float32)
    print(f"  - Extracted ESM features successfully, shape: {esm_features.shape}")

    # 2. Iterate through the four base features and perform horizontal concatenation
    for feat in feature_names:
        feat_path = f"features/{split_prefix}_{feat}.npy"

        # Ensure the base feature file exists
        if not os.path.exists(feat_path):
            print(f"  ⚠️ Warning: File {feat_path} not found. Skipping feature.")
            continue

        # Load the base feature matrix, Shape: (N, D)
        base_features = np.load(feat_path)

        # Use np.hstack to concatenate (N, D) and (N, 16) into (N, D + 16)
        combined_features = np.hstack((base_features, esm_features))

        # Save the combined feature matrix
        output_path = f"features/{split_prefix}_{feat}_with_esm.npy"
        np.save(output_path, combined_features)

        print(f"  ✅ {feat:8s} + ESM -> Saved to: {output_path} | New shape: {combined_features.shape}")


# ================= 3. Execution =================
if __name__ == "__main__":
    # Ensure target directory exists
    os.makedirs("features", exist_ok=True)

    # Process training split
    # Adjust paths based on your local directory structure if necessary
    append_esm_to_features("./../data_split/train_set_N96.csv", "train")

    # Process testing split
    append_esm_to_features("./../data_split/test_set_Remaining.csv", "test")

    print("\n🎉 Concatenation of features and ESM scores completed successfully!")