import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

########################################
# Configuration & Paths
########################################
current_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Define input file paths (Beta-lactamase dataset and data splits)
FULL_CSV = os.path.abspath(os.path.join(current_dir, "../../Beta-lactamase_evopro_esm-score_with_seq.csv"))
TRAIN_CSV = os.path.abspath(os.path.join(current_dir, "../data_split/train_set_N96.csv"))
TEST_CSV = os.path.abspath(os.path.join(current_dir, "../data_split/test_set_Remaining.csv"))

# 2. Paths for storing full sequence feature arrays
FULL_DIR = os.path.abspath(os.path.join(current_dir, "../../"))
FULL_GEORGIEV_PATH = os.path.join(FULL_DIR, "full_georgiev.npy")
FULL_AAINDEX_PATH = os.path.join(FULL_DIR, "full_aaindex.npy")
FULL_ONEHOT_PATH = os.path.join(FULL_DIR, "full_onehot.npy")

# 3. Output directory for the extracted train/test features
OUTPUT_DIR = os.path.join(current_dir, "features")
os.makedirs(OUTPUT_DIR, exist_ok=True)

########################################
# Load Full Sequences
########################################
print(f"Loading full sequences from: {FULL_CSV}")
full_df = pd.read_csv(FULL_CSV)
full_seqs = full_df["mutated_sequence"].tolist()
print(f"Total full sequences: {len(full_seqs)}")
print(f"Sequence length: {len(full_seqs[0])}")

########################################
# Georgiev Parameters & Functions
########################################
gg_1 = {'Q': -2.54, 'L': 2.72, 'T': -0.65, 'C': 2.66, 'I': 3.1, 'G': 0.15, 'V': 2.64, 'K': -3.89, 'M': 1.89, 'F': 3.12,
        'N': -2.02, 'R': -2.8, 'H': -0.39, 'E': -3.08, 'W': 1.89, 'A': 0.57, 'D': -2.46, 'Y': 0.79, 'S': -1.1,
        'P': -0.58, '*': 0}
gg_2 = {'Q': 1.82, 'L': 1.88, 'T': -1.6, 'C': -1.52, 'I': 0.37, 'G': -3.49, 'V': 0.03, 'K': 1.47, 'M': 3.88, 'F': 0.68,
        'N': -1.92, 'R': 0.31, 'H': 1, 'E': 3.45, 'W': -0.09, 'A': 3.37, 'D': -0.66, 'Y': -2.62, 'S': -2.05, 'P': -4.33,
        '*': 0}
gg_3 = {'Q': -0.82, 'L': 1.92, 'T': -1.39, 'C': -3.29, 'I': 0.26, 'G': -2.97, 'V': -0.67, 'K': 1.95, 'M': -1.57,
        'F': 2.4, 'N': 0.04, 'R': 2.84, 'H': -0.63, 'E': 0.05, 'W': 4.21, 'A': -3.66, 'D': -0.57, 'Y': 4.11, 'S': -2.19,
        'P': -0.02, '*': 0}
gg_4 = {'Q': -1.85, 'L': 5.33, 'T': 0.63, 'C': -3.77, 'I': 1.04, 'G': 2.06, 'V': 2.34, 'K': 1.17, 'M': -3.58,
        'F': -0.35, 'N': -0.65, 'R': 0.25, 'H': -3.49, 'E': 0.62, 'W': -2.77, 'A': 2.34, 'D': 0.14, 'Y': -0.63,
        'S': 1.36, 'P': -0.21, '*': 0}
gg_5 = {'Q': 0.09, 'L': 0.08, 'T': 1.35, 'C': 2.96, 'I': -0.05, 'G': 0.7, 'V': 0.64, 'K': 0.53, 'M': -2.55, 'F': -0.88,
        'N': 1.61, 'R': 0.2, 'H': 0.05, 'E': -0.49, 'W': 0.72, 'A': -1.07, 'D': 0.75, 'Y': 1.89, 'S': 1.78, 'P': -8.31,
        '*': 0}
gg_6 = {'Q': 0.6, 'L': 0.09, 'T': -2.45, 'C': -2.23, 'I': -1.18, 'G': 7.47, 'V': -2.01, 'K': 0.1, 'M': 2.07, 'F': 1.62,
        'N': 2.08, 'R': -0.37, 'H': 0.41, 'E': 0, 'W': 0.86, 'A': -0.4, 'D': 0.24, 'Y': -0.53, 'S': -3.36, 'P': -1.82,
        '*': 0}
gg_7 = {'Q': 0.25, 'L': 0.27, 'T': -0.65, 'C': 0.44, 'I': -0.21, 'G': 0.41, 'V': -0.33, 'K': 4.01, 'M': 0.84,
        'F': -0.15, 'N': 0.4, 'R': 3.81, 'H': 1.61, 'E': -5.66, 'W': -1.07, 'A': 1.23, 'D': -5.15, 'Y': -1.3, 'S': 1.39,
        'P': -0.12, '*': 0}
gg_8 = {'Q': 2.11, 'L': -4.06, 'T': 3.43, 'C': -3.49, 'I': 3.45, 'G': 1.62, 'V': 3.93, 'K': -0.01, 'M': 1.85,
        'F': -0.41, 'N': -2.47, 'R': 0.98, 'H': -0.6, 'E': -0.11, 'W': -1.66, 'A': -2.32, 'D': -1.17, 'Y': 1.31,
        'S': -1.21, 'P': -1.18, '*': 0}
gg_9 = {'Q': -1.92, 'L': 0.43, 'T': 0.34, 'C': 2.22, 'I': 0.86, 'G': -0.47, 'V': -0.21, 'K': -0.26, 'M': -2.05,
        'F': 4.2, 'N': -0.07, 'R': 2.43, 'H': 3.55, 'E': 1.49, 'W': -5.87, 'A': -2.01, 'D': 0.73, 'Y': -0.56,
        'S': -2.83, 'P': 0, '*': 0}
gg_10 = {'Q': -1.67, 'L': -1.2, 'T': 0.24, 'C': -3.78, 'I': 1.98, 'G': -2.9, 'V': 1.27, 'K': -1.66, 'M': 0.78,
         'F': 0.73, 'N': 7.02, 'R': -0.99, 'H': 1.52, 'E': -2.26, 'W': -0.66, 'A': 1.31, 'D': 1.5, 'Y': -0.95,
         'S': 0.39, 'P': -0.66, '*': 0}
gg_11 = {'Q': 0.7, 'L': 0.67, 'T': -0.53, 'C': 1.98, 'I': 0.89, 'G': -0.98, 'V': 0.43, 'K': 5.86, 'M': 1.53, 'F': -0.56,
         'N': 1.32, 'R': -4.9, 'H': -2.28, 'E': -1.62, 'W': -2.49, 'A': -1.14, 'D': 1.51, 'Y': 1.91, 'S': -2.92,
         'P': 0.64, '*': 0}
gg_12 = {'Q': -0.27, 'L': -0.29, 'T': 1.91, 'C': -0.43, 'I': -1.67, 'G': -0.62, 'V': -1.71, 'K': -0.06, 'M': 2.44,
         'F': 3.54, 'N': -2.44, 'R': 2.09, 'H': -3.12, 'E': -3.97, 'W': -0.3, 'A': 0.19, 'D': 5.61, 'Y': -1.26,
         'S': 1.27, 'P': -0.92, '*': 0}
gg_13 = {'Q': -0.99, 'L': -2.47, 'T': 2.66, 'C': -1.03, 'I': -1.02, 'G': -0.11, 'V': -2.93, 'K': 1.38, 'M': -0.26,
         'F': 5.25, 'N': 0.37, 'R': -3.08, 'H': -1.45, 'E': 2.3, 'W': -0.5, 'A': 1.66, 'D': -3.85, 'Y': 1.57, 'S': 2.86,
         'P': -0.37, '*': 0}
gg_14 = {'Q': -1.56, 'L': -4.79, 'T': -3.07, 'C': 0.93, 'I': -1.21, 'G': 0.15, 'V': 4.22, 'K': 1.78, 'M': -3.09,
         'F': 1.73, 'N': -0.89, 'R': 0.82, 'H': -0.77, 'E': -0.06, 'W': 1.64, 'A': 4.39, 'D': 1.28, 'Y': 0.2,
         'S': -1.88, 'P': 0.17, '*': 0}
gg_15 = {'Q': 6.22, 'L': 0.8, 'T': 0.2, 'C': 1.43, 'I': -1.78, 'G': -0.53, 'V': 1.06, 'K': -2.71, 'M': -1.39, 'F': 2.14,
         'N': 3.13, 'R': 1.32, 'H': -4.18, 'E': -0.35, 'W': -0.72, 'A': 0.18, 'D': -1.98, 'Y': -0.76, 'S': -2.42,
         'P': 0.36, '*': 0}
gg_16 = {'Q': -0.18, 'L': -1.43, 'T': -2.2, 'C': 1.45, 'I': 5.71, 'G': 0.35, 'V': -1.31, 'K': 1.62, 'M': -1.02,
         'F': 1.1, 'N': 0.79, 'R': 0.69, 'H': -2.91, 'E': 1.51, 'W': 1.75, 'A': -2.6, 'D': 0.05, 'Y': -5.19, 'S': 1.75,
         'P': 0.08, '*': 0}
gg_17 = {'Q': 2.72, 'L': 0.63, 'T': 3.73, 'C': -1.15, 'I': 1.54, 'G': 0.3, 'V': -1.97, 'K': 0.96, 'M': -4.32, 'F': 0.68,
         'N': -1.54, 'R': -2.62, 'H': 3.37, 'E': -2.29, 'W': 2.73, 'A': 1.49, 'D': 0.9, 'Y': -2.56, 'S': -2.77,
         'P': 0.16, '*': 0}
gg_18 = {'Q': 4.35, 'L': -0.24, 'T': -5.46, 'C': -1.64, 'I': 2.11, 'G': 0.32, 'V': -1.21, 'K': -1.09, 'M': -1.34,
         'F': 1.46, 'N': -1.71, 'R': -1.49, 'H': 1.87, 'E': -1.47, 'W': -2.2, 'A': 0.46, 'D': 1.38, 'Y': 2.87,
         'S': 3.36, 'P': -0.34, '*': 0}
gg_19 = {'Q': 0.92, 'L': 1.01, 'T': -0.73, 'C': -1.05, 'I': -4.18, 'G': 0.05, 'V': 4.77, 'K': 1.36, 'M': 0.09,
         'F': 2.33, 'N': -0.25, 'R': -2.57, 'H': 2.17, 'E': 0.15, 'W': 0.9, 'A': -4.22, 'D': -0.03, 'Y': -3.43,
         'S': 2.67, 'P': 0.04, '*': 0}

georgiev_parameters = [gg_1, gg_2, gg_3, gg_4, gg_5, gg_6, gg_7, gg_8, gg_9, gg_10, gg_11, gg_12, gg_13, gg_14, gg_15,
                       gg_16, gg_17, gg_18, gg_19]


def get_georgiev_params_for_aa(aa):
    return [gg[aa] for gg in georgiev_parameters]


def get_georgiev_params_for_seq(seq):
    return np.array([get_georgiev_params_for_aa(aa) for aa in seq])


def seqs_to_georgiev(seqs):
    return np.stack([get_georgiev_params_for_seq(seq) for seq in seqs])


########################################
# AAindex encoding functions
########################################
def featurize_aa_idx(seqs):
    CHARS = ["*", "A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
    C2I_MAPPING = {c: i for i, c in enumerate(CHARS)}

    # Absolute path pointing to pca-19.npy
    pca_path = os.path.join(current_dir, "../../../pca-19.npy")
    aa_features = np.load(pca_path)
    aa_features = np.insert(aa_features, 0, np.zeros(aa_features.shape[1]), axis=0)

    X = []
    for seq in seqs:
        seq_feat = []
        for aa in seq:
            seq_feat.append(aa_features[C2I_MAPPING[aa]])
        X.append(np.array(seq_feat))
    return np.stack(X)


########################################
# One-hot encoding functions
########################################
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


def onehot_encode(seq):
    feat = np.zeros((len(seq), 20))
    for i, aa in enumerate(seq):
        if aa in AA_TO_IDX:
            feat[i, AA_TO_IDX[aa]] = 1
    return feat


########################################
# 1. Generate/Load Full Georgiev Features
########################################
print("\n--- Processing Full Georgiev features ---")
if os.path.exists(FULL_GEORGIEV_PATH):
    print("Loading existing full Georgiev features...")
    full_georgiev = np.load(FULL_GEORGIEV_PATH)
else:
    print("Generating full Georgiev features...")
    full_georgiev = seqs_to_georgiev(full_seqs)
    full_georgiev = full_georgiev.reshape(full_georgiev.shape[0], -1)
    np.save(FULL_GEORGIEV_PATH, full_georgiev)
    print("Saved full Georgiev features.")

########################################
# 2. Generate/Load Full AAindex Features
########################################
print("\n--- Processing Full AAindex features ---")
if os.path.exists(FULL_AAINDEX_PATH):
    print("Loading existing full AAindex features...")
    full_aaindex = np.load(FULL_AAINDEX_PATH)
else:
    print("Generating full AAindex features...")
    full_aaindex = featurize_aa_idx(full_seqs)
    full_aaindex = full_aaindex.reshape(full_aaindex.shape[0], -1)
    np.save(FULL_AAINDEX_PATH, full_aaindex)
    print("Saved full AAindex features.")

########################################
# 3. Generate/Load Full One-hot Features
########################################
print("\n--- Processing Full One-hot features ---")
if os.path.exists(FULL_ONEHOT_PATH):
    print("Loading existing full One-hot features...")
    full_onehot = np.load(FULL_ONEHOT_PATH)
else:
    print("Generating full One-hot features...")
    full_onehot = np.stack([onehot_encode(seq) for seq in tqdm(full_seqs)])
    full_onehot = full_onehot.reshape(full_onehot.shape[0], -1)
    np.save(FULL_ONEHOT_PATH, full_onehot)
    print("Saved full One-hot features.")

########################################
# Build Mappings for Fast Extraction
########################################
print("\nCreating sequence to feature mappings...")
seq2georgiev = dict(zip(full_seqs, full_georgiev))
seq2aaindex = dict(zip(full_seqs, full_aaindex))
seq2onehot = dict(zip(full_seqs, full_onehot))


########################################
# Extract Features for Train and Test
########################################
def extract_and_save_subset(csv_path, subset_name):
    print(f"\n--- Extracting features for {subset_name} ---")
    df = pd.read_csv(csv_path)
    subset_seqs = df["mutated_sequence"].tolist()

    georgiev_list = []
    aaindex_list = []
    onehot_list = []
    missing_count = 0

    for seq in subset_seqs:
        if seq in seq2georgiev:
            georgiev_list.append(seq2georgiev[seq])
            aaindex_list.append(seq2aaindex[seq])
            onehot_list.append(seq2onehot[seq])
        else:
            missing_count += 1
            print(f"Warning: Sequence not found in full data! \n{seq[:50]}...")
            georgiev_list.append(np.zeros_like(full_georgiev[0]))
            aaindex_list.append(np.zeros_like(full_aaindex[0]))
            onehot_list.append(np.zeros_like(full_onehot[0]))

    if missing_count > 0:
        print(f"Warning: {missing_count} sequences were missing from the full dataset.")

    X_georgiev = np.array(georgiev_list)
    X_aaindex = np.array(aaindex_list)
    X_onehot = np.array(onehot_list)

    print(f"{subset_name} Georgiev shape: {X_georgiev.shape}")
    print(f"{subset_name} AAindex shape: {X_aaindex.shape}")
    print(f"{subset_name} One-hot shape: {X_onehot.shape}")

    # Save the numpy arrays
    np.save(os.path.join(OUTPUT_DIR, f"{subset_name}_georgiev.npy"), X_georgiev)
    np.save(os.path.join(OUTPUT_DIR, f"{subset_name}_aaindex.npy"), X_aaindex)
    np.save(os.path.join(OUTPUT_DIR, f"{subset_name}_onehot.npy"), X_onehot)

    # Save sequences to CSV for reference
    pd.DataFrame({"sequence": subset_seqs}).to_csv(
        os.path.join(OUTPUT_DIR, f"{subset_name}_sequences.csv"), index=False
    )
    print(f"Features and sequences for '{subset_name}' saved successfully in {OUTPUT_DIR}/")


# Perform feature extraction for both subsets
extract_and_save_subset(TRAIN_CSV, "train")
extract_and_save_subset(TEST_CSV, "test")

print("\nAll tasks completed successfully!")