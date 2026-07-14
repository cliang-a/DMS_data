import pandas as pd
import numpy as np

# 1. Set file path and random seed (for reproducibility)
file_path = "../../Beta-lactamase_evopro_esm-score_with_seq.csv"
random_seed = 100
np.random.seed(random_seed)

# 2. Load the dataset
df = pd.read_csv(file_path)

# 3. Compute the ensemble score by averaging 5 ESM-1v models
esm1v_cols = [
    'esm1v_t33_650M_UR90S_1',
    'esm1v_t33_650M_UR90S_2',
    'esm1v_t33_650M_UR90S_3',
    'esm1v_t33_650M_UR90S_4',
    'esm1v_t33_650M_UR90S_5'
]
# Create a new column to store the ensemble score, which serves as the guide for sampling
df['esm1v_ensemble_score'] = df[esm1v_cols].mean(axis=1)

# ================= Dataset Split Configuration =================
N = 96                          # Total size of the training set
n_top = int(N * 0.3)            # Top 30% of samples (approx. 28)
n_bottom = int(N * 0.3)         # Bottom 30% of samples (approx. 28)
n_random = N - n_top - n_bottom # Remaining samples selected randomly (approx. 40)

# 4. Sort the dataset by the ensemble score in descending order
# (assuming higher scores imply better predicted mutation activity)
df_sorted = df.sort_values(by='esm1v_ensemble_score', ascending=False)

# 5. Extract top and bottom samples
# Select the highest-scoring n_top mutations
top_samples = df_sorted.head(n_top)
# Select the lowest-scoring n_bottom mutations
bottom_samples = df_sorted.tail(n_bottom)

# 6. Randomly sample n_random from the remaining data to ensure diversity
# Exclude the already selected top and bottom samples from the pool first
remaining_df = df.drop(top_samples.index).drop(bottom_samples.index)
random_samples = remaining_df.sample(n=n_random, random_state=random_seed)

# 7. Concatenate the three subsets to form the training set
train_df = pd.concat([top_samples, bottom_samples, random_samples])

# 8. Assign all remaining unselected data to the test set
test_df = remaining_df.drop(random_samples.index)

# Shuffle the training set to prevent any ordering bias during model training
train_df = train_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# 9. Print results to verify the split configuration
print(f"Total dataset size: {len(df)}")
print(f"Training set size: {len(train_df)}")
print(f" - Top samples count: {len(top_samples)}")
print(f" - Bottom samples count: {len(bottom_samples)}")
print(f" - Random samples count: {len(random_samples)}")
print(f"Test set size (remaining): {len(test_df)}")

# (Optional) Save the split datasets as CSV files for subsequent steps
train_df.to_csv("./train_set_N96.csv", index=False)
test_df.to_csv("./test_set_Remaining.csv", index=False)