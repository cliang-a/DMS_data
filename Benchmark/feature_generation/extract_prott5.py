import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

# Set Hugging Face mirror endpoint for connection reliability if needed
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import T5EncoderModel, T5Tokenizer

########################################
# Configuration & Paths
########################################
current_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Define input file paths
FULL_CSV = os.path.join(current_dir, "../../Beta-lactamase_evopro_esm-score_with_seq.csv")
TRAIN_CSV = os.path.join(current_dir, "../data_split/train_set_N96.csv")
TEST_CSV = os.path.join(current_dir, "../data_split/test_set_Remaining.csv")

# 2. Define path for local model cache
model_path = os.path.abspath(os.path.join(current_dir, "../../../models"))
os.makedirs(model_path, exist_ok=True)

# 3. Save location for full sequence features
FULL_DIR = os.path.abspath(os.path.join(current_dir, "../../"))
FULL_NPY_PATH = os.path.join(FULL_DIR, "full_prott5.npy")
FULL_CSV_OUT = os.path.join(FULL_DIR, "full_prott5.csv")

# 4. Output directory for extracted training/testing subset features
OUTPUT_DIR = os.path.join(current_dir, "features")
os.makedirs(OUTPUT_DIR, exist_ok=True)

########################################
# Device Selection
########################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

########################################
# 1. Load Full Sequences
########################################
print(f"Loading full sequences from: {FULL_CSV}")
full_df = pd.read_csv(FULL_CSV)
# Assumes the target sequence column name is "mutated_sequence"
# If the column name differs in your dataset, modify this key accordingly (e.g., "Sequence")
full_seqs = full_df["mutated_sequence"].tolist()
print(f"Total full sequences: {len(full_seqs)}")

########################################
# 2. Generate or Load Full Embeddings
########################################
if os.path.exists(FULL_NPY_PATH):
    print(f"Found existing full embeddings at {FULL_NPY_PATH}. Loading directly...")
    full_embeddings = np.load(FULL_NPY_PATH)
else:
    print("Generating full embeddings from scratch...")
    # --- Load model ---
    MODEL_NAME = "Rostlab/prot_t5_xl_half_uniref50-enc"
    print(f"Loading model: {MODEL_NAME} from {model_path}")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False, cache_dir=model_path)
    model = T5EncoderModel.from_pretrained(MODEL_NAME, cache_dir=model_path)

    if device == torch.device("cpu"):
        model.to(torch.float32)
    model = model.to(device)
    model.eval()

    # --- Batch embedding ---
    MAX_BATCH = 8
    full_embeddings_list = []
    start = time.time()

    for i in tqdm(range(0, len(full_seqs), MAX_BATCH), desc="Embedding Full Data"):
        batch_seqs = full_seqs[i:i + MAX_BATCH]

        # ProtT5 preprocessing
        batch_seqs_processed = [
            " ".join(list(seq.replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")))
            for seq in batch_seqs
        ]

        # Tokenize
        token_encoding = tokenizer.batch_encode_plus(
            batch_seqs_processed,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt"
        )
        input_ids = token_encoding["input_ids"].to(device)
        attention_mask = token_encoding["attention_mask"].to(device)

        # Forward
        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

        # Mean pooling
        for batch_idx, seq in enumerate(batch_seqs_processed):
            seq_len = len(seq.split())
            emb = embedding_repr.last_hidden_state[batch_idx, :seq_len]
            emb = emb.mean(dim=0)
            full_embeddings_list.append(emb.detach().cpu().numpy())

    full_embeddings = np.array(full_embeddings_list)
    print("Full Embedding shape:", full_embeddings.shape)

    # Save full features
    np.save(FULL_NPY_PATH, full_embeddings)
    pd.DataFrame(full_embeddings).to_csv(FULL_CSV_OUT, index=False)

    end = time.time()
    print(f"Finished embedding full sequences in {end - start:.2f} seconds")
    print(f"Saved full numpy features to: {FULL_NPY_PATH}")

########################################
# 3. Create Mapping Dictionary
########################################
# Pair each sequence with its calculated embedding vector for fast lookup and extraction
print("Creating sequence to embedding mapping...")
seq2emb = dict(zip(full_seqs, full_embeddings))


########################################
# 4. Extract Features for Train and Test
########################################
def extract_and_save_subset(csv_path, subset_name):
    print(f"--- Extracting features for {subset_name} ---")
    df = pd.read_csv(csv_path)
    subset_seqs = df["mutated_sequence"].tolist()

    subset_embeddings = []
    missing_count = 0

    for seq in subset_seqs:
        if seq in seq2emb:
            subset_embeddings.append(seq2emb[seq])
        else:
            # If a sequence is missing from the full set, print a warning and pad with a zero vector
            print(f"Warning: Sequence not found in full data! \n{seq[:50]}...")
            subset_embeddings.append(np.zeros_like(full_embeddings[0]))
            missing_count += 1

    subset_embeddings = np.array(subset_embeddings)
    print(f"{subset_name} Embedding shape: {subset_embeddings.shape}")
    if missing_count > 0:
        print(f"Warning: {missing_count} sequences were missing from the full dataset.")

    # Save the extracted subset features into the features directory
    npy_path = os.path.join(OUTPUT_DIR, f"{subset_name}_prott5.npy")
    csv_path_out = os.path.join(OUTPUT_DIR, f"{subset_name}_prott5.csv")

    np.save(npy_path, subset_embeddings)
    pd.DataFrame(subset_embeddings).to_csv(csv_path_out, index=False)

    print(f"Saved {subset_name} features to:\n  {npy_path}\n  {csv_path_out}\n")


# Perform feature extraction for both subsets
extract_and_save_subset(TRAIN_CSV, "train")
extract_and_save_subset(TEST_CSV, "test")

print("All tasks completed successfully!")