# Protein Fitness Landscape Regression & Enrichment Analysis Pipeline

This repository contains a comprehensive, machine learning-driven pipeline designed to predict protein fitness landscapes and analyze beneficial mutant enrichment from Deep Mutational Scanning (DMS) datasets (e.g., Beta-lactamase). 

The workflow integrates standard sequence encodings (One-hot, AAindex PCA, and Georgiev physical parameters) with advanced representations, including Hugging Face ProtT5 protein language model embeddings and Ensemble Evolutionary Scores (ESM models). It systematically benchmarks 11 machine learning regressors across multiple random dataset splits, outputs performance evaluations, and aggregates statistical metrics.

---

## 📂 Repository Structure

Below is the directory layout of the repository as configured:

```text
├── Data_split/
│   └── data_split.py                  # Partitions dataset based on ESM scores and random sampling
├── feature_generation/
│   ├── extract_prott5.py              # Generates ProtT5 embeddings using PyTorch & Hugging Face
│   ├── extract_features_train_test.py # Generates traditional features (One-hot, AAindex, Georgiev)
│   └── combine_features.py            # Appends 16 ESM scores to base feature arrays
├── simulation_results_hit_rate_recall/# Stores simulation outputs, evaluation metrics, and aggregates
│   └── summary/                       # Contains aggregated statistical reports over all seeds
├── benchmark_models_evaluation.py     # Main model training and benchmarking script (11 regressors)
├── merge_aggregated_results.py        # Merges and compiles statistics over multiple random seeds
├── run_pipeline_for_seeds.sh          # Orchestrates the entire pipeline over a series of seeds
└── README.md                          # Repository documentation

🛠️ Prerequisites & Setup
1. Python Environment Dependencies
Ensure you have Python 3.8+ installed. You can install all required packages using pip:

Bash
pip install numpy pandas scikit-learn xgboost scipy tqdm torch transformers umap-learn matplotlib seaborn
2. Required External Files
The pipeline relies on relative paths to locate specific dataset and PCA feature files. Ensure the following files are placed in their respective locations relative to this repository:

Input Dataset: ../../Beta-lactamase_evopro_esm-score_with_seq.csv

Description: Contains sequence entries and corresponding target fitness labels.

PCA Matrix: ../../../pca-19.npy

Description: Required by the AAindex encoding algorithm inside the feature_generation/ module.

🔄 Detailed Workflow & Scripts
Step 1: Data Splitting (Data_split/data_split.py)
This script partitions the original dataset into training (N=96) and test sets (remaining variants) under guided criteria:

Calculates an ESM Ensemble Score by averaging 5 independent ESM-1v scores.

Selects the top 30% predicted fitness mutations (n_top) as positive training samples.

Selects the bottom 30% predicted fitness mutations (n_bottom) as negative training samples.

Randomly samples the remaining required training capacity (n_random) from the intermediate population to preserve manifold diversity.

Assigns all unselected variants to the test set.

Step 2: Feature Generation (feature_generation/)
This module constructs diverse numerical representations for the train and test subsets:

extract_prott5.py: Downloads/caches the Rostlab/prot_t5_xl_half_uniref50-enc model, processes the protein sequences, performs mean pooling on the last hidden state, and saves ProtT5 embeddings.

extract_features_train_test.py: Computes traditional descriptors:

One-hot Encoding: Categorical representation of amino acids.

AAindex Encoding: Amino acid physicochemical profiles reduced via Principal Component Analysis (PCA).

Georgiev parameters: 19-dimensional multidimensional physical parameter profiles.

combine_features.py: Horizontally concatenates the base representation matrices (One-hot, AAindex, ProtT5, Georgiev) with a 16-column matrix of ESM score predictions.

Step 3: Model Evaluation (benchmark_models_evaluation.py)
Trains 11 distinct regressor families on scaled representations and evaluates them against the test set:

Model Pool: Linear, Ridge, Lasso, ElasticNet, MLP Neural Network, Random Forest, XGBoost, KNN, Gaussian Process, PLS Regression, SVR, and Extra Trees.

Metrics Calculated: Spearman correlation coefficient (ρ), Mean Squared Error (MSE), and Coefficient of Determination (R²).

Enrichment Tracking: Simulates laboratory screening by selecting the top 20 predicted variants, counting the verified beneficial mutants (Hit Rate), and exporting enrichment curves coordinates (Recall).

Step 4: Batch Orchestration (run_pipeline_for_seeds.sh)
An automated master bash script that executes the complete pipeline end-to-end for a list of random seeds (e.g., 1, 42, 100). To prevent outputs from overwriting each other, it dynamically:

Rewrites random_seed, n_top, and n_bottom inside data_split.py using sed.

Invokes dataset splitting, feature extraction, and benchmark training.

Automatically renames and archives output files in simulation_results_hit_rate_recall/ using a unique seed-specific prefix (e.g., seed1-0.3-0.3-).

Step 5: Statistical Merging (merge_aggregated_results.py)
After multi-seed simulations finish, this script parses all files matching patterns like seed*-benchmark_results.csv, compiles results, and exports consolidated reports into simulation_results_hit_rate_recall/summary/:

enrichment_curves.csv: Aggregated mean and standard deviation of Hit Rate and Recall at every top percentile increment (Top 1% to 100%).

core_metrics.csv: Compiled mean and standard deviation of regression metrics (Spearman, MSE, R²) and top 20 variants metrics across all seeds, sorted descending by mean Spearman score.

raw_predictions_distribution.csv: Combined mean, standard deviation, and count of model predictions per variant across different dataset splits (useful for plotting true vs. predicted scatter plots and evaluating uncertainty).
