# Protein Fitness Landscape Regression & Enrichment Analysis Pipeline

This repository contains a comprehensive, machine learning-driven pipeline designed to predict protein fitness landscapes and analyze beneficial mutant enrichment from Deep Mutational Scanning (DMS) datasets. 

The workflow integrates standard sequence encodings (One-hot, AAindex PCA, and Georgiev physical parameters) with advanced representations, including Hugging Face ProtT5 protein language model embeddings and Ensemble Evolutionary Scores (ESM models). It systematically benchmarks 11 machine learning regressors across multiple random dataset splits, outputs performance evaluations, and aggregates statistical metrics.

---

## Repository Structure

Below is the directory layout of the repository as configured:

```text
DMS_data/
├── Benchmark/                         # Active workspace for scripts and execution
│   ├── Data_split/
│   │   └── data_split.py              # Partitions dataset based on ESM scores
│   ├── feature_generation/
│   │   ├── extract_prott5.py          # Generates ProtT5 embeddings
│   │   ├── extract_features_train_test.py # Generates traditional features
│   │   └── combine_features.py        # Appends ESM scores to base feature arrays
│   ├── simulation_results_hit_rate_recall/ # Stores simulation outputs
│   │   └── summary/                   # Contains aggregated statistical reports
│   ├── benchmark_models_evaluation.py # Main model training and benchmarking script
│   ├── merge_aggregated_results.py        # Merges statistics over multiple random seeds
│   ├── run_pipeline_for_seeds.sh      # Orchestrates the entire pipeline over seeds
│   └── README.md                      # Repository documentation (This file)
│
├── data1/
│   └── ESM_Socre/                     # Contains raw DMS and ESM score files
│       ├── Beta-lactamase_evopro_esm-score_with_seq.csv
│       ├── brenan_evopro_esm-score_with_seq.csv
│       ├── doud_evopro_esm-score_with_seq.csv
│       └── ... (other dataset CSV files)
│
└── sequence/                          # Contains reference fasta or raw sequences
**Prerequisites and Setup**
1. Python Environment Dependencies
Make sure you have Python 3.8+ installed. You can install the required packages using pip:
code
Bash
pip install numpy pandas scikit-learn xgboost scipy tqdm torch transformers umap-learn matplotlib seaborn
2. External File Requirements
The scripts use relative paths to locate datasets and PCA features. Before running, ensure:
PCA Matrix: A file named pca-19.npy must be placed in the parent directory three levels up relative to feature_generation (i.e., ../../../pca-19.npy).
Running on Other Datasets
The entire pipeline is pre-configured to run on the Beta-lactamase_evopro_esm-score_with_seq.csv dataset.
If you want to run the workflow on any other dataset located in the data1/ESM_Socre/ folder (such as brenan, doud, lee, or zikv datasets):
Open the following scripts in your text editor:
Benchmark/Data_split/data_split.py (file_path variable)
Benchmark/benchmark_models_evaluation.py (FULL_CSV variable)
Benchmark/feature_generation/extract_prott5.py (FULL_CSV variable)
Benchmark/feature_generation/extract_features_train_test.py (FULL_CSV variable)
Locate the path definitions inside these scripts and change the dataset filename from:
Beta-lactamase_evopro_esm-score_with_seq.csv
to your target dataset name, for example:
brenan_evopro_esm-score_with_seq.csv
Execute the pipeline as normal. The steps will automatically process your newly selected target protein.
Detailed Workflow and Scripts
Step 1: Data Splitting (Data_split/data_split.py)
This script partitions the selected dataset into training (N=96) and test sets (remaining variants):
Calculates an ESM Ensemble Score by averaging 5 independent ESM-1v scores.
Selects the top 30% predicted fitness mutations as positive training samples.
Selects the bottom 30% predicted fitness mutations as negative training samples.
Randomly samples the remaining required training capacity from the intermediate population to preserve diversity.
Assigns all unselected variants to the test set.
Step 2: Feature Generation (feature_generation/)
This module constructs diverse numerical representations for the train and test subsets:
extract_prott5.py: Downloads/caches the Rostlab/prot_t5_xl_half_uniref50-enc model, processes the protein sequences, performs mean pooling, and saves ProtT5 embeddings.
extract_features_train_test.py: Computes traditional descriptors:
One-hot Encoding: Categorical representation of amino acids.
AAindex Encoding: Amino acid physicochemical profiles reduced via Principal Component Analysis (PCA).
Georgiev parameters: 19-dimensional physical parameter profiles.
combine_features.py: Horizontally concatenates the base representation matrices with a 16-column matrix of ESM score predictions.
Step 3: Model Evaluation (benchmark_models_evaluation.py)
Trains 11 distinct regressor families on scaled representations and evaluates them against the test set:
Model Pool: Linear, Ridge, Lasso, ElasticNet, MLP Neural Network, Random Forest, XGBoost, KNN, Gaussian Process, PLS Regression, SVR, and Extra Trees.
Metrics Calculated: Spearman correlation coefficient, Mean Squared Error (MSE), and Coefficient of Determination (R2).
Enrichment Tracking: Simulates laboratory screening by selecting the top 20 predicted variants, counting the verified beneficial mutants (Hit Rate), and exporting enrichment curves coordinates (Recall).
Step 4: Batch Orchestration (run_pipeline_for_seeds.sh)
An automated master bash script that executes the complete pipeline end-to-end for a list of random seeds (e.g., 1, 42, 100). To prevent outputs from overwriting each other, it dynamically:
Rewrites random_seed, top_ratio, and bottom_ratio parameters inside data_split.py.
Invokes dataset splitting, feature extraction, and benchmark training.
Automatically renames and archives output files in simulation_results_hit_rate_recall/ using a unique seed-specific prefix (e.g., seed1-0.3-0.3-).
Step 5: Statistical Merging (merge_aggregated_results.py)
After multi-seed simulations finish, this script parses all files matching patterns like seed*-benchmark_results.csv, compiles results, and exports consolidated reports into simulation_results_hit_rate_recall/summary/:
enrichment_curves.csv: Aggregated mean and standard deviation of Hit Rate and Recall at every top percentile increment (Top 1% to 100%).
core_metrics.csv: Compiled mean and standard deviation of regression metrics (Spearman, MSE, R2) and top 20 variants metrics across all seeds, sorted descending by mean Spearman score.
raw_predictions_distribution.csv: Combined mean, standard deviation, and count of model predictions per variant across different dataset splits (useful for plotting true vs. predicted scatter plots and evaluating uncertainty).
How to Run the Pipeline
Automated Batch Execution (Recommended)
You can run the entire workflow over multiple seeds with a single terminal command:
code
Bash
chmod +x run_pipeline_for_seeds.sh
./run_pipeline_for_seeds.sh
This script will sequentially partition, featurize, evaluate, and subsequently aggregate the results automatically.
Manual Step-by-Step Execution
If you prefer running individual tasks manually:
Partition the Dataset:
code
Bash
cd Data_split
python data_split.py
cd ..
Generate Features:
code
Bash
cd feature_generation
python extract_prott5.py
python extract_features_train_test.py
python combine_features.py
cd ..
Train and Benchmark Models:
code
Bash
python benchmark_models_evaluation.py
Merge Multi-Seed Outputs:
code
Bash
python merge_aggregated_results.py
Outputs Description
After running the pipeline, the final aggregated metrics are located under simulation_results_hit_rate_recall/summary/:
core_metrics.csv: Contains benchmarking scores of different combinations (e.g., prott5_with_esm_XGBoost), allowing you to easily identify which model-feature combination yields the highest Spearman correlation.
enrichment_curves.csv: Provides data points showing how fast beneficial variants are enriched as you screen a certain top percentage of predicted candidates.
raw_predictions_distribution.csv: Houses detailed metrics on prediction means and standard deviations, which can be plotted to visualize predictions or track regression uncertainties.
