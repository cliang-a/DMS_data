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
