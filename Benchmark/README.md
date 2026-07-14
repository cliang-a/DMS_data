# Protein Fitness Landscape Regression & Enrichment Analysis Pipeline

This repository contains a comprehensive, machine learning-driven pipeline designed to predict protein fitness landscapes and analyze beneficial mutant enrichment from Deep Mutational Scanning (DMS) datasets.

The workflow integrates standard sequence encodings (One-hot, AAindex PCA, and Georgiev physical parameters) with advanced representations, including Hugging Face ProtT5 protein language model embeddings and Ensemble Evolutionary Scores (ESM models). It systematically benchmarks 11 machine learning regressors across multiple random dataset splits, outputs performance evaluations, and aggregates statistical metrics.

---

## 📁 Repository Structure

Below is the directory layout of the repository as configured:

```text
DMS_data/
├── Benchmark/                         # Active workspace for scripts and execution
│   ├── Data_split/
│   │   └── data_split.py                  # Partitions dataset based on ESM scores
│   ├── feature_generation/
│   │   ├── extract_prott5.py              # Generates ProtT5 embeddings
│   │   ├── extract_features_train_test.py # Generates traditional features
│   │   └── combine_features.py            # Appends ESM scores to base feature arrays
│   ├── simulation_results_hit_rate_recall/# Stores simulation outputs
│   │   └── summary/                       # Contains aggregated statistical reports
│   ├── benchmark_models_evaluation.py     # Main model training and benchmarking script
│   ├── merge_aggregated_results.py        # Merges statistics over multiple random seeds
│   ├── run_pipeline_for_seeds.sh          # Orchestrates the entire pipeline over seeds
│   └── README.md                          # Repository documentation (This file)
│
├── data1/
│   └── ESM_Socre/                         # Contains raw DMS and ESM score files
│       ├── Beta-lactamase_evopro_esm-score_with_seq.csv
│       ├── brenan_evopro_esm-score_with_seq.csv
│       ├── doud_evopro_esm-score_with_seq.csv
│       └── ... (other dataset CSV files)
│
└── sequence/                          # Contains reference fasta or raw sequences
