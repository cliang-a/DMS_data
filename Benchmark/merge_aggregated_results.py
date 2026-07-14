import os
import glob
import pandas as pd


def merge_results():
    # Base folder path for input data
    base_output_dir = "simulation_results_hit_rate_recall"

    if not os.path.exists(base_output_dir):
        print(f"❌ Cannot find directory {base_output_dir}, aggregation aborted.")
        return

    # ----- Create a "summary" subdirectory -----
    summary_dir = os.path.join(base_output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    print(f"📊 Starting aggregation of multi-seed simulation data in {base_output_dir}...")
    print(f"📂 Merged final results will be saved to: {summary_dir}")

    # ==========================================
    # Task 1: Aggregate Enrichment Curve Data
    # Logic: Align by 'Top_Percent' and compute the mean and standard deviation
    # of HitRate and Recall across all seeds.
    # ==========================================
    enrichment_files = glob.glob(os.path.join(base_output_dir, "seed*-plot_data_all_models_enrichment.csv"))
    if enrichment_files:
        df_enrich_list = [pd.read_csv(f) for f in enrichment_files]
        df_enrich_all = pd.concat(df_enrich_list, ignore_index=True)

        # Group by 'Top_Percent' and calculate the mean and standard deviation
        enrich_agg = df_enrich_all.groupby('Top_Percent').agg(['mean', 'std']).reset_index()

        # Flatten multi-level columns, e.g., mapping ('aaindex_Lasso_HitRate', 'mean') to 'aaindex_Lasso_HitRate_mean'
        enrich_agg.columns = ['_'.join(col).strip('_') for col in enrich_agg.columns.values]

        # ----- Save aggregated results -----
        out_enrich = os.path.join(summary_dir, "enrichment_curves.csv")
        enrich_agg.to_csv(out_enrich, index=False)
        print(f"✅ Enrichment curves aggregation complete -> {out_enrich}")
    else:
        print("⚠️ No enrichment result files found.")

    # ==========================================
    # Task 2: Aggregate Benchmark Reports
    # Logic: Compute the mean and standard deviation for metrics such as Spearman, MSE, R2, etc.
    # ==========================================
    benchmark_files = glob.glob(os.path.join(base_output_dir, "seed*-benchmark_results.csv"))
    if benchmark_files:
        df_bench_list = [pd.read_csv(f) for f in benchmark_files]
        df_bench_all = pd.concat(df_bench_list, ignore_index=True)

        # Core evaluation metrics
        numeric_cols = [
            'Spearman', 'MSE', 'R2',
            'Top20_Beneficial_Count', 'Top20_Best_TrueRank',
            'Top20_Median_TrueFit', 'Top20_Max_TrueFit'
        ]

        # Perform double-grouping by feature type and model name
        bench_agg = df_bench_all.groupby(['Feature', 'Model'])[numeric_cols].agg(['mean', 'std']).reset_index()

        # Flatten multi-level columns
        bench_agg.columns = ['_'.join(col).strip('_') for col in bench_agg.columns.values]

        # Sort descending by mean Spearman correlation to rank feature-model combinations
        bench_agg = bench_agg.sort_values(by="Spearman_mean", ascending=False).reset_index(drop=True)

        # ----- Save aggregated results -----
        out_bench = os.path.join(summary_dir, "core_metrics.csv")
        bench_agg.to_csv(out_bench, index=False)
        print(f"✅ Core metrics benchmark aggregation complete -> {out_bench}")
    else:
        print("⚠️ No benchmark_results files found.")

    # ==========================================
    # Task 3: Aggregate Raw Prediction Data
    # Logic: Automatically identify prediction columns, align by 'variant',
    # and calculate mean and standard deviation statistics.
    # ==========================================
    scatter_files = glob.glob(os.path.join(base_output_dir, "seed*-plot_data_all_predictions_raw.csv"))
    if scatter_files:
        df_scatter_list = [pd.read_csv(f) for f in scatter_files]
        df_scatter_all = pd.concat(df_scatter_list, ignore_index=True)

        # Automatically identify prediction columns to aggregate, excluding fixed metadata columns
        base_cols = ['variant', 'y_actual', 'binary']
        pred_cols = [col for col in df_scatter_all.columns if col not in base_cols]

        # Configure aggregation rules: Keep target values and binary labels from the first record
        agg_rules = {
            'y_actual': 'first',
            'binary': 'first'
        }
        # Dynamically apply statistics (mean, std, count) for each prediction column
        for col in pred_cols:
            agg_rules[col] = ['mean', 'std', 'count']

        scatter_agg = df_scatter_all.groupby('variant').agg(agg_rules).reset_index()

        # Flatten multi-level columns
        scatter_agg.columns = ['_'.join(col).strip('_') for col in scatter_agg.columns.values]

        # Refine column headers (remove redundant '_first' suffixes)
        scatter_agg.rename(columns={
            'y_actual_first': 'y_actual',
            'binary_first': 'binary'
        }, inplace=True)

        # ----- Save aggregated results -----
        out_scatter = os.path.join(summary_dir, "raw_predictions_distribution.csv")
        scatter_agg.to_csv(out_scatter, index=False)
        print(f"✅ Raw prediction data aggregation complete -> {out_scatter}")
    else:
        print("⚠️ No raw prediction/scatter result files found.")


if __name__ == "__main__":
    merge_results()