import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Import regression models
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import xgboost
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

########################################
# 1. Configuration & Parameters
########################################
TARGET_COL = "fitness_scaled"
BINARY_COL = "fitness_binary"  # Label column name for beneficial/harmful classification

# List of features to evaluate
feature_types = [
    "aaindex", "aaindex_with_esm",
    "onehot", "onehot_with_esm",
    "georgiev", "georgiev_with_esm",
    "prott5", "prott5_with_esm"
]

# Define regression models dictionary (10 models selected)
models_dict = {
    "Linear": linear_model.LinearRegression(),
    "Ridge": linear_model.RidgeCV(),
    "Lasso": linear_model.LassoCV(max_iter=100000, tol=1e-3),
    "ElasticNet": linear_model.ElasticNetCV(max_iter=100000, tol=1e-3),

    "RandomForest": RandomForestRegressor(
        n_estimators=100, criterion="friedman_mse", max_depth=None, min_samples_split=2,
        min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0,
        max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False,
        n_jobs=None, random_state=1, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None
    ),

    "XGBoost": xgboost.XGBRegressor(
        objective="reg:squarederror", colsample_bytree=0.5, learning_rate=0.1,
        max_depth=3, alpha=0.1, n_estimators=100, random_state=42
    ),

    "KNN": KNeighborsRegressor(n_neighbors=5),

    "PLSRegression": PLSRegression(n_components=5),

    "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}


########################################
# 2. Evaluation Core Function
########################################
def evaluate_all():
    # Target folder to save metrics, hit rates, and recall files
    output_dir = "simulation_results_hit_rate_recall"
    os.makedirs(output_dir, exist_ok=True)

    # Load labels and split datasets
    df_train = pd.read_csv("data_split/train_set_N96.csv")
    df_test = pd.read_csv("data_split/test_set_Remaining.csv")

    y_train = df_train[TARGET_COL].values
    y_test = df_test[TARGET_COL].values

    # Compute true global rank descending in the test set
    df_test['true_global_rank'] = df_test[TARGET_COL].rank(ascending=False, method='min').astype(int)

    results = []

    best_spearman = -1.0
    best_preds = None
    best_combination_name = ""

    all_preds_df = pd.DataFrame({
        "variant": df_test["mutant"],
        "y_actual": y_test,
        "binary": df_test[BINARY_COL]
    })

    # Iterate through each feature type
    for feat in feature_types:
        train_feat_path = f"feature_generation/features/train_{feat}.npy"
        test_feat_path = f"feature_generation/features/test_{feat}.npy"

        if not (os.path.exists(train_feat_path) and os.path.exists(test_feat_path)):
            print(f"⚠️ Feature files for {feat} not found, skipping...")
            continue

        print(f"\n[{feat.upper()}] Loading features...")
        X_train = np.load(train_feat_path)
        X_test = np.load(test_feat_path)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Iterate through each model
        for model_name, model in models_dict.items():
            print(f"  -> Training {model_name}...")

            try:
                # Train the model
                model.fit(X_train_scaled, y_train)

                # Perform predictions
                y_pred_test = model.predict(X_test_scaled)

                # Flatten prediction array for models yielding multidimensional outputs (e.g., PLSRegression)
                if y_pred_test.ndim > 1:
                    y_pred_test = y_pred_test.ravel()

                comb_col_name = f"{feat}_{model_name}"
                all_preds_df[comb_col_name] = y_pred_test.copy()

                # Compute standard regression metrics
                mse = mean_squared_error(y_test, y_pred_test)
                r2 = r2_score(y_test, y_pred_test)

                # Prevent calculation on constant predictions (defensive check)
                if len(np.unique(y_pred_test)) <= 1:
                    spearman_rho = np.nan
                else:
                    spearman_rho, _ = spearmanr(y_test, y_pred_test)

                # Check if this is the highest performing ranking model so far
                if pd.notna(spearman_rho) and spearman_rho > best_spearman:
                    best_spearman = spearman_rho
                    best_preds = y_pred_test.copy()
                    best_combination_name = comb_col_name

                # Construct local evaluation DataFrame
                eval_df = pd.DataFrame({
                    "y_pred": y_pred_test,
                    "y_actual": y_test,
                    "binary": df_test[BINARY_COL],
                    "true_global_rank": df_test["true_global_rank"]
                })

                # Sort by predicted fitness descending
                eval_df_sorted = eval_df.sort_values(by="y_pred", ascending=False).reset_index(drop=True)

                # Select top 20 mutations predicted by the model
                top_20 = eval_df_sorted.head(20)

                top20_beneficial_count = int((top_20["binary"] == 1).sum())
                best_idx_in_top20 = top_20["y_actual"].idxmax()
                top20_best_global_rank = int(top_20.loc[best_idx_in_top20, "true_global_rank"])
                median_activity_top20 = top_20["y_actual"].median()
                max_activity_top20 = top_20["y_actual"].max()

                # Record the evaluation results
                results.append({
                    "Feature": feat,
                    "Model": model_name,
                    "Spearman": spearman_rho,
                    "MSE": mse,
                    "R2": r2,
                    "Top20_Beneficial_Count": top20_beneficial_count,
                    "Top20_Best_TrueRank": top20_best_global_rank,
                    "Top20_Median_TrueFit": median_activity_top20,
                    "Top20_Max_TrueFit": max_activity_top20
                })

            except Exception as e:
                print(f"  ❌ {model_name} training failed: {e}")

    ########################################
    # 3. Summary & Benchmark Report
    ########################################
    if len(results) == 0:
        print("\n⚠️ No models successfully generated results. Please check the dataset or environment.")
        return

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Spearman", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 75)
    print("🏆 Benchmark Report (Sorted by Spearman Correlation) 🏆")
    print("=" * 75)
    print(results_df[
              ["Feature", "Model", "Spearman", "Top20_Beneficial_Count", "Top20_Best_TrueRank", "Top20_Max_TrueFit"]])

    benchmark_path = os.path.join(output_dir, "benchmark_results.csv")
    results_df.to_csv(benchmark_path, index=False)
    print(f"\n✅ All evaluation results saved to: {benchmark_path}")

    ########################################
    # 4. Compute Enrichment Curve Data (Hit Rate & Recall)
    ########################################
    print("\n" + "=" * 75)
    print("📊 Generating Hit Rate and Recall enrichment data for all combinations...")

    percentages = np.arange(1, 101, 1)  # Top 1% to Top 100%

    # Use a dictionary to buffer data and avoid Pandas fragmentation warnings
    enrichment_data_dict = {'Top_Percent': percentages}
    total_beneficial = all_preds_df['binary'].sum()

    for col in all_preds_df.columns:
        if col in ["variant", "y_actual", "binary"]:
            continue

        sorted_df = all_preds_df[["binary", col]].sort_values(by=col, ascending=False).reset_index(drop=True)

        hit_rates = []
        recalls = []

        for p in percentages:
            top_k_count = max(1, int(len(sorted_df) * (p / 100.0)))
            top_k_df = sorted_df.head(top_k_count)
            hits = top_k_df['binary'].sum()

            hit_rates.append(hits / top_k_count)
            recalls.append(hits / total_beneficial if total_beneficial > 0 else 0)

        # Store current data in the dictionary
        enrichment_data_dict[f'{col}_HitRate'] = hit_rates
        enrichment_data_dict[f'{col}_Recall'] = recalls

    # Convert dictionary to DataFrame after loop completion
    enrichment_results = pd.DataFrame(enrichment_data_dict)

    enrichment_path = os.path.join(output_dir, "plot_data_all_models_enrichment.csv")
    enrichment_results.to_csv(enrichment_path, index=False)
    print(f"💾 Enrichment curve data exported successfully: {enrichment_path}")

    scatter_path = os.path.join(output_dir, "plot_data_all_predictions_raw.csv")
    all_preds_df.to_csv(scatter_path, index=False)
    print(f"💾 Raw prediction distributions saved to: {scatter_path}")
    print("=" * 75)

    ########################################
    # 5. Export Test Set with Predictions from Top Performing Model
    ########################################
    if best_preds is not None:
        new_col_name = f"pred_fitness_{best_combination_name}"
        df_test[new_col_name] = best_preds

        output_test_file = os.path.join(output_dir, "test_set_with_best_predictions.csv")
        df_test.to_csv(output_test_file, index=False)

        print(f"🚀 Top performing combination: {best_combination_name} (Spearman: {best_spearman:.4f})")
        print(f"📂 Updated test set with top predictions exported to: {output_test_file}")
        print("=" * 75)


if __name__ == "__main__":
    evaluate_all()
