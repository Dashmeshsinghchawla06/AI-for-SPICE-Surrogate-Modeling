"""
train_baseline_model.py

Clean baseline surrogate model for AI-for-SPICE.

Inputs:
    WN, WP, VDD, CL, VTO_SHIFT

Targets:
    tpdhl, tpdlh, trise, tfall, avg_power_w

The model uses Gradient Boosting Regression.
Targets are modeled in log space.

The scaler is fitted ONLY on the training set to avoid
data leakage from the held-out test set.
"""

import argparse
import time
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


FEATURES = ["WN", "WP", "VDD", "CL", "VTO_SHIFT"]

TARGETS = [
    "tpdhl",
    "tpdlh",
    "trise",
    "tfall",
    "avg_power_w"
]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/dataset.csv"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    df = pd.read_csv(args.dataset)

    df = df[
        df["sim_ok"] == True
    ].dropna(
        subset=FEATURES + TARGETS
    )

    print(
        f"Loaded {len(df)} valid samples "
        f"from {args.dataset}"
    )

    results = []

    models = {}
    scalers = {}

    # --------------------------------------------------
    # Train one model per target
    # --------------------------------------------------

    for target in TARGETS:

        print(f"\nTraining model for {target}...")

        # Remove non-positive values before log transform
        d = df[df[target] > 0].copy()

        X = d[FEATURES].values
        y = np.log(d[target].values)

        # --------------------------------------------------
        # IMPORTANT:
        # Split BEFORE fitting the scaler.
        # --------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.seed
        )

        # Fit scaler ONLY on training data
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)

        X_test_scaled = scaler.transform(X_test)

        scalers[target] = scaler

        # --------------------------------------------------
        # Gradient Boosting model
        # --------------------------------------------------

        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            random_state=args.seed
        )

        model.fit(
            X_train_scaled,
            y_train
        )

        models[target] = model

        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------

        y_pred_log = model.predict(
            X_test_scaled
        )

        # R² in log space
        r2 = r2_score(
            y_test,
            y_pred_log
        )

        # Convert back to physical units
        y_test_real = np.exp(y_test)
        y_pred_real = np.exp(y_pred_log)

        rmse_real = np.sqrt(
            mean_squared_error(
                y_test_real,
                y_pred_real
            )
        )

        mae_real = np.mean(
            np.abs(
                y_test_real -
                y_pred_real
            )
        )

        relative_rmse = (
            rmse_real /
            (np.mean(y_test_real) + 1e-30)
        )

        results.append({
            "target": target,
            "r2_log_space": r2,
            "rmse_original_units": rmse_real,
            "mae_original_units": mae_real,
            "relative_rmse_pct": relative_rmse * 100,
            "n_samples": len(d)
        })

        print(
            f"R² = {r2:.4f}"
        )

        print(
            f"RMSE = {rmse_real:.4e}"
        )

        print(
            f"MAE = {mae_real:.4e}"
        )

    # --------------------------------------------------
    # Results table
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    print(
        "\n=== Final Baseline Model Accuracy ==="
    )

    print(
        results_df.to_string(index=False)
    )

    # --------------------------------------------------
    # Speed benchmark
    # --------------------------------------------------

    n_bench = min(
        50,
        len(df)
    )

    t0 = time.perf_counter()

    for target in TARGETS:

        X_bench = scalers[target].transform(
            df[FEATURES].values[:n_bench]
        )

        models[target].predict(
            X_bench
        )

    t_model = (
        time.perf_counter() -
        t0
    )

    per_sample_model_us = (
        t_model /
        n_bench
    ) * 1e6

    print(
        "\n=== ML Inference Speed ==="
    )

    print(
        f"{per_sample_model_us:.1f} "
        "microseconds/sample "
        "(all five targets)"
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    results_df.to_csv(
        "results/baseline_results.csv",
        index=False
    )

    print(
        "\nSaved:"
        " results/baseline_results.csv"
    )


if __name__ == "__main__":
    main()
