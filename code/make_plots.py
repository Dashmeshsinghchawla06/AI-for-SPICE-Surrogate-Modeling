import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

# Load valid SPICE simulations
df = pd.read_csv("data/dataset.csv")
df = df[df["sim_ok"] == True].dropna()

FEATURES = ["WN", "WP", "VDD", "CL", "VTO_SHIFT"]
TARGETS = ["tpdhl", "tpdlh", "trise", "tfall", "avg_power_w"]

X = df[FEATURES].values

# Same 80/20 split used in your project
train_idx, test_idx = train_test_split(
    np.arange(len(df)),
    test_size=0.2,
    random_state=0
)

X_train = X[train_idx]
X_test = X[test_idx]

predictions = pd.DataFrame(index=test_idx)

for target in TARGETS:

    # Remove invalid/non-positive values exactly as the training approach does
    valid = df[target].values > 0

    valid_train = train_idx[valid[train_idx]]
    valid_test = test_idx[valid[test_idx]]

    X_target_train = X[valid_train]
    X_target_test = X[valid_test]

    y_train = np.log(df.iloc[valid_train][target].values)

    scaler = StandardScaler().fit(X_target_train)

    X_train_scaled = scaler.transform(X_target_train)
    X_test_scaled = scaler.transform(X_target_test)

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        random_state=0
    )

    model.fit(X_train_scaled, y_train)

    y_pred_log = model.predict(X_test_scaled)
    y_pred = np.exp(y_pred_log)

    actual = df.iloc[valid_test][target].values

    predictions.loc[valid_test, target + "_actual"] = actual
    predictions.loc[valid_test, target + "_predicted"] = y_pred

    r2 = r2_score(np.log(actual), y_pred_log)

    print(f"{target}: R² = {r2:.4f}")

    # Actual vs predicted plot
    plt.figure(figsize=(7, 6))
    plt.scatter(actual, y_pred)

    minimum = min(actual.min(), y_pred.min())
    maximum = max(actual.max(), y_pred.max())

    plt.plot([minimum, maximum], [minimum, maximum])

    plt.xlabel("Actual SPICE")
    plt.ylabel("AI Predicted")
    plt.title(f"{target}: SPICE vs Gradient Boosting")

    plt.tight_layout()
    plt.savefig(f"{target}_actual_vs_predicted.png", dpi=300)
    plt.close()

predictions.to_csv("test_predictions.csv")

print("\nPrediction data saved to test_predictions.csv")
print("Graphs saved for all five targets.")

