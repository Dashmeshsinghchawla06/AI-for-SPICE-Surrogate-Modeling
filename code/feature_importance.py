import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/dataset.csv")
df = df[df["sim_ok"] == True].dropna()

FEATURES = ["WN", "WP", "VDD", "CL", "VTO_SHIFT"]
TARGETS = ["tpdhl", "tpdlh", "trise", "tfall", "avg_power_w"]

print(f"Loaded {len(df)} valid samples")

for target in TARGETS:

    d = df[df[target] > 0].copy()

    X = d[FEATURES].values
    y = np.log(d[target].values)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=0
    )

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        random_state=0
    )

    model.fit(X_train, y_train)

    importance = model.feature_importances_

    print(f"\n=== {target} ===")

    ranking = sorted(
        zip(FEATURES, importance),
        key=lambda x: x[1],
        reverse=True
    )

    for feature, value in ranking:
        print(f"{feature:12s}: {value:.4f}")

    print(f"Sum: {importance.sum():.4f}")

