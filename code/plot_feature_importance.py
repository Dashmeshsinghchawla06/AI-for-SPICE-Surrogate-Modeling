import matplotlib.pyplot as plt
import pandas as pd

importance = {
    "tpdhl": {
        "CL": 0.5395,
        "WN": 0.3002,
        "VDD": 0.1517,
        "WP": 0.0047,
        "VTO_SHIFT": 0.0038
    },
    "tpdlh": {
        "CL": 0.5475,
        "WP": 0.2954,
        "VDD": 0.1385,
        "WN": 0.0149,
        "VTO_SHIFT": 0.0036
    },
    "trise": {
        "CL": 0.5758,
        "WP": 0.3034,
        "VDD": 0.1183,
        "VTO_SHIFT": 0.0023,
        "WN": 0.0001
    },
    "tfall": {
        "CL": 0.5669,
        "WN": 0.3064,
        "VDD": 0.1244,
        "VTO_SHIFT": 0.0023,
        "WP": 0.0001
    },
    "avg_power_w": {
        "CL": 0.7484,
        "VDD": 0.2499,
        "WP": 0.0015,
        "VTO_SHIFT": 0.0001,
        "WN": 0.0001
    }
}

for target, values in importance.items():

    df = pd.DataFrame(
        list(values.items()),
        columns=["Feature", "Importance"]
    ).sort_values("Importance")

    plt.figure(figsize=(8, 5))

    plt.barh(df["Feature"], df["Importance"])

    plt.xlabel("Feature Importance")
    plt.ylabel("Circuit Parameter")
    plt.title(f"{target}: Gradient Boosting Feature Importance")

    plt.tight_layout()

    filename = f"{target}_feature_importance.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved {filename}")
