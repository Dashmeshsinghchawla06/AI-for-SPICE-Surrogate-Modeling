# AI-Based Surrogate Modeling for SPICE Simulation of CMOS Circuits

<p align="center">
  <strong>AI for Circuit Simulation • SPICE Surrogate Modeling • VLSI</strong>
</p>

<p align="center">
  <em>Learning from SPICE simulations to make repeated CMOS circuit evaluation faster.</em>
</p>

---

## Project Overview

SPICE is widely used to evaluate circuit behaviour, but repeatedly simulating many combinations of circuit parameters can become expensive. This project explores whether a machine-learning model can learn from SPICE-generated data and provide a fast approximation for repeated design-space evaluations.

The current study uses a **three-stage CMOS inverter chain** and real `ngspice` transient simulations. A Gradient Boosting surrogate is trained to predict five SPICE-derived performance metrics.

### Results at a Glance

| Metric | Result |
|---|---:|
| SPICE simulations | **1,500** |
| Input parameters | **5** |
| Predicted metrics | **5** |
| Best R² | **0.9976** |
| Lowest R² | **0.9890** |
| ML inference time | **46.7 μs/sample** |
| SPICE reference time | **~24 ms/sample** |
| Evaluation-time ratio | **~514×** |

> **Current status:** The initial SPICE-to-ML baseline is complete. The next stage focuses on understanding model error, scaling the dataset, testing other models and circuit topologies, and eventually predicting complete transient waveforms.

---

# 1. Research Question

> **Can a machine-learning model trained on SPICE-generated data accurately predict the performance of a parameterized CMOS circuit while being substantially faster to evaluate than a new SPICE simulation?**

The current model predicts:

- `tpdhl` — high-to-low propagation delay
- `tpdlh` — low-to-high propagation delay
- `trise` — rise time
- `tfall` — fall time
- `avg_power_w` — average power

The longer-term goal is to investigate whether this approach can be extended from scalar metrics to complete transient waveform prediction.

---

# 2. Circuit and Parameter Space

The circuit is a **three-stage CMOS inverter chain** driven by a transient pulse and connected to a capacitive output load.

The five parameters varied during dataset generation are:

| Parameter | Range | Description |
|---|---:|---|
| `WN` | 0.5–4.0 μm | NMOS width |
| `WP` | 1.0–8.0 μm | PMOS width |
| `VDD` | 0.9–1.8 V | Supply voltage |
| `CL` | 10–200 fF | Output load capacitance |
| `VTO_SHIFT` | −0.05–0.05 V | Threshold-voltage shift |

The channel length is held fixed. The current implementation uses generic MOSFET models rather than a foundry-specific PDK, so the results demonstrate the surrogate-modeling methodology rather than process-specific device behaviour.

### Circuit Schematic

![Parameterized three-stage CMOS inverter chain](results/figures/circuit_schematic.png)

### Representative SPICE Response

A representative transient simulation is shown below. The transient analysis covers 20 ns with a maximum timestep of 10 ps.

![SPICE transient response](results/figures/example_spice_waveform.png)

---

# 3. Methodology

The project follows a simple simulation-to-surrogate workflow:

```text
Circuit Parameters
       ↓
ngspice Transient Simulation
       ↓
Performance Metric Extraction
       ↓
SPICE Dataset
       ↓
Data Preparation
       ↓
Gradient Boosting Surrogate
       ↓
Held-Out Test Evaluation
```

The five circuit parameters form the model inputs, while the five SPICE-derived metrics form the prediction targets.

The dataset contains **1,500 successful simulations**. For `tpdhl` and `tpdlh`, two samples contain non-positive values and are excluded from the corresponding log-space regression because `log(y)` is undefined for non-positive values.

The baseline uses an 80/20 train-test split and a Gradient Boosting Regressor with:

- 200 estimators
- maximum depth of 3
- learning rate of 0.1
- random state of 0

The input features are standardized using `StandardScaler`.

---

# 4. Key Modeling Observation — Log-Space Regression

One of the useful findings from the baseline experiment was the effect of representing the timing targets in logarithmic space.

Instead of directly fitting:

```text
y = timing
```

the timing targets are modeled as:

```text
y_log = log(y)
```

and transformed back to their original units for RMSE and MAE evaluation.

The preliminary raw-space comparison performed substantially worse for the timing targets, while log-space modeling produced a much stronger regression relationship on the current dataset.

This is a useful observation for the project because the timing quantities vary over a relatively wide numerical range and their relationship with circuit parameters is nonlinear.

> **Important:** This finding is specific to the current circuit and dataset. It is being treated as an experimentally observed modeling choice, not as a universal rule for SPICE surrogate modeling.

---

# 5. Baseline Results

The Gradient Boosting baseline was evaluated on a held-out test set.

| Target | R² | RMSE | MAE | Relative RMSE |
|---|---:|---:|---:|---:|
| `tpdhl` | **0.991945** | 3.953193e-11 | 2.027215e-11 | 13.54% |
| `tpdlh` | **0.988988** | 4.231690e-11 | 1.831914e-11 | 14.55% |
| `trise` | **0.994086** | 7.800899e-11 | 3.582475e-11 | 13.23% |
| `tfall` | **0.992201** | 5.854914e-11 | 3.122635e-11 | 10.87% |
| `avg_power_w` | **0.997590** | 6.885861e-07 | 4.975517e-07 | 3.41% |

The model achieved R² values between **0.988988 and 0.997590** across all five targets. Average power was the most accurately predicted quantity, while `tpdlh` had the lowest R².

These results indicate that the model captures the relationship between the sampled circuit parameters and the SPICE-derived scalar metrics well within the investigated design space.

## SPICE vs. Machine-Learning Predictions

### `tpdhl` — Propagation Delay

![tpdhl actual vs predicted](results/figures/tpdhl_actual_vs_predicted.png)

### `tpdlh` — Propagation Delay

![tpdlh actual vs predicted](results/figures/tpdlh_actual_vs_predicted.png)

### `trise` — Rise Time

![trise actual vs predicted](results/figures/trise_actual_vs_predicted.png)

### `tfall` — Fall Time

![tfall actual vs predicted](results/figures/tfall_actual_vs_predicted.png)

### `avg_power_w` — Average Power

![Average power actual vs predicted](results/figures/avg_power_w_actual_vs_predicted.png)

---

# 6. Feature Importance

Feature-importance analysis was used to understand which input parameters contributed most strongly to each Gradient Boosting model.

The five features are:

`WN`, `WP`, `VDD`, `CL`, and `VTO_SHIFT`.

These values represent model-based importance within the explored dataset and should not be interpreted as direct proof of physical causality.

### `tpdhl`

![tpdhl feature importance](results/figures/tpdhl_feature_importance.png)

### `tpdlh`

![tpdlh feature importance](results/figures/tpdlh_feature_importance.png)

### `trise`

![trise feature importance](results/figures/trise_feature_importance.png)

### `tfall`

![tfall feature importance](results/figures/tfall_feature_importance.png)

### `avg_power_w`

![Average power feature importance](results/figures/avg_power_w_feature_importance.png)

---

# 7. Inference Speed

The measured inference time for predicting all five quantities was approximately:

**46.7 μs per sample**

The corresponding SPICE reference time was approximately:

**24 ms per simulation**

Therefore:

```text
24 ms / 46.7 μs ≈ 514
```

This gives an evaluation-time ratio of approximately **514×** for the current benchmark.

This should not be interpreted as a 514× end-to-end project speedup. The comparison is between an **already-trained surrogate** and one new SPICE evaluation. The cost of generating the original training dataset and training the model is not included.

---

# 8. Limitations and Next Research Steps

The current work is a baseline study rather than a complete replacement for SPICE.

### Current limitations

- Only one circuit topology has been studied.
- Generic MOSFET models are used instead of a foundry PDK.
- Channel length is held fixed.
- The current dataset contains 1,500 simulations.
- The model predicts five scalar quantities rather than the complete transient waveform.
- Generalization outside the sampled design space has not yet been established.

### Next steps

**1. Rigorous error analysis**  
Investigate maximum-error samples, error distributions, relative error, and controlled SPICE-versus-ML timing measurements.

**2. Model comparison**  
Compare Gradient Boosting with Random Forest, XGBoost, and other suitable regression approaches.

**3. Dataset scaling**  
Evaluate performance as the dataset grows from approximately 500 to 5,000 simulations.

**4. Generalization**  
Test previously unseen parameter combinations, especially near design-space boundaries and nonlinear regions.

**5. Additional circuit topology**  
Evaluate whether the approach transfers beyond the three-stage inverter chain.

**6. Waveform-level surrogate**  
Move from predicting five scalar metrics to predicting the complete transient `v(out,t)` waveform.

The waveform-level model is the longer-term goal because it would provide a closer surrogate for the actual transient behaviour produced by SPICE.

---

# 9. Reproducibility

Install the required Python packages:

```bash
pip install numpy pandas scipy scikit-learn
```

Install `ngspice` separately.

### Generate the dataset

```bash
python3 code/generate_dataset.py --n-samples 1500
```

### Train the baseline

```bash
python3 code/train_baseline_model.py
```

### Main outputs

```text
data/dataset.csv
results/baseline_results.csv
results/test_predictions.csv
results/figures/
```

---

# 10. Conclusion

This project establishes a working baseline for AI-based surrogate modeling of SPICE simulations.

Using **1,500 successful `ngspice` simulations** of a parameterized three-stage CMOS inverter chain, a Gradient Boosting model was trained to predict propagation delay, rise time, fall time, and average power.

The baseline achieved R² values from **0.988988 to 0.997590**. The trained surrogate required approximately **46.7 μs per sample** to predict all five quantities, compared with an approximately **24 ms** SPICE reference evaluation.

The results suggest that a machine-learning surrogate can reproduce important SPICE-derived scalar metrics with high accuracy within the current design space while providing substantially faster evaluation.

The main research question for the next stage is whether this performance remains reliable with more training data, different circuit topologies, and ultimately complete transient waveform prediction.

---

## Project Structure

```text
AI-for-SPICE-Surrogate-Modeling/

├── README.md
├── code/
│   ├── generate_dataset.py
│   ├── train_baseline_model.py
│   ├── make_plots.py
│   ├── feature_importance.py
│   └── plot_feature_importance.py
├── data/
│   └── dataset.csv
├── documentation/
│   └── RESEARCH_PROGRESS_REPORT.md
└── results/
    ├── baseline_results.csv
    ├── test_predictions.csv
    └── figures/
```

---

**Dashmesh Singh Chawla**  
Electronics Engineering — VLSI Design  
Thapar Institute of Engineering and Technology

*AI for Circuit Simulation / SPICE Surrogate Modeling / VLSI*