AI-BASED SURROGATE MODELING FOR SPICE SIMULATION OF CMOS CIRCUITS

Research Progress Report

Student: Dashmesh Singh Chawla
Program: Electronics Engineering — VLSI Design
Institution: Thapar Institute of Engineering and Technology
Research Area: AI for Circuit Simulation / SPICE Surrogate Modeling / VLSI
Date: 19 August 2026

================================================================================

1. PROJECT OBJECTIVE

The objective of this project is to investigate whether a machine-learning surrogate model can learn the relationship between CMOS circuit design parameters and SPICE-simulated performance metrics.

The motivation is to reduce the computational cost of repeated SPICE simulations during circuit design-space exploration. Instead of executing a complete SPICE simulation for every new parameter combination, a trained machine-learning model can provide rapid estimates of circuit performance.

The current work focuses on a three-stage CMOS inverter chain and investigates the prediction of propagation delay, transition time, and average power.

The objective is not to eliminate SPICE completely. Instead, SPICE is used to generate high-quality reference data, after which a trained machine-learning model can act as a surrogate for repeated evaluations within the explored circuit design space.

================================================================================

2. RESEARCH CONCEPT

The central idea is to use SPICE as the reference or ground-truth simulation engine and train a machine-learning model to approximate the relationship between circuit parameters and simulated performance.

The overall workflow is:

Circuit Parameters
        ↓
SPICE Simulation
        ↓
Performance Metrics
        ↓
Dataset
        ↓
Data Cleaning / Preparation
        ↓
Machine-Learning Training
        ↓
Prediction on Unseen Test Samples
        ↓
Accuracy and Speed Evaluation

The machine-learning model is intended to serve as a surrogate model after sufficient SPICE data have been generated.

The surrogate therefore does not replace SPICE as the final verification mechanism. Instead, it is intended to accelerate repeated evaluations and design-space exploration.

================================================================================

3. CIRCUIT AND SIMULATION SETUP

The current experiment uses a three-stage CMOS inverter chain.

The circuit is parameterized using five primary input variables:

Parameter       Description                    Unit
WN              NMOS width                    m
WP              PMOS width                    m
VDD             Supply voltage                V
CL              Load capacitance              F
VTO_SHIFT       Threshold-voltage shift       V

The five output quantities used as machine-learning targets are:

tpdhl       - high-to-low propagation delay
tpdlh       - low-to-high propagation delay
trise       - output rise time
tfall       - output fall time
avg_power_w - average power

The dataset also contains iavg and sim_ok fields. The sim_ok field is used to identify successful simulations.

================================================================================

4. DATASET GENERATION

A dataset containing 1500 circuit parameter combinations was generated.

The current dataset contains:

Total rows:                         1500
Successful simulations:             1500
Failed simulations:                    0
Valid dataset rows:                 1500

The dataset is stored as:

data/dataset.csv

An earlier generated copy is also retained as:

data/dataset_1500.csv

The dataset contains the following columns:

WN, WP, VDD, CL, VTO_SHIFT, sample_id,
tpdhl, tpdlh, trise, tfall, iavg, avg_power_w, sim_ok

All 1500 current dataset rows have sim_ok=True.

================================================================================

5. DATA PREPARATION

The dataset was loaded using Pandas.

The data-preparation process consists of:

1. Loading the SPICE-generated CSV dataset.
2. Selecting successful simulations using the sim_ok field.
3. Checking the required input and target columns for missing values.
4. Selecting the five circuit parameters as input features.
5. Separating the five performance metrics as prediction targets.
6. Removing non-positive target values only for targets that are modeled in logarithmic space.

The five input features are:

WN
WP
VDD
CL
VTO_SHIFT

The target quantities are:

tpdhl
tpdlh
trise
tfall
avg_power_w

The current dataset contains 1500 successful simulations.

Two propagation-delay targets contain two non-positive values each. These samples are excluded only when training the corresponding logarithmic regression model because the logarithm of a non-positive value is undefined.

Therefore, the number of samples used for each target is:

Target          Samples Used
tpdhl           1498
tpdlh           1498
trise           1500
tfall           1500
avg_power_w     1500

The dataset itself still contains 1500 successful simulations. The reduction to 1498 occurs only for the two affected logarithmic timing targets.

================================================================================

6. MACHINE-LEARNING METHODOLOGY

A Gradient Boosting Regressor was selected as the initial baseline surrogate model.

A separate regression model is trained for each target quantity.

The machine-learning workflow is:

SPICE Dataset
      ↓
Data Cleaning
      ↓
Feature Selection
      ↓
Train/Test Split
      ↓
Feature Standardization
      ↓
Log Transformation of Timing Targets
      ↓
Gradient Boosting Regression
      ↓
Prediction on Test Data
      ↓
R² / RMSE / MAE Evaluation

The five input features are:

1. WN
2. WP
3. VDD
4. CL
5. VTO_SHIFT

The baseline Gradient Boosting configuration is:

Model:                  GradientBoostingRegressor
Number of estimators:   200
Maximum depth:          3
Learning rate:          0.1
Random state:           0
Test fraction:          20%

An 80/20 train-test split is used with a fixed random seed of 0.

The input features are standardized using StandardScaler.

================================================================================

7. LOG-SPACE MODELING

The timing quantities are highly nonlinear and span a range of values.

The following targets are modeled in logarithmic space:

tpdhl
tpdlh
trise
tfall

The transformation is:

y_log = log(y)

The model is trained using the logarithm of the target and predictions are transformed back to the original units using:

y_pred = exp(y_pred_log)

Log-space modeling is used because delay and transition-time quantities can have strongly skewed distributions and can vary substantially across the circuit design space.

This transformation also prevents the regression model from being dominated by a small number of large-delay samples.

Average power is included as a target and its reported error metrics are converted back to the original units for interpretation.

================================================================================

8. BASELINE MODEL RESULTS

The current baseline model was trained using the 1500-row dataset.

The resulting held-out test-set performance is:

Target          R²          RMSE                  MAE                   Relative RMSE
tpdhl           0.991945    3.9532 × 10^-11       2.0272 × 10^-11       13.54%
tpdlh           0.988988    4.2317 × 10^-11       1.8319 × 10^-11       14.55%
trise           0.994086    7.8009 × 10^-11       3.5825 × 10^-11       13.23%
tfall           0.992201    5.8549 × 10^-11       3.1226 × 10^-11       10.87%
avg_power_w     0.997590    6.8859 × 10^-7        4.9755 × 10^-7        3.41%

The model achieved R² values between approximately 0.9890 and 0.9976.

The highest R² was obtained for average power:

R² = 0.997590

The lowest R² was obtained for tpdlh:

R² = 0.988988

These results indicate strong agreement between the surrogate predictions and SPICE-derived reference values within the explored design space.

The results are considered preliminary because further validation, error analysis, model comparison, and generalization testing are still required.

================================================================================

9. PREDICTION ACCURACY

The following metrics are used to evaluate the surrogate model.

R²

R² measures the proportion of variance in the target that is explained by the model.

A value close to 1 indicates strong agreement between predicted and reference values.

RMSE

Root Mean Squared Error measures the typical magnitude of prediction error while giving greater weight to larger errors.

MAE

Mean Absolute Error measures the average absolute difference between the prediction and the reference value.

Relative RMSE

Relative RMSE expresses RMSE relative to the mean magnitude of the corresponding test-set target.

The current results show high R² values for all five targets, while the relative errors indicate that additional analysis is required before claiming general-purpose replacement of SPICE.

================================================================================

10. SPICE VERSUS MACHINE-LEARNING PREDICTION PLOTS

Prediction plots have been generated by comparing SPICE reference values against Gradient Boosting predictions on held-out data.

The figures are stored in:

results/figures/

Current prediction plots include:

tpdhl_actual_vs_predicted.png
tpdlh_actual_vs_predicted.png
trise_actual_vs_predicted.png
tfall_actual_vs_predicted.png
avg_power_w_actual_vs_predicted.png

These plots provide a visual check of how closely the machine-learning predictions follow the SPICE reference values.

A prediction close to the diagonal reference relationship indicates good agreement between the surrogate and SPICE.

================================================================================

11. FEATURE IMPORTANCE ANALYSIS

Feature importance was calculated from the Gradient Boosting models to investigate which circuit parameters contribute most strongly to prediction performance.

11.1 tpdhl

Feature          Importance
CL               0.5395
WN               0.3002
VDD              0.1517
WP               0.0047
VTO_SHIFT        0.0038

11.2 tpdlh

Feature          Importance
CL               0.5475
WP               0.2954
VDD              0.1385
WN               0.0149
VTO_SHIFT        0.0036

11.3 trise

Feature          Importance
CL               0.5758
WP               0.3034
VDD              0.1183
VTO_SHIFT        0.0023
WN               0.0001

11.4 tfall

Feature          Importance
CL               0.5669
WN               0.3064
VDD              0.1244
VTO_SHIFT        0.0023
WP               0.0001

11.5 Average Power

Feature          Importance
CL               0.7484
VDD              0.2499
WP               0.0015
VTO_SHIFT        0.0001
WN               0.0001

Load capacitance (CL) is the most important feature across all five targets in the current model.

The results also show that transistor width and supply voltage have substantial importance for timing and power prediction.

These feature-importance values should be interpreted as model-based importance measures rather than direct proof of physical causality.

The generated feature-importance figures are stored in:

results/figures/

Files include:

tpdhl_feature_importance.png
tpdlh_feature_importance.png
trise_feature_importance.png
tfall_feature_importance.png
avg_power_w_feature_importance.png

================================================================================

12. MACHINE-LEARNING INFERENCE SPEED

A preliminary inference benchmark was performed for the trained Gradient Boosting models.

The measured machine-learning inference time was:

47.5 microseconds per sample for all five targets.

The current project notes use an approximate SPICE simulation time of:

25–30 milliseconds per circuit evaluation

for the three-stage inverter chain on the development machine.

Using these values as a preliminary comparison:

25 ms / 47.5 microseconds ≈ 526×
30 ms / 47.5 microseconds ≈ 632×

This corresponds to an approximate potential speed advantage of:

~526×–632× per evaluated sample

under the present benchmark assumptions.

This comparison is preliminary. A more rigorous benchmark should use the same input conditions, repeated measurements, warm-up runs, and identical evaluation procedures for both SPICE and ML.

================================================================================

13. CURRENT PROJECT STATUS

The following components have currently been completed:

- SPICE dataset generation
- 1500 successful simulation records
- Dataset cleaning and validation
- Baseline Gradient Boosting model
- 80/20 train-test evaluation
- Log-space modeling for timing targets
- R² calculation
- RMSE calculation
- MAE calculation
- Relative RMSE calculation
- Prediction plots
- Feature-importance analysis
- Preliminary ML inference-speed measurement
- Initial research documentation

The current implementation is therefore a functioning baseline surrogate-model pipeline rather than only a conceptual proposal.

================================================================================

14. CURRENT LIMITATIONS

The current work has several limitations that need to be addressed before drawing stronger conclusions.

14.1 Dataset Size

The current dataset contains 1500 successful simulations. Larger datasets should be investigated to determine whether additional SPICE samples improve generalization.

14.2 Single Baseline Model

Only Gradient Boosting has currently been evaluated as the main surrogate model.

Other regression models should be compared.

14.3 Limited Design Space

The conclusions currently apply only to the circuit topology and parameter ranges represented in the generated dataset.

14.4 Generalization

The current held-out test set evaluates performance on unseen samples from the current dataset distribution. Additional tests near parameter-space boundaries and on deliberately unseen combinations are required.

14.5 Timing Benchmark

The current ML-versus-SPICE speed comparison is preliminary. A controlled repeated benchmark should be performed before using a final speedup value in a publication.

14.6 Feature Importance

Feature importance indicates how useful features are to the trained model, but it should not automatically be interpreted as physical causality.

================================================================================

15. PLANNED WORK

The next phase of the project will focus on rigorous evaluation and expansion of the baseline system.

Phase 1 — Complete Baseline Evaluation

- Perform a controlled SPICE versus ML timing benchmark.
- Calculate RMSE and MAE consistently for all targets.
- Calculate relative prediction error.
- Analyze maximum prediction error.
- Generate error-distribution plots.
- Identify the worst-performing samples.
- Inspect whether errors concentrate near particular circuit operating regions.

Phase 2 — Machine-Learning Model Comparison

Compare the Gradient Boosting baseline with additional regression approaches such as:

- Random Forest
- XGBoost or another boosting approach
- Other suitable regression models

Compare the models using:

- R²
- RMSE
- MAE
- inference time

Phase 3 — Dataset Scaling

Investigate model performance as the number of SPICE-generated training samples increases.

Potential dataset sizes:

- 500
- 1000
- 1500
- 3000
- 5000

The objective is to determine the relationship between dataset size and surrogate-model accuracy.

Phase 4 — Generalization Testing

Evaluate the model on parameter combinations near the boundaries of the design space and on deliberately unseen combinations.

The purpose is to determine whether the surrogate can generalize beyond the exact distribution used during training.

Phase 5 — Final Surrogate Evaluation

Select the best-performing model based on both prediction accuracy and computational cost.

The final objective will be to establish the trade-off between:

SPICE reference accuracy → ML surrogate accuracy → computational cost

================================================================================

16. EXPECTED RESEARCH CONTRIBUTION

The current work is intended to investigate a practical workflow for replacing repeated SPICE evaluations with a machine-learning surrogate after sufficient SPICE data have been generated.

The potential contribution is not simply a high R² value. The more important research questions are:

1. How accurately can the surrogate reproduce SPICE-derived circuit metrics?
2. Which circuit parameters dominate surrogate prediction?
3. How does accuracy change as the SPICE training dataset grows?
4. How well does the model generalize to unseen parameter combinations?
5. How much computational time can be saved?
6. Which machine-learning model provides the best accuracy-versus-cost trade-off?

These questions will form the basis of the later experimental and comparative stages of the project.

================================================================================

17. CONCLUSION

This preliminary study demonstrates a working machine-learning surrogate-model pipeline for a parameterized three-stage CMOS inverter chain.

A dataset of 1500 successful SPICE simulations was generated using five circuit parameters: WN, WP, VDD, CL, and VTO_SHIFT.

A Gradient Boosting Regressor was trained separately for five SPICE-derived performance metrics:

- tpdhl
- tpdlh
- trise
- tfall
- avg_power_w

The baseline model achieved R² values from approximately 0.9890 to 0.9976 on the held-out evaluation data.

The strongest result was obtained for average power with:

R² = 0.997590

Feature-importance analysis showed that load capacitance was the dominant input feature across all five prediction targets.

A preliminary inference benchmark measured approximately 47.5 microseconds per sample for all five ML predictions. Compared with the current approximate 25–30 ms SPICE evaluation time, this suggests a potential speed advantage of approximately 526×–632× under the present benchmark assumptions.

These results are promising, but they should be treated as preliminary. Further work is required in error analysis, controlled timing benchmarks, model comparison, dataset scaling, and generalization testing.

The next stage will therefore focus on strengthening the experimental methodology and determining whether the observed accuracy and computational advantage remain consistent across larger datasets, alternative models, and previously unseen circuit parameter combinations.

================================================================================

18. REPRODUCIBILITY

The project is organized as follows:

Ai_for_spice/

├── documentation/
│   └── RESEARCH_PROGRESS_REPORT.md
│
├── code/
│   ├── generate_dataset.py
│   ├── train_baseline_model.py
│   ├── train_baseline_model_original.py
│   ├── make_plots.py
│   ├── feature_importance.py
│   └── plot_feature_importance.py
│
├── data/
│   ├── dataset.csv
│   └── dataset_1500.csv
│
├── results/
│   ├── baseline_results.csv
│   ├── test_predictions.csv
│   └── figures/
│       ├── tpdhl_actual_vs_predicted.png
│       ├── tpdlh_actual_vs_predicted.png
│       ├── trise_actual_vs_predicted.png
│       ├── tfall_actual_vs_predicted.png
│       ├── avg_power_w_actual_vs_predicted.png
│       ├── tpdhl_feature_importance.png
│       ├── tpdlh_feature_importance.png
│       ├── trise_feature_importance.png
│       ├── tfall_feature_importance.png
│       └── avg_power_w_feature_importance.png
│
└── README.md

The main scripts are:

Dataset generation:
code/generate_dataset.py

Baseline model training:
code/train_baseline_model.py

Prediction plotting:
code/make_plots.py

Feature-importance calculation:
code/feature_importance.py

Feature-importance plotting:
code/plot_feature_importance.py

The main numerical results are stored in:

results/baseline_results.csv

The held-out predictions are stored in:

results/test_predictions.csv

The complete SPICE dataset is stored in:

data/dataset.csv

================================================================================

APPENDIX A — MAIN INPUT AND OUTPUT VARIABLES

Input Features

WN          NMOS width
WP          PMOS width
VDD         Supply voltage
CL          Load capacitance
VTO_SHIFT   Threshold-voltage shift

Output Targets

tpdhl       High-to-low propagation delay
tpdlh       Low-to-high propagation delay
trise       Output rise time
tfall       Output fall time
avg_power_w Average power

================================================================================

APPENDIX B — BASELINE RESULTS SUMMARY

Target          R²          RMSE                  MAE
tpdhl           0.991945    3.9532 × 10^-11       2.0272 × 10^-11
tpdlh           0.988988    4.2317 × 10^-11       1.8319 × 10^-11
trise           0.994086    7.8009 × 10^-11       3.5825 × 10^-11
tfall           0.992201    5.8549 × 10^-11       3.1226 × 10^-11
avg_power_w     0.997590    6.8859 × 10^-7        4.9755 × 10^-7

================================================================================

APPENDIX C — FEATURE IMPORTANCE SUMMARY

Target          Most Important Feature       Importance
tpdhl           CL                           0.5395
tpdlh           CL                           0.5475
trise           CL                           0.5758
tfall           CL                           0.5669
avg_power_w     CL                           0.7484

The consistent dominance of CL is an important observation for further investigation, but it should be validated using additional datasets and model-analysis techniques before making a physical interpretation.

================================================================================

APPENDIX D — RESEARCH STATUS

Current stage: Baseline surrogate model completed.

Completed:

- SPICE dataset generation
- Dataset validation
- Gradient Boosting baseline
- Log-space timing regression
- Held-out evaluation
- R² / RMSE / MAE analysis
- Prediction plots
- Feature-importance analysis
- Preliminary speed measurement
- Initial documentation

Next Immediate Tasks:

1. Controlled SPICE-versus-ML timing benchmark
2. Maximum-error and error-distribution analysis
3. Comparison with additional ML models
4. Dataset-size scaling experiments
5. Generalization testing
6. Final model selection
7. Preparation of a research-paper-style manuscript

================================================================================

END OF RESEARCH PROGRESS REPORT
