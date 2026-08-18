I-Based Surrogate Modeling for SPICE Simulation of
CMOS Circuits
Research Project README — Complete Copy Version
Research Area: AI for Circuit Simulation / SPICE Surrogate Modeling / VLSI
1. Research Question
Can a machine-learning model trained on SPICE-generated data accurately predict the performance of a
parameterized CMOS circuit while providing substantially faster evaluation than running a new SPICE
simulation for every design point?
The current study focuses on a three-stage CMOS inverter chain and predicts propagation delay (tpdhl, tpdlh),
transition time (trise, tfall), and average power (avg_power_w).
2. Circuit Under Study
The current experiment uses a three-stage CMOS inverter chain driven by a transient pulse input and connected to
a capacitive output load.
The circuit is parameterized using five primary input variables.
Parameter Range Description Unit
WN 0.5–4.0 NMOS width μm
WP 1.0–8.0 PMOS width μm
VDD 0.9–1.8 Supply voltage V
CL 10–200 Output load capacitance fF
VTO_SHIFT -0.05–0.05 Threshold-voltage shift V
The threshold-voltage shift is used to introduce variation in the operating conditions explored by the simulations.
The current implementation uses generic MOSFET models rather than a foundry-specific PDK.
3. Overall Methodology
The overall research workflow is:
4. Dataset Generation
The dataset was generated using ngspice transient simulations.
The circuit parameters were varied across the defined design space and the resulting transient responses were
processed to extract scalar performance metrics.
5. Dataset Generation Code
The dataset-generation implementation is located at:
6. Data Preparation
The dataset is loaded using Pandas. Rows are filtered using the sim_ok field and the required input and target
columns are checked for valid values.
The five machine-learning input features are WN, WP, VDD, CL, VTO_SHIFT.
The five prediction targets are tpdhl, tpdlh, trise, tfall, avg_power_w.
The current dataset contains 1500 valid simulation samples. For the propagation-delay targets, two samples
contain non-positive values. These samples cannot be used for logarithmic regression because the logarithm of a
non-positive value is undefined.
7. Machine-Learning Baseline
The first surrogate model selected for the project is a Gradient Boosting Regressor. A separate regression
model is trained for each target quantity.
The baseline model uses n_estimators=200, max_depth=3, learning_rate=0.1, and random_state=0.
An 80/20 train-test split is used and the input features are standardized using StandardScaler.
8. Log-Space Modeling
The timing quantities are highly nonlinear and span a range of values. The targets tpdhl, tpdlh, trise, and
tfall are modeled in logarithmic space.
The transformation is:
y_log = log(y)
Predicted values are transformed back into their original units for RMSE and MAE evaluation. Preliminary
comparison showed that fitting timing quantities directly in raw seconds can perform substantially worse, while
logarithmic modeling provides a much stronger regression relationship for the current dataset.
9. Baseline Results
The latest Gradient Boosting baseline was evaluated on a held-out test set. The current results are:
Target R² RMSE MAE Relative RMSE
tpdhl 0.9919 3.9532 × 10n¹¹ 2.0272 × 10n¹¹ 13.54%
tpdlh 0.9890 4.2317 × 10n¹¹ 1.8319 × 10n¹¹ 14.55%
trise 0.9941 7.8009 × 10n¹¹ 3.5825 × 10n¹¹ 13.23%
tfall 0.9922 5.8549 × 10n¹¹ 3.1226 × 10n¹¹ 10.87%
avg_power_w 0.9976 6.8859 × 10nn 4.9755 × 10nn 3.41%
The model achieves R² values between approximately 0.9890 and 0.9976. The highest R² is obtained for average
power (0.9976), while the lowest is obtained for tpdlh (0.9890).
10. Machine-Learning Inference Speed
The latest benchmark measured the machine-learning inference time for all five target models at 46.7
microseconds/sample.
Using approximately 24 ms as the SPICE reference, the ratio is approximately 24 ms / 46.7 μs ≈ 514. Therefore,
the current benchmark suggests an evaluation-time advantage of approximately 500× for the measured inference
comparison.
This should be interpreted carefully: the ML model requires an already-trained model and the original SPICE
simulations are required to generate the training dataset. The comparison represents evaluation of an
already-trained surrogate against running a new SPICE simulation.
11. Prediction Plots
Prediction figures are stored in results/figures/ and compare SPICE reference values against Gradient
Boosting predictions for tpdhl, tpdlh, trise, tfall, and avg_power_w.
12. Feature Importance
Feature-importance analysis is implemented using code/feature_importance.py and visualized using
code/plot_feature_importance.py. Feature importance should be interpreted as model-based importance
within the explored dataset rather than as direct proof of physical causality.
13. Result Files
Baseline metric table: results/baseline_results.csv
Test-set predictions: results/test_predictions.csv
14. Project Structure
AI-for-SPICE-Surrogate-Modeling/
nnn README.md
nnn code/
n nnn generate_dataset.py
n nnn train_baseline_model.py
n nnn train_baseline_model_original.py
n nnn make_plots.py
n nnn feature_importance.py
n nnn plot_feature_importance.py
nnn data/
n nnn dataset.csv
n nnn dataset_1500.csv
nnn documentation/
n nnn RESEARCH_PROGRESS_REPORT.md
nnn results/
n nnn baseline_results.csv
n nnn test_predictions.csv
n nnn figures/
nnn test_dataset.csv
15. Reproducibility
Install the required Python packages:
pip install numpy pandas scipy scikit-learn
Install ngspice separately according to the operating system.
Generate a dataset using python3 code/generate_dataset.py --n-samples 1500.
Train the baseline using python3 code/train_baseline_model.py.
16. Current Research Status
Completed: parameterized CMOS inverter-chain simulation; automated SPICE dataset generation; 1500 valid
simulation samples; data preparation; Gradient Boosting baseline; log-space regression; R², RMSE, MAE and
relative-error evaluation; ML inference-speed measurement; prediction plots; feature-importance analysis;
documentation; and Git-based version control.
17. Limitations
Generic transistor model: The simulations use generic MOSFET models rather than a foundry-specific PDK.
Single circuit topology: The current study uses one three-stage CMOS inverter chain.
Fixed channel length: Channel length is currently held fixed.
Limited dataset size: The current dataset contains 1500 simulations.
Scalar prediction: The current baseline predicts five scalar metrics and does not yet reproduce the complete
transient waveform.
18. Planned Research Work
Phase 1 — Rigorous Error Analysis: calculate RMSE, MAE, relative error, maximum-error samples,
worst-performing design points, error-distribution plots, and more rigorous SPICE/ML timing benchmarks.
Phase 2 — Model Comparison: compare Gradient Boosting with Random Forest, XGBoost or another boosting
approach, and other suitable regression models using R², RMSE, MAE, and inference time.
Phase 3 — Dataset Scaling: investigate 500, 1000, 1500, 3000, and 5000-sample datasets to determine how
accuracy scales with training-data size.
Phase 4 — Generalization Testing: test previously unseen parameter combinations, especially near
design-space boundaries and nonlinear regions.
Phase 5 — Other Circuits: evaluate a second circuit family to test whether the methodology generalizes beyond
one topology.
Phase 6 — Waveform-Level Surrogate: predict the complete transient v(out,t) waveform rather than only five
scalar metrics.
19. Research Significance
The motivation is the high computational cost of repeated circuit simulation during design-space exploration. A
trained surrogate may provide rapid approximate evaluations after sufficient SPICE-generated training data are
available.
The research therefore investigates the trade-off between SPICE simulation cost, training-data generation,
surrogate accuracy, and fast prediction.
20. Interpretation of Current Results
The current R² values demonstrate strong predictive performance within the explored design space. They do not
yet prove that the model can replace SPICE for arbitrary CMOS circuits.
Generalization beyond the current topology, extrapolation outside the training distribution, larger datasets, different
circuit configurations, waveform-level prediction, alternative ML models, and rigorous end-to-end
computational-cost analysis remain to be investigated.
The current work should therefore be described as a baseline surrogate-modeling study and research
prototype rather than a complete replacement for SPICE.
21. Conclusion
The current project demonstrates a complete initial workflow for AI-based surrogate modeling of SPICE circuit
simulations. A dataset of 1500 valid SPICE simulations was generated for a parameterized three-stage CMOS
inverter chain.
A Gradient Boosting regression model was trained to predict propagation delay, rise time, fall time, and average
power. The baseline achieves R² values from 0.9890 to 0.9976 across the five target quantities.
The measured ML inference time is approximately 46.7 μs/sample for all five target predictions. The results provide
evidence that a machine-learning surrogate can reproduce important SPICE-derived scalar metrics with high
accuracy within the investigated design space while providing substantially faster inference than a new SPICE
simulation.
The next stages will focus on rigorous error analysis, alternative model comparison, dataset scaling, generalization
testing, and waveform-level surrogate modeling.
22. Author
Dashmesh Singh Chawla
Electronics Engineering — VLSI Design
Thapar Institute of Engineering and Technology
Research Area: AI for Circuit Simulation / SPICE Surr
