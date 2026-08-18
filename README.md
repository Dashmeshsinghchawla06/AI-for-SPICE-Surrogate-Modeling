AI-BASED SURROGATE MODELING FOR SPICE SIMULATION OF CMOS CIRCUITS

Research Project

Author: Dashmesh Singh Chawla
Program: Electronics Engineering — VLSI Design
Institution: Thapar Institute of Engineering and Technology
Research Area: AI for Circuit Simulation / SPICE Surrogate Modeling / VLSI


ABSTRACT

Repeated SPICE simulation can become computationally expensive during circuit design-space exploration, where many combinations of device and operating parameters must be evaluated. This project investigates whether a machine-learning surrogate can learn the relationship between parameterized CMOS circuit variables and SPICE-derived performance metrics, while substantially reducing the evaluation time for subsequent design points.

The current study considers a three-stage CMOS inverter chain driven by a transient pulse and connected to a capacitive output load. Five circuit parameters are varied: NMOS width (WN), PMOS width (WP), supply voltage (VDD), output load capacitance (CL), and threshold-voltage shift (VTO_SHIFT). A dataset of 1500 successful ngspice simulations is used to train and evaluate Gradient Boosting regression models for five scalar targets: tpdhl, tpdlh, trise, tfall, and average power.

The current baseline achieves R² values between 0.988988 and 0.997590. The measured inference time for predicting all five targets is approximately 46.7 microseconds per sample. Using an approximately 24 ms SPICE reference time, the measured evaluation-time ratio is approximately 514×. This comparison applies only to an already-trained surrogate and does not include the cost of generating the training data or training the model.

The results provide an initial demonstration of accurate scalar SPICE surrogate modeling within the explored design space. Further work will investigate rigorous error analysis, alternative machine-learning models, dataset scaling, generalization to additional circuit topologies, and waveform-level prediction.


1. INTRODUCTION

Circuit simulation is an essential component of modern electronic design workflows. SPICE-based simulation provides detailed estimates of circuit behavior, but repeated simulation can become costly when large design spaces must be explored.

Machine-learning surrogate modeling offers an alternative approach: an expensive simulator is first used to generate a representative dataset, after which a trained model can approximate simulator outputs for new parameter combinations. The potential benefit is particularly relevant to design-space exploration, optimization, and repeated evaluation.

This project investigates a baseline version of this approach for a parameterized CMOS circuit. SPICE remains the reference simulation engine; the machine-learning model is treated as a surrogate for rapid repeated evaluation within the investigated design space.

The current work is intentionally presented as a research prototype rather than as a complete replacement for SPICE.


2. RESEARCH QUESTION

Can a machine-learning model trained on SPICE-generated data accurately predict the performance of a parameterized CMOS circuit while providing substantially faster evaluation than running a new SPICE simulation for every design point?

The present study predicts five scalar performance metrics:

- tpdhl — high-to-low propagation delay
- tpdlh — low-to-high propagation delay
- trise — output rise time
- tfall — output fall time
- avg_power_w — average power


3. CIRCUIT AND DESIGN SPACE

The circuit under study is a three-stage CMOS inverter chain driven by a transient pulse input and connected to a capacitive load at the output.

The five primary input variables are:

Parameter       Range             Description
WN              0.5–4.0 µm        NMOS width
WP              1.0–8.0 µm        PMOS width
VDD             0.9–1.8 V         Supply voltage
CL              10–200 fF         Output load capacitance
VTO_SHIFT       −0.05–0.05 V      Threshold-voltage shift

The threshold-voltage shift is used to introduce variation in the operating conditions explored by the simulations. Channel length is held fixed while WN, WP, VDD, CL, and VTO_SHIFT are varied during dataset generation.

The current implementation uses generic MOSFET models rather than a foundry-specific PDK. Therefore, the current results demonstrate the surrogate-modeling methodology rather than production-level predictions for a specific semiconductor technology.

The repository includes a schematic of the parameterized circuit and a representative SPICE transient response.


4. METHODOLOGY

The overall workflow is:

Circuit design parameters
        ↓
SPICE transient simulation
        ↓
Performance-metric extraction
        ↓
Dataset generation
        ↓
Data preparation
        ↓
Machine-learning training
        ↓
Prediction on held-out data
        ↓
Accuracy and inference-speed evaluation

SPICE is treated as the reference simulation. The machine-learning model is trained only after SPICE-generated data are available.


5. DATASET GENERATION

The dataset was generated using ngspice transient simulations. Circuit parameters were varied across the defined design space and the resulting transient responses were processed to extract scalar performance metrics.

The current dataset contains:

- Total simulations: 1500
- Successful simulations: 1500
- Failed simulations: 0
- Input features: 5
- Prediction targets: 5

The primary dataset is stored in:

data/dataset.csv

Each row contains circuit parameters, simulation outputs, and simulation-status information.

Primary columns:

WN
WP
VDD
CL
VTO_SHIFT
sample_id
tpdhl
tpdlh
trise
tfall
iavg
avg_power_w
sim_ok

Dataset generation is implemented in:

code/generate_dataset.py


6. DATA PREPARATION

The dataset is loaded using Pandas. Rows are filtered using the sim_ok field, and the required feature and target columns are checked for valid values.

Input features:

WN
WP
VDD
CL
VTO_SHIFT

Prediction targets:

tpdhl
tpdlh
trise
tfall
avg_power_w

All 1500 simulations have sim_ok = True.

Two samples contain non-positive values for each of the propagation-delay targets tpdhl and tpdlh. These samples cannot be used for logarithmic regression because the logarithm of a non-positive value is undefined.

Target-specific sample counts are therefore:

Target          Samples used
tpdhl           1498
tpdlh           1498
trise           1500
tfall           1500
avg_power_w     1500

The dataset itself still contains all 1500 successful simulations.


7. MACHINE-LEARNING BASELINE

The initial surrogate model is a Gradient Boosting Regressor. A separate regression model is trained for each target quantity.

The baseline configuration is:

Model                    GradientBoostingRegressor
Number of estimators     200
Maximum depth            3
Learning rate            0.1
Random state              0
Test fraction             20%

The input features are standardized using StandardScaler. An 80/20 train-test split is used for evaluation.

The implementation is located in:

code/train_baseline_model.py


8. LOG-SPACE MODELING

The timing and power quantities modeled here are positive and span a relatively wide numerical range. The current implementation therefore models the targets in logarithmic space:

y_log = log(y)

The predicted values are transformed back to their original units before calculating RMSE and MAE.

A preliminary comparison showed that fitting timing quantities directly in raw seconds can perform substantially worse, while logarithmic modeling provides a much stronger regression relationship for the current dataset.

This is treated as an important modeling observation rather than an implementation detail. A more rigorous comparison between raw-space and log-space modeling will be included in later experiments.


9. BASELINE RESULTS

The latest Gradient Boosting baseline was evaluated on a held-out test set.

Target          R²             RMSE                  MAE                   Relative RMSE
tpdhl           0.991945       3.953193e-11          2.027215e-11          13.54%
tpdlh           0.988988       4.231690e-11          1.831914e-11          14.55%
trise           0.994086       7.800899e-11          3.582475e-11          13.23%
tfall           0.992201       5.854914e-11          3.122635e-11          10.87%
avg_power_w     0.997590       6.885861e-07          4.975517e-07          3.41%

The model achieves R² values between 0.988988 and 0.997590 across the five target quantities.

The highest R² is obtained for average power:

R² = 0.997590

The lowest R² is obtained for tpdlh:

R² = 0.988988


10. KEY FINDINGS

The initial baseline experiment provides three main observations.

First, the Gradient Boosting surrogate achieves strong predictive performance within the explored design space, with R² values close to 0.99 or higher for all five targets.

Second, logarithmic modeling provides a substantially stronger regression relationship for the timing quantities than direct modeling in raw seconds in the preliminary comparison.

Third, once trained, the surrogate can evaluate the five target quantities substantially faster than a new SPICE simulation under the current benchmark conditions.

These findings establish a useful baseline for the next stages of the research, but they do not demonstrate general replacement of SPICE.


11. SPICE AND MACHINE-LEARNING COMPARISON

The repository contains actual-versus-predicted plots for:

- tpdhl
- tpdlh
- trise
- tfall
- avg_power_w

These plots compare SPICE reference values with predictions from the Gradient Boosting surrogate on the held-out test data.

The figures are stored in:

results/figures/

The plots provide a visual assessment of how closely the surrogate follows the SPICE-derived values across the test set.


12. FEATURE IMPORTANCE

Feature-importance analysis was performed to investigate which circuit parameters contribute most strongly to the predictions of the Gradient Boosting models.

The five features are:

- WN
- WP
- VDD
- CL
- VTO_SHIFT

The resulting feature-importance plots are stored in results/figures/.

Feature importance should be interpreted as model-based importance within the explored dataset. It should not be treated as direct proof of physical causality.


13. INFERENCE SPEED

The latest benchmark measured machine-learning inference for all five target models at approximately:

46.7 microseconds/sample

The reference SPICE runtime is approximately:

24 milliseconds/sample

The approximate ratio is:

24 ms / 46.7 µs ≈ 514

Thus, the current benchmark indicates an evaluation-time advantage of approximately 514× for the measured comparison.

This comparison is deliberately qualified. The surrogate requires an already-trained model, and the original SPICE simulations are required to generate the training dataset. The reported ratio therefore compares the evaluation cost of an already-trained surrogate against one new SPICE evaluation.

A more rigorous timing experiment using repeated controlled measurements will be performed in a later phase.


14. REPRODUCIBILITY

The main Python dependencies are:

numpy
pandas
scipy
scikit-learn

ngspice must be installed separately according to the operating system.

Dataset generation:

python3 code/generate_dataset.py --n-samples 1500

Baseline training:

python3 code/train_baseline_model.py

The resulting baseline metrics are saved to:

results/baseline_results.csv

Test predictions are saved to:

results/test_predictions.csv


15. PROJECT STRUCTURE

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


16. CURRENT RESEARCH STATUS

Completed:

- Parameterized CMOS inverter-chain simulation
- Automated SPICE dataset generation
- 1500 successful simulation samples
- Dataset validation and preparation
- Gradient Boosting baseline model
- Log-space regression
- Held-out test evaluation
- R², RMSE, MAE, and relative-RMSE evaluation
- ML inference-speed measurement
- Actual-versus-predicted plots
- Feature-importance analysis
- Research documentation
- Git-based version control

The project currently provides a functioning end-to-end SPICE-to-machine-learning surrogate-model pipeline.


17. LIMITATIONS

17.1 Generic transistor models

The simulations use generic MOSFET models rather than a foundry-specific PDK. The results therefore demonstrate the machine-learning methodology rather than production-level device predictions.

17.2 Single circuit topology

The current study uses one three-stage CMOS inverter chain. Generalization to other circuit topologies has not yet been demonstrated.

17.3 Fixed channel length

Channel length is currently held fixed. Only WN, WP, VDD, CL, and VTO_SHIFT are varied.

17.4 Dataset size

The current dataset contains 1500 simulations. Larger datasets are required to determine how surrogate accuracy scales with training-data size.

17.5 Scalar prediction

The current baseline predicts five scalar metrics. It does not yet reproduce the complete transient waveform.

17.6 Generalization

The current results are valid only within the investigated parameter distribution. Extrapolation outside the training distribution has not yet been established.


18. PLANNED RESEARCH

18.1 Rigorous error analysis

Future experiments will include:

- RMSE and MAE analysis
- relative prediction error
- maximum-error samples
- worst-performing design points
- error-distribution plots
- controlled SPICE-versus-ML timing benchmarks

18.2 Model comparison

Gradient Boosting will be compared with additional suitable regression approaches, such as Random Forest and XGBoost or another boosting method.

Models will be compared using:

- R²
- RMSE
- MAE
- inference time

18.3 Dataset scaling

Model performance will be investigated for different training-set sizes, potentially including:

500
1000
1500
3000
5000

The objective is to determine how surrogate accuracy changes as additional SPICE data are provided.

18.4 Generalization testing

The trained model will be evaluated on previously unseen parameter combinations, including points near design-space boundaries and regions exhibiting nonlinear behavior.

18.5 Additional circuit topologies

A second circuit family will be investigated to determine whether the methodology generalizes beyond the three-stage inverter chain.

Potential candidates include a longer inverter chain, another CMOS logic circuit, or a simple differential pair.

18.6 Waveform-level surrogate modeling

The longer-term objective is to predict the complete transient output waveform rather than only five scalar metrics.

The intended workflow is:

Circuit parameters
        ↓
Machine-learning model
        ↓
Predicted v(out,t)
        ↓
Comparison against SPICE waveform

Waveform-level modeling represents a more complete surrogate for transient SPICE simulation and is a major direction for the next stage of the project.


19. RESEARCH SIGNIFICANCE

The central motivation is the computational cost of repeated circuit simulation during design-space exploration.

A surrogate model has the potential to provide rapid approximate evaluations after sufficient SPICE-generated training data have been collected.

The research therefore investigates the trade-off between:

SPICE simulation cost
        ↓
Training-data generation
        ↓
Model training
        ↓
Fast surrogate evaluation

The important research question is not simply whether machine learning can fit the current dataset. It is whether the cost of generating and training on SPICE data can be justified by the reduction in evaluation cost for subsequent circuit exploration.

This distinction motivates the planned work on dataset scaling, model comparison, generalization, and waveform-level prediction.


20. INTERPRETATION OF CURRENT RESULTS

The current R² values demonstrate strong predictive performance within the explored design space.

However, these results do not establish that the model can replace SPICE for arbitrary CMOS circuits.

Important questions remain:

- Does the accuracy remain high with larger datasets?
- Does the model generalize to unseen regions of the design space?
- Does it generalize to different circuit topologies?
- Which machine-learning model provides the best accuracy-speed trade-off?
- Can a model reproduce the complete transient waveform?
- Does surrogate modeling provide a meaningful end-to-end computational advantage after training-data generation and training costs are included?

The current work should therefore be described as a baseline surrogate-modeling study and research prototype, rather than a complete replacement for SPICE.


21. CONCLUSION

This project demonstrates an initial end-to-end workflow for AI-based surrogate modeling of SPICE circuit simulations.

A dataset of 1500 successful SPICE simulations was generated for a parameterized three-stage CMOS inverter chain. A Gradient Boosting regression model was then trained to predict propagation delay, rise time, fall time, and average power.

The baseline achieved R² values ranging from 0.988988 to 0.997590 across the five target quantities. The measured machine-learning inference time was approximately 46.7 microseconds per sample for all five target predictions. Compared with an approximately 24 ms SPICE reference time, the current benchmark corresponds to an evaluation-time ratio of approximately 514×.

These results provide evidence that a machine-learning surrogate can reproduce important SPICE-derived scalar metrics with high accuracy within the investigated design space while providing substantially faster inference than a new SPICE simulation.

The work remains an initial research baseline. The next stages will focus on rigorous error analysis, alternative model comparison, dataset scaling, generalization to additional circuits, and waveform-level surrogate modeling.


22. AUTHOR

Dashmesh Singh Chawla
Electronics Engineering — VLSI Design
Thapar Institute of Engineering and Technology

Research Area:
AI for Circuit Simulation / SPICE Surrogate Modeling / VLSI