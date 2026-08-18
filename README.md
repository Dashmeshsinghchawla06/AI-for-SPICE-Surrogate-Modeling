# AI Surrogate Model for SPICE Transient Simulation

**Research question**: Can a model trained on SPICE-generated data predict the
transient behavior of a parameterized CMOS inverter chain fast enough and
accurately enough to replace SPICE for early-stage design-space exploration?

This is a working, tested starting point for that project — not a toy demo.
Every number below came from an actual `ngspice` run on this machine.

## What's in here

- `generate_dataset.py` — runs real ngspice transient simulations across a
  Latin-Hypercube-sampled design space and extracts labeled metrics.
- `train_baseline_model.py` — trains the Step 2 baseline surrogate model
  (scalar metrics only) and reports accuracy + speedup.
- `dataset_1500.csv` — a 1500-sample dataset already generated (took ~35–40s
  to generate on this machine; regenerate larger with `--n-samples`).

## Circuit under study

A 3-stage CMOS inverter chain (generic LEVEL=1 MOSFET models, ~130nm-scale
parameters) driven by a pulse input, with a capacitive load on the final
stage. Swept parameters:

| Parameter | Range | Meaning |
|---|---|---|
| WN | 0.5–4.0 µm | NMOS width |
| WP | 1.0–8.0 µm | PMOS width |
| VDD | 0.9–1.8 V | Supply voltage |
| CL | 10–200 fF | Output load capacitance |
| VTO_SHIFT | ±0.05 V | Threshold voltage shift (emulates process variation) |

Predicted labels: `tpdhl`, `tpdlh` (propagation delay), `trise`, `tfall`
(10–90% rise/fall time), `avg_power_w` (average supply power).

## Results so far (1500-sample dataset, gradient boosting baseline)

| Target | R² (log-space) | Relative RMSE |
|---|---|---|
| tpdhl | 0.992 | 13.5% |
| tpdlh | 0.989 | 14.6% |
| trise | 0.994 | 13.2% |
| tfall | 0.992 | 10.9% |
| avg_power_w | 0.998 | 3.4% |

**Speed**: ngspice takes ~24 ms per run on this machine for this circuit.
The trained model predicts all 5 targets for a new design point in ~130 µs
(batched) — roughly a **180x speedup**, with R² ≈ 0.99 across the board.

### A real modeling finding, worth a paragraph in your report
Fitting the delay/rise/fall targets in raw seconds gives **negative R²**
(worse than just predicting the mean) — don't be alarmed if you see this
first. The fix: fit in **log space**. RC-style delays are naturally
multiplicative in the design parameters (delay ∝ R·C, and both R and C
depend multiplicatively on W, L, Vdd, etc.), so their distribution is
closer to log-normal than normal, and a small number of large-delay,
near-threshold operating points otherwise dominate the raw-value variance.
This single change took R² from ~0 to ~0.99. That's a genuine, reportable
result about the structure of the problem — include it, don't hide it.

## How to reproduce

```bash
# 1. Install dependencies (ngspice is a system package, not pip)
apt-get install -y ngspice
pip install numpy pandas scipy scikit-learn --break-system-packages

# 2. Generate a dataset (scale up n-samples for your real experiments)
python3 generate_dataset.py --n-samples 1500 --out dataset.csv --seed 42

# 3. Train and evaluate the baseline model
python3 train_baseline_model.py --dataset dataset.csv
```

To also save full transient waveforms (needed for the harder, Step 3
waveform-level model rather than just scalar metrics):

```bash
python3 generate_dataset.py --n-samples 200 --save-waveforms --out small.csv
```

## Suggested next steps (in order)

1. **Scale the dataset up** (5,000–10,000 samples; runtime is roughly
   linear, so this is ~10–15 minutes) and re-run the baseline — confirm
   accuracy holds or improves.
2. **Try a second circuit family** (e.g. a 5-stage chain, or a simple
   differential pair) to show the approach generalizes, not just a fit to
   one topology.
3. **Waveform-level model (the actual novel part)**: instead of predicting
   5 scalar numbers, predict the full `v(out)` waveform from the
   parameters (e.g. a small sequence model, or a network predicting
   waveform samples at fixed time points). This is what turns "a
   regression fit" into "a surrogate for SPICE."
4. **Write it up**: motivation, dataset generation methodology, baseline
   result (including the log-space finding above), waveform-level result,
   accuracy/speed benchmark table, limitations (single circuit family,
   generic models rather than a foundry PDK). 6–10 pages is enough for a
   first version to show a professor.
5. **Take the working baseline (this repo) to a professor now** — you
   don't need to wait for the waveform-level model to be done. A working
   prototype with a real result is the strongest opening for a
   supervision conversation.

## Known limitations (be upfront about these in any write-up)

- Generic LEVEL=1 MOSFET models, not a real foundry PDK — good enough to
  demonstrate the ML methodology, not to make device-physics claims.
- Single circuit topology (3-stage inverter chain) — generalization to
  other topologies is future work, not yet demonstrated.
- Channel length L is held fixed — only 5 of the many possible design
  parameters are swept.
