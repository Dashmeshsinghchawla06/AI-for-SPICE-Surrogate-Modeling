"""
generate_dataset.py
--------------------
Generates a labeled dataset for the "AI Surrogate Model for SPICE Transient
Simulation" project.

Circuit under study: a 3-stage CMOS inverter chain (generic 130nm-like
LEVEL=1 MOSFET models), driven by a pulse input, with a capacitive load on
the final stage.

For each randomly sampled design point (transistor widths, supply voltage,
load capacitance, and an optional threshold-voltage shift to emulate
process variation), we run a real ngspice transient simulation and extract:

    - tpdhl, tpdlh : propagation delay (high->low, low->high), seconds
    - trise, tfall : 10%-90% rise / fall time at the output, seconds
    - avg_power_w  : average power drawn from the supply, watts

These become the labels an ML model will learn to predict directly from
the circuit parameters -- without running SPICE.

Usage:
    python3 generate_dataset.py --n-samples 500 --out dataset.csv
    python3 generate_dataset.py --n-samples 50 --save-waveforms --out small_test.csv

Requirements:
    ngspice (system binary, `apt install ngspice` on Debian/Ubuntu)
    numpy, pandas, scipy  (pip install numpy pandas scipy)
"""

import argparse
import os
import re
import shutil
import subprocess
import tempfile
import uuid

import numpy as np
import pandas as pd
from scipy.stats import qmc

# --------------------------------------------------------------------------
# Parameter ranges. Edit these to explore a different design space.
# --------------------------------------------------------------------------
PARAM_RANGES = {
    "WN":        (0.5e-6, 4.0e-6),   # NMOS width, meters
    "WP":        (1.0e-6, 8.0e-6),   # PMOS width, meters
    "VDD":       (0.9, 1.8),         # supply voltage, volts
    "CL":        (10e-15, 200e-15),  # load capacitance, farads
    "VTO_SHIFT": (-0.05, 0.05),      # threshold-voltage variation, volts
}
L_FIXED = 0.5e-6  # channel length held fixed for this study

NETLIST_TEMPLATE = """CMOS Inverter Chain - Delay/Power Characterization
.param WN={WN} WP={WP} L={L} VDD={VDD} CL={CL}
.param VTO_N={VTO_N} VTO_P={VTO_P}

.model NMOS_MODEL NMOS (LEVEL=1 VTO={{VTO_N}} KP=200u LAMBDA=0.02)
.model PMOS_MODEL PMOS (LEVEL=1 VTO={{VTO_P}} KP=100u LAMBDA=0.02)

Vdd vdd 0 DC {{VDD}}
Vin in 0 PULSE(0 {{VDD}} 1n 0.1n 0.1n 5n 12n)

M1n out1 in 0 0 NMOS_MODEL W={{WN}} L={{L}}
M1p out1 in vdd vdd PMOS_MODEL W={{WP}} L={{L}}
M2n out2 out1 0 0 NMOS_MODEL W={{WN}} L={{L}}
M2p out2 out1 vdd vdd PMOS_MODEL W={{WP}} L={{L}}
M3n out out2 0 0 NMOS_MODEL W={{WN}} L={{L}}
M3p out out2 vdd vdd PMOS_MODEL W={{WP}} L={{L}}
Cload out 0 {{CL}}

.tran 10p 20n

.measure tran tpdhl TRIG v(in) VAL={vhalf} RISE=1 TARG v(out) VAL={vhalf} FALL=1
.measure tran tpdlh TRIG v(in) VAL={vhalf} FALL=1 TARG v(out) VAL={vhalf} RISE=1
.measure tran trise TRIG v(out) VAL={v10} RISE=1 TARG v(out) VAL={v90} RISE=1
.measure tran tfall TRIG v(out) VAL={v90} FALL=1 TARG v(out) VAL={v10} FALL=1
.measure tran iavg AVG I(Vdd)

.control
run
{wrdata_line}
.endc
.end
"""

MEASURE_RE = re.compile(r"^(tpdhl|tpdlh|trise|tfall|iavg)\s*=\s*([-\d.eE+]+)", re.MULTILINE)


def sample_parameters(n_samples: int, seed: int = 0) -> pd.DataFrame:
    """Latin Hypercube sampling over PARAM_RANGES for good coverage of the
    design space with far fewer runs than a full grid sweep would need."""
    names = list(PARAM_RANGES.keys())
    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    unit_samples = sampler.random(n=n_samples)
    lo = np.array([PARAM_RANGES[k][0] for k in names])
    hi = np.array([PARAM_RANGES[k][1] for k in names])
    scaled = qmc.scale(unit_samples, lo, hi)
    df = pd.DataFrame(scaled, columns=names)
    return df


def build_netlist(params: dict, save_waveform_path: str | None) -> str:
    wrdata_line = f"wrdata {save_waveform_path} v(in) v(out)" if save_waveform_path else ""
    vdd = params["VDD"]
    return NETLIST_TEMPLATE.format(
        WN=params["WN"], WP=params["WP"], L=L_FIXED, VDD=vdd, CL=params["CL"],
        VTO_N=0.4 + params["VTO_SHIFT"], VTO_P=-0.4 - params["VTO_SHIFT"],
        vhalf=0.5 * vdd, v10=0.1 * vdd, v90=0.9 * vdd,
        wrdata_line=wrdata_line,
    )


def run_single(params: dict, workdir: str, save_waveforms: bool, sample_id: int) -> dict:
    waveform_path = None
    if save_waveforms:
        waveform_path = os.path.join(workdir, f"waveform_{sample_id}.txt")

    netlist_path = os.path.join(workdir, f"run_{sample_id}.cir")
    with open(netlist_path, "w") as f:
        f.write(build_netlist(params, waveform_path))

    result = subprocess.run(
        ["ngspice", "-b", netlist_path],
        capture_output=True, text=True, timeout=30,
    )

    metrics = {"tpdhl": np.nan, "tpdlh": np.nan, "trise": np.nan,
               "tfall": np.nan, "iavg": np.nan}
    for match in MEASURE_RE.finditer(result.stdout):
        metrics[match.group(1)] = float(match.group(2))

    metrics["avg_power_w"] = abs(metrics["iavg"] * params["VDD"]) if not np.isnan(metrics["iavg"]) else np.nan
    metrics["sim_ok"] = not any(np.isnan(v) for k, v in metrics.items() if k != "iavg")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Generate CMOS inverter-chain SPICE dataset.")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--out", type=str, default="dataset.csv")
    parser.add_argument("--save-waveforms", action="store_true",
                         help="Also save full v(in)/v(out) transient waveform per sample "
                              "(needed for the waveform-level surrogate model, not the baseline).")
    parser.add_argument("--waveform-dir", type=str, default="waveforms")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if shutil.which("ngspice") is None:
        raise SystemExit("ngspice not found on PATH. Install it first (e.g. `apt install ngspice`).")

    param_df = sample_parameters(args.n_samples, seed=args.seed)

    if args.save_waveforms:
        os.makedirs(args.waveform_dir, exist_ok=True)

    workdir = tempfile.mkdtemp(prefix="spice_sweep_")
    print(f"Working directory: {workdir}")

    rows = []
    n_failed = 0
    for i, row in param_df.iterrows():
        params = row.to_dict()
        try:
            metrics = run_single(
                params, workdir,
                save_waveforms=args.save_waveforms,
                sample_id=i,
            )
        except subprocess.TimeoutExpired:
            metrics = {"tpdhl": np.nan, "tpdlh": np.nan, "trise": np.nan,
                       "tfall": np.nan, "avg_power_w": np.nan, "sim_ok": False}

        if not metrics["sim_ok"]:
            n_failed += 1
        else:
            if args.save_waveforms:
                src = os.path.join(workdir, f"waveform_{i}.txt")
                dst = os.path.join(args.waveform_dir, f"waveform_{i}.txt")
                if os.path.exists(src):
                    shutil.move(src, dst)

        record = {**params, "sample_id": i, **metrics}
        rows.append(record)

        if (i + 1) % 50 == 0 or (i + 1) == args.n_samples:
            print(f"  {i + 1}/{args.n_samples} runs complete ({n_failed} failed so far)")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    shutil.rmtree(workdir, ignore_errors=True)

    n_ok = int(out_df["sim_ok"].sum())
    print(f"\nDone. {n_ok}/{args.n_samples} simulations succeeded.")
    print(f"Dataset written to {args.out}")
    if args.save_waveforms:
        print(f"Waveforms written to {args.waveform_dir}/")


if __name__ == "__main__":
    main()
