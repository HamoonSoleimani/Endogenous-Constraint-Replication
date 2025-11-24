# The Endogenous Constraint: Replication Package

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the source code for the **Forensic Econometric Suite** used in the paper:

> **The Endogenous Constraint: Hysteresis, Stagflation, and the Structural Inhibition of Monetary Velocity in the Bitcoin Network**  
> *Hamoon Soleimani (2025)*

## 📄 Abstract

Bitcoin operates as a macroeconomic paradox: it combines a fixed monetary schedule with a rigid block weight limit. This research proposes the **Endogenous Constraint Hypothesis**, positing that internal architectural constraints generate a non-linear negative feedback loop between network friction and monetary velocity. This codebase provides the empirical validation tools, including the Transaction Cost Index (TCI) generator, threshold regression validators, and hysteresis topology visualizers.

## 🛠 Features

The `main.py` script launches a GUI-based research suite ("Forensic Audit Suite v7.0") that performs the following functions described in the paper:

1. **Data Harmonization:** Parses CSV/JSON data and interpolates block-height data to daily time series.
2. **Forensic Engine:**
   - Calculates the **Transaction Cost Index (TCI)** (Paper Section 2.3).
   - Identifies **Retail Insolvency** periods (Paper Section 5.3).
   - Detects structural regimes (Normal vs. Shock).
3. **Econometric Validator:**
   - Runs **Threshold Regression** (Hansen, 2000) to calculate Net Damage.
   - Performs **Bootstrap Validation** (5,000 iterations).
   - Executes **Granger Causality** tests (Friction → Velocity).
4. **Visualization:** Generates the specific figures used in the paper (Phase-State Hysteresis, Stagflation Matrix, Sensitivity Curves).

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher.
- A desktop environment (the suite uses `tkinter` for the GUI).

### Installation

```bash
git clone https://github.com/hamoon-soleimani/endogenous-constraint-replication.git
cd endogenous-constraint-replication
pip install -r requirements.txt
```

### Usage

Run the application:

```bash
python main.py
```

**Mode A (Synthetic Data):** Click "GENERATE DUMMY DATA" to see the logic in action using the synthetic generator described in the code. This will produce all plots immediately.

**Mode B (Replication):**

1. Load your CSV files (Price, Fees, Volume, etc.) using the sidebar "LOAD" buttons.
2. Adjust the "Friction Threshold %" slider (Default: 90% per Paper Section 3.2).
3. Click "RUN ANALYSIS".
4. Click "EXPORT REPORT" to generate the HTML forensic log.

## 🔬 Methodology Mapping

This codebase directly implements the mathematical framework defined in the research:

| Paper Section | Concept | Python Class/Method |
|---------------|---------|---------------------|
| Sec 2.3 | Transaction Cost Index (TCI) | `ForensicEngine._calc_tci` |
| Sec 3.2 | Threshold Regression | `EconometricValidator.run_threshold_regression` |
| Sec 5.2 | Hysteresis (Ewing Loop) | `ForensicEngine._calc_phase_state` |
| Appx B | Bootstrap Confidence Intervals | `EconometricValidator.run_bootstrap_ci` |

## 📦 Dependencies

- **pandas** - Time series manipulation.
- **numpy** - Vectorized calculations.
- **statsmodels** - OLS regression and Granger Causality tests.
- **scipy** - Welch's t-tests and statistical functions.
- **matplotlib / seaborn** - Visualization generation.
- **scikit-learn** - Data preprocessing.

## 📚 Citation

If you use this code in your research, please cite the paper:

```bibtex
@article{soleimani2025endogenous,
  title={The Endogenous Constraint: Hysteresis, Stagflation, and the Structural Inhibition of Monetary Velocity in the Bitcoin Network},
  author={Soleimani, Hamoon},
  year={2025},
  month={November},
  note={Available at GitHub: https://github.com/hamoon-soleimani/endogenous-constraint-replication}
}
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
