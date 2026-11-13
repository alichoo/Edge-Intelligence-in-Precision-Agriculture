# Edge Intelligence in Precision Agriculture

Supplementary code and results for the manuscript:

**"Edge Intelligence in Precision Agriculture: A Multimodal IoT-Driven Framework for Predictive Eustress Management in Medicinal Plants, with Statistically Validated Reinforcement-Learning Control"** (MDPI *Technologies*, manuscript ID technologies-4671091).

## What this repository is, and is not

This is a **simulation study** of irrigation control plus a **proposed** (not validated) perception architecture.

- The control experiments (PPO, PID and baselines) run on a simulated eustress MDP and act on the simulated stress state directly.
- The multimodal fusion network is **not trained** in this study (manuscript Section 3.3). It is provided as the proposed architecture. No accuracy metric is reported for it.
- The Raspberry Pi 5 benchmark times forward passes of the **untrained** network (random weights). It does not measure energy and does not demonstrate an end-to-end sensor-to-valve system (manuscript Section 3.4).
- No measurement of plants, stress or secondary metabolites is made. All stress thresholds are generic modelling choices.

## Contents of this repository

| File | Content | Manuscript |
|---|---|---|
| `eustress_round2_reproducibility.py` | Single-script supplementary analyses (PID winners, reward-matched tuning, capacity matrix, noise test) | Sec. 3.10, 5.4, Table 7 Panel B |
| `pid_grid_tuning.csv` | Raw output of the 180-configuration PID grid search | Table 7 Panel A |
| `pid_provenance.csv` | Provenance of every PID result reported in the paper | Table 7 Panel B |
| `pid_water_aware_winners.csv` | Water-aware and Eq. (3)-matched PID winners | Sec. 3.10 |
| `pid_matrix.csv` | 5 × 5 demand-by-capacity matrix for the four baselines and PID | Sec. 5.4 |
| `noise_test.csv` | Observation-noise sensitivity results | Sec. 5.4 |

*Note: The original simulation environment, the NumPy PPO implementation, the trained policy checkpoints, and the per-scenario evaluation results behind Tables 9–15 are available from the corresponding author on reasonable request (as stated in the manuscript's Data Availability Statement).*

## Requirements

The supplementary script needs only **Python 3.10+ with NumPy and Pandas**. No GPU and no deep-learning framework are required.

```bash
pip install numpy pandas
