# Edge Intelligence in Precision Agriculture

Code, results and configuration files for the manuscript:

**"Edge Intelligence in Precision Agriculture: A Multimodal IoT-Driven Framework for Predictive Eustress Management in Medicinal Plants, with Statistically Validated Reinforcement-Learning Control"** (MDPI *Technologies*, manuscript ID technologies-4671091).

## What this repository is, and is not

This is a **simulation study** of irrigation control plus a **proposed** (not validated) perception architecture.

- The control experiments (PPO, PID and baselines) run on a simulated eustress MDP and act on the simulated stress state directly.
- The multimodal fusion network (`src/fusion_network.py`) is **not trained** in this study (manuscript Section 3.3). It is provided as the proposed architecture. No accuracy metric is reported for it.
- The Raspberry Pi 5 benchmark times forward passes of the **untrained** network (random weights). It does not measure energy and does not demonstrate an end-to-end sensor-to-valve system (manuscript Section 3.4).
- No measurement of plants, stress or secondary metabolites is made. All stress thresholds are generic modelling choices.

## Contents

| Path | Content | Manuscript |
|---|---|---|
| `src/eustress_env.py` | Eustress-aware irrigation MDP (Equations 1–3) | Sec. 3.5 |
| `src/ppo_agent.py` | Self-contained NumPy PPO with handwritten forward/backward passes | Sec. 3.8 |
| `src/pid_controller.py` | PID controller and 180-configuration grid search | Sec. 3.10 |
| `src/baselines.py` | No irrigation, constant dose, rule-based controller | Sec. 3.9 |
| `src/fusion_network.py` | MobileNetV2 + LSTM + Transformer architecture (untrained) | Sec. 3.3 |
| `experiments/train_ppo.py` | PPO training | Table 9 |
| `eustress_round2_reproducibility.py` | Single-script supplementary analyses (PID winners, reward-matched tuning, capacity matrix, noise test) | Sec. 3.10, 5.4, Table 7 Panel B |
| `raspberry_pi/` | <configuration files and benchmark script> | Sec. 3.4 |
| `results/` | <per-scenario results (JSON/CSV) and trained policy checkpoints> | Tables 9–15 |

## Requirements

- PPO, PID, baselines and the supplementary script need only **Python 3.10+ with NumPy** (the supplementary script also uses pandas). No GPU and no deep-learning framework are required for the control experiments.
- `src/fusion_network.py` and the Raspberry Pi benchmark need PyTorch and torchvision.

```bash
pip install -r requirements.txt
```

## Reproducing the results

| Result in the paper | Command | Output |
|---|---|---|
| Table 9 (PPO training, seeds 1–5) | `python experiments/train_ppo.py --seed 1` | <path> |
| Tables 9–15 (evaluation, sweeps, OOD) | `<command>` | `results/<file>` |
| Table 7 Panel B, Sections 3.10 and 5.4 | `python eustress_round2_reproducibility.py` | `pid_provenance.csv`, `pid_matrix.csv`, `pid_water_aware_winners.csv`, `noise_test.csv` |
| Latency benchmark (Sec. 3.4) | `<command>` | `raspberry_pi/<file>.json` |

### Seeds used in the paper

| Purpose | Seeds |
|---|---|
| PPO training | 1–5 |
| Ablation and sweep | 1–3 |
| PID tuning | 70000–70009 |
| Convergence probes | 80000–80004 |
| Main evaluation | 90000–90029 |
| OOD (heatwave) evaluation | 95000–95014 |

## Supplementary script

`eustress_round2_reproducibility.py` re-implements the simulator, the four baselines and the PID grid search in one file and prints a reproduction report comparing its outputs with the values quoted in the manuscript. Expected last lines:

```
== Reproduction report: 50/50 values match the manuscript ==
```

Note: the manuscript does not specify the initial moisture and stress ranges or the rain depth. The script uses values calibrated once so that the four baseline rows of Table 9 are approximately reproduced (calibration check printed at the start of the run). Its absolute numbers therefore differ from Table 9 (for example PID with K_p = 4: 36.7 % band occupancy and 8.6 % distress against 32.8 % and 12.4 %). Table 9 itself comes from the original code in `src/`. Table 7, Panel B of the paper lists which numbers come from which code base.

## What is not included

- No trained perception model and no plant, sensor or metabolite data.
- No energy measurement for the Raspberry Pi.

## Licence

Code: <MIT or other>. See `LICENSE`.

## Citation

If you use this code, please cite the manuscript above. <Add the DOI when the paper is published.>

## Contact

Corresponding author: see the manuscript.
