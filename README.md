# Edge Intelligence in Precision Agriculture

Reproducibility code for the manuscript:
**"Edge Intelligence in Precision Agriculture: A Multimodal IoT-Driven Framework for Predictive Eustress Management in Medicinal Plants, with Statistically Validated Reinforcement-Learning Control"**

## Overview
This repository contains the simulation environment, the NumPy-only PPO implementation, the PID baseline, and the multimodal fusion network architecture described in the paper. 

**Important:** The fusion network (`src/fusion_network.py`) is **not trained** in this study (see Section 3.3 of the manuscript). It is provided as the proposed architecture for future work. The control experiments isolate the PPO policy by acting on the simulated stress state directly.

## Contents
- `src/eustress_env.py`: The eustress-aware irrigation MDP (Equations 1–3).
- `src/ppo_agent.py`: Self-contained NumPy PPO with handwritten forward/backward passes (Section 3.8).
- `src/pid_controller.py`: PID controller and grid search logic (Section 3.10).
- `src/fusion_network.py`: MobileNetV2 + LSTM + Transformer architecture (Section 3.3).
- `src/baselines.py`: No irrigation, constant dose, and rule-based controllers.

## Usage

### 1. Environment & PPO Training (Table 9)
```bash
python experiments/train_ppo.py --seed 1
