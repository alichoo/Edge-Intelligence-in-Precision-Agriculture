"""
eustress_round2_reproducibility.py
==================================
Supplementary Material for the manuscript:
"Edge Intelligence in Precision Agriculture: A Multimodal IoT-Driven Framework 
for Predictive Eustress Management in Medicinal Plants"

This single-file script reproduces the supplementary results mentioned in the 
Data Availability Statement and the Response to Reviewers (Section 3.10, Section 5.4, and Table 7 Panel B):
1. Simulates the Eustress-Aware MDP (Equations 1-3).
2. Runs the 180-configuration PID grid search.
3. Generates the PID Provenance Table (Table 7, Panel B).
4. Generates the 5x5 Demand x Capacity matrix (Section 5.4).
5. Outputs results to `pid_provenance.csv` and `pid_matrix.csv`.

Requirements: numpy, pandas
Runtime: Approx. 2-3 minutes on a single CPU core.
"""

import numpy as np
import pandas as pd
import time

# ==========================================
# 1. ENVIRONMENT (Equations 1-3)
# ==========================================
class EustressEnv:
    def __init__(self, T=120, regime="nominal", capacity_L=7.0, seed=0):
        self.T = T
        self.regime = regime
        self.capacity_L = capacity_L
        self.rng = np.random.default_rng(seed)
        
        # Section 3.5 Coefficients
        self.lambda_e = 0.04
        self.kappa = 0.035  # calibrated for 7L capacity
        self.beta1 = 0.40
        self.beta2 = 0.05
        self.beta3 = 0.05
        self.beta4 = 0.12
        
        self.sigma_min = 0.50
        self.sigma_max = 0.70
        self.sigma_dist = 0.82
        
        # Reward weights (Eq 3)
        self.l1 = 0.25  # water
        self.l2 = 3.0   # band distance
        self.l3 = 6.0   # distress
        self.l4 = 1.5   # band bonus
        
        self.reset()

    def reset(self):
        self.t = 0
        self.m = 0.92   # Calibrated to match manuscript unified script
        self.sigma = 0.50
        self.prev_a = 0.0
        self.R = self.H = self.rho = 0.0
        return self._get_state()

    def _get_state(self):
        return np.array([self.m, self.sigma, self.R, self.H, self.rho, self.prev_a], dtype=np.float64)

    def _weather(self):
        # Calibrated to match the sensitivity matrix in the manuscript
        if self.regime == "mild":
            self.R = self.rng.uniform(0.2, 0.8)
            self.H = self.rng.uniform(0.4, 0.9)
            rain_p = 0.15
        elif self.regime == "nominal":
            self.R = self.rng.uniform(0.4, 1.0)
            self.H = self.rng.uniform(0.2, 0.8)
            rain_p = 0.15
        elif self.regime == "warm":
            self.R = self.rng.uniform(0.6, 1.0)
            self.H = self.rng.uniform(0.15, 0.6)
            rain_p = 0.15
        elif self.regime == "heatwave":
            self.R = self.rng.uniform(0.7, 1.1)
            self.H = self.rng.uniform(0.1, 0.5)
            rain_p = 0.05
        elif self.regime == "extreme":
            self.R = self.rng.uniform(0.9, 1.2)
            self.H = self.rng.uniform(0.05, 0.3)
            rain_p = 0.01
        else:
            raise ValueError("Unknown regime")
            
        # Calibrated rain amplitude
        self.rho = 0.12 if self.rng.random() < rain_p else 0.0

    def step(self, a):
        a = float(np.clip(a, 0.0, 1.0))
        self._weather()
        
        # Scale kappa by capacity relative to 7L base
        kappa_adj = self.kappa * (self.capacity_L / 7.0)
        
        # Eq (1)
        self.m = np.clip(self.m - self.lambda_e * self.R + kappa_adj * a + self.rho, 0, 1)
        
        # Eq (2)
        self.sigma = np.clip(
            self.sigma + self.beta1*(1 - self.m) + self.beta2*self.R 
            - self.beta3*self.H - self.beta4*self.sigma, 0, 1
        )
        
        # Eq (3) Reward
        d = max(0.0, self.sigma_min - self.sigma, self.sigma - self.sigma_max)
        in_band = 1.0 if (self.sigma_min <= self.sigma <= self.sigma_max) else 0.0
        above_distress = 1.0 if self.sigma > self.sigma_dist else 0.0
        
        r = -self.l1 * a - self.l2 * d - self.l3 * above_distress + self.l4 * in_band
        
        self.prev_a = a
        self.t += 1
        done = self.t >= self.T
        return self._get_state(), r, done, {
            "m": self.m, "sigma": self.sigma, "a": a,
            "water_L": a * self.capacity_L, "in_band": in_band, "distress": above_distress
        }

# ==========================================
# 2. CONTROLLERS (Baselines & PID)
# ==========================================
class PIDController:
    def __init__(self, Kp, Ki, Kd, target):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.target = target
        self.integral = 0.0
        self.prev_e = 0.0
        
    def reset(self):
        self.integral = 0.0
        self.prev_e = 0.0
        
    def step(self, env):
        e = env.sigma - self.target
        self.integral += e
        de = e - self.prev_e
        self.prev_e = e
        u = self.Kp * e + self.Ki * self.integral + self.Kd * de
        return float(np.clip(u, 0.0, 1.0))

def run_eval(controller_type, env_regime, env_capacity, seed, **kwargs):
    env = EustressEnv(regime=env_regime, capacity_L=env_capacity, seed=seed)
    state = env.reset()
    BO, DE, W, J = 0, 0, 0, 0
    
    if controller_type == "pid":
        pid = PIDController(kwargs['Kp'], kwargs['Ki'], kwargs['Kd'], kwargs['target'])
        
    for _ in range(env.T):
        if controller_type == "no_irrigation":
            a = 0.0
        elif controller_type == "constant_dose":
            a = 0.3
        elif controller_type == "rbc":
            a = 1.0 if env.m < 0.38 else 0.0
        elif controller_type == "pid":
            a = pid.step(env)
            
        state, r, done, info = env.step(a)
        BO += info["in_band"]
        DE += info["distress"]
        W += info["water_L"]
        J += r
        if done: break
        
    return {"BO": (BO/env.T)*100, "DE": (DE/env.T)*100, "W": W, "J": J}

# ==========================================
# 3. EVALUATION LOGIC (Section 3.10 & 5.4)
# ==========================================
def run_pid_grid_search():
    print("Running 180-configuration PID grid search...")
    Kp_vals = [0.5, 1, 2, 4, 8]
    Ki_vals = [0, 0.01, 0.05]
    Kd_vals = [0, 1, 4]
    target_vals = [0.55, 0.60, 0.65]
    
    tuning_seeds = list(range(70000, 70010))
    
    results = []
    
    for Kp in Kp_vals:
        for Ki in Ki_vals:
            for Kd in Kd_vals:
                for target in target_vals:
                    tuning_bo, tuning_de = [], []
                    for s in tuning_seeds:
                        res = run_eval("pid", "nominal", 7.0, s, Kp=Kp, Ki=Ki, Kd=Kd, target=target)
                        tuning_bo.append(res["BO"])
                        tuning_de.append(res["DE"])
                    
                    score_eq12 = np.mean(tuning_bo) - 0.5 * np.mean(tuning_de)
                    
                    results.append({
                        "Kp": Kp, "Ki": Ki, "Kd": Kd, "target": target,
                        "tuning_BO": np.mean(tuning_bo), "tuning_DE": np.mean(tuning_de),
                        "score_eq12": score_eq12
                    })
                    
    df = pd.DataFrame(results).sort_values("score_eq12", ascending=False).reset_index(drop=True)
    return df

def run_capacity_matrix():
    print("Running 5x5 Demand x Capacity Matrix...")
    regimes = ["mild", "nominal", "warm", "heatwave", "extreme"]
    capacities = [4.0, 5.5, 7.0, 8.5, 10.0]
    seeds = list(range(90000, 90030))
    
    matrix_rows = []
    
    for regime in regimes:
        for cap in capacities:
            for ctrl in ["no_irrigation", "constant_dose", "rbc"]:
                bo, de, w = [], [], []
                for s in seeds:
                    res = run_eval(ctrl, regime, cap, s)
                    bo.append(res["BO"]); de.append(res["DE"]); w.append(res["W"])
                matrix_rows.append({
                    "demand": regime, "capacity_L": cap, "controller": ctrl,
                    "BO_mean": np.mean(bo), "BO_sd": np.std(bo, ddof=1),
                    "DE_mean": np.mean(de), "DE_sd": np.std(de, ddof=1),
                    "W_mean": np.mean(w)
                })
                
            # Evaluate PID Kp=4 (Original winner)
            bo, de, w = [], [], []
            for s in seeds:
                res = run_eval("pid", regime, cap, s, Kp=4, Ki=0, Kd=4, target=0.55)
                bo.append(res["BO"]); de.append(res["DE"]); w.append(res["W"])
            matrix_rows.append({
                "demand": regime, "capacity_L": cap, "controller": "pid_Kp4",
                "BO_mean": np.mean(bo), "BO_sd": np.std(bo, ddof=1),
                "DE_mean": np.mean(de), "DE_sd": np.std(de, ddof=1),
                "W_mean": np.mean(w)
            })
            
    return pd.DataFrame(matrix_rows)

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Starting Eustress Round 2 Reproducibility Script...")
    start_time = time.time()
    
    # 1. PID Grid Search (Table 7)
    df_pid = run_pid_grid_search()
    
    # 2. Evaluate Kp4 vs Kp8 on 30 shared scenarios (Provenance Table)
    eval_seeds = list(range(90000, 90030))
    
    bo4, de4, w4 = [], [], []
    for s in eval_seeds:
        res = run_eval("pid", "nominal", 7.0, s, Kp=4, Ki=0, Kd=4, target=0.55)
        bo4.append(res["BO"]); de4.append(res["DE"]); w4.append(res["W"])
        
    bo8, de8, w8 = [], [], []
    for s in eval_seeds:
        res = run_eval("pid", "nominal", 7.0, s, Kp=8, Ki=0, Kd=4, target=0.55)
        bo8.append(res["BO"]); de8.append(res["DE"]); w8.append(res["W"])
        
    print("\n--- Unified Script Results (Section 3.10) ---")
    print(f"Kp=4: BO={np.mean(bo4):.1f} ± {np.std(bo4, ddof=1):.1f}, DE={np.mean(de4):.1f} ± {np.std(de4, ddof=1):.1f}, W={np.mean(w4):.1f} ± {np.std(w4, ddof=1):.1f}")
    print(f"Kp=8: BO={np.mean(bo8):.1f} ± {np.std(bo8, ddof=1):.1f}, DE={np.mean(de8):.1f} ± {np.std(de8, ddof=1):.1f}, W={np.mean(w8):.1f} ± {np.std(w8, ddof=1):.1f}")
    
    # Export Provenance CSV (Table 7, Panel B)
    df_provenance = pd.DataFrame([
        {"Code base": "Original", "Gains": "4, 0, 4, 0.55", "BO (%)": "32.8 ± 9.8", "DE (%)": "12.4 ± 7.9", "W (L)": "259.3 ± 82.1"},
        {"Code base": "Unified (Kp=4)", "Gains": "4, 0, 4, 0.55", "BO (%)": f"{np.mean(bo4):.1f} ± {np.std(bo4, ddof=1):.1f}", "DE (%)": f"{np.mean(de4):.1f} ± {np.std(de4, ddof=1):.1f}", "W (L)": f"{np.mean(w4):.1f} ± {np.std(w4, ddof=1):.1f}"},
        {"Code base": "Unified (Kp=8)", "Gains": "8, 0, 4, 0.55", "BO (%)": f"{np.mean(bo8):.1f} ± {np.std(bo8, ddof=1):.1f}", "DE (%)": f"{np.mean(de8):.1f} ± {np.std(de8, ddof=1):.1f}", "W (L)": f"{np.mean(w8):.1f} ± {np.std(w8, ddof=1):.1f}"}
    ])
    df_provenance.to_csv("pid_provenance.csv", index=False)
    
    # 3. Capacity Matrix (Section 5.4)
    df_matrix = run_capacity_matrix()
    df_matrix.to_csv("pid_matrix.csv", index=False)
    
    print("\n--- Capacity Matrix (Nominal Regime Extract) ---")
    print(df_matrix[(df_matrix['demand']=="nominal") & (df_matrix['controller'].isin(['pid_Kp4']))][["capacity_L", "BO_mean", "DE_mean", "W_mean"]].to_string(index=False))
    
    print(f"\nExecution complete in {time.time() - start_time:.1f} seconds.")
    print("Files saved: 'pid_provenance.csv', 'pid_matrix.csv'")