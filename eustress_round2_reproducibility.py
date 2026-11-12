"""
eustress_round2_reproducibility.py
==================================
Single-script reproducibility code for the supplementary PID analyses of
"Edge Intelligence in Precision Agriculture: A Multimodal IoT-Driven Framework for
Predictive Eustress Management in Medicinal Plants, with Statistically Validated
Reinforcement-Learning Control" (Sections 3.10 and 5.4, Table 7 Panel B).

What it does (all on one code base, CPU only, ~1-2 minutes):
  1. Implements the eustress MDP simulator (Equations 1-3) and four baselines
     (no irrigation, constant dose, RBC, PID).
  2. Calibration check: compares the four baseline rows with Table 9.
  3. Runs the 180-configuration PID grid search on tuning seeds 70000-70009 and selects
     winners under (a) Equation (12), (b) the water-aware score for mu in {0,0.5,1,2,4,8},
     (c) the episodic return J of Equation (3) at lambda3 = 6 and lambda3 = 2.
  4. Evaluates the Kp=4 and Kp=8 gain sets on the 30 evaluation seeds 90000-90029
     (Table 7 Panel B rows, mean and SD, plus J).
  5. Runs the 5 x 5 demand-by-capacity matrix (Section 5.4) for the four baselines and
     the PID with Kp=4 and Kp=8.
  6. Observation-noise sensitivity of the PID (how much stress-estimation error it tolerates).
  7. Compares the key outputs with the values quoted in the manuscript.

Not included: the NumPy PPO training code and the Table 9-15 evaluation of PPO; those are
in the separate files of the repository.

IMPORTANT: initial moisture/stress ranges and the rain depth are not specified in the
manuscript equations. The values in ASSUMPTIONS were calibrated once so that the four
baseline rows of Table 9 are approximately reproduced (see the calibration check).

Requirements: numpy, pandas.   Outputs: pid_grid_tuning.csv, pid_water_aware_winners.csv,
pid_provenance.csv, pid_matrix.csv, noise_test.csv
"""
import itertools, time
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ 1. simulator
T = 120
KAPPA7, LAM_E = 0.035, 0.04                 # kappa is defined for a 7 L full dose
B1, B2, B3, B4 = 0.40, 0.05, 0.05, 0.12
SMIN, SMAX, SDIST = 0.50, 0.70, 0.82
REGIMES = {                                  # (radiation range, humidity range, rain probability)
    "milder":          dict(R=(0.3, 0.8), H=(0.3, 0.9), p_rain=0.20),
    "nominal":         dict(R=(0.4, 1.0), H=(0.2, 0.8), p_rain=0.15),
    "warm":            dict(R=(0.55, 1.05), H=(0.15, 0.65), p_rain=0.10),
    "heatwave":        dict(R=(0.7, 1.1), H=(0.1, 0.5), p_rain=0.05),
    "beyond-heatwave": dict(R=(0.8, 1.2), H=(0.05, 0.4), p_rain=0.03),
}
# ---- ASSUMPTIONS (not given in the manuscript; calibrated once against Table 9) ----
M0_RANGE, S0_RANGE, RAIN_AMT = (0.60, 0.66), (0.45, 0.73), (0.09, 0.24)
RBC_THRESHOLD, RBC_DOSE, CONST_DOSE = 0.38, 1.0, 0.3
# -----------------------------------------------------------------------------------
LAM = dict(l1=0.25, l2=3.0, l4=1.5)          # lambda3 is varied: 6 (main) and 2 (balanced)

def make_scenarios(seeds, regime="nominal"):
    reg = REGIMES[regime]; N = len(seeds)
    R = np.empty((N, T)); H = np.empty((N, T)); RHO = np.empty((N, T)); m0 = np.empty(N); s0 = np.empty(N)
    for i, sd in enumerate(seeds):
        g = np.random.default_rng(int(sd))
        R[i] = g.uniform(*reg["R"], T); H[i] = g.uniform(*reg["H"], T)
        RHO[i] = (g.random(T) < reg["p_rain"]) * g.uniform(*RAIN_AMT, T)
        m0[i] = g.uniform(*M0_RANGE); s0[i] = g.uniform(*S0_RANGE)
    return dict(R=R, H=H, RHO=RHO, m0=m0, s0=s0, N=N)

def step(m, s, a, R, H, rho, cap):
    m1 = np.clip(m - LAM_E * R + KAPPA7 * (cap / 7.0) * a + rho, 0, 1)                 # Eq. (1)
    s1 = np.clip(s + B1 * (1 - m1) + B2 * R - B3 * H - B4 * s, 0, 1)                    # Eq. (2)
    return m1, s1

def reward(a, s1, l3):                                                                 # Eq. (3)
    d = np.maximum(0, np.maximum(SMIN - s1, s1 - SMAX))
    return -LAM["l1"] * a - LAM["l2"] * d - l3 * (s1 > SDIST) + LAM["l4"] * ((s1 >= SMIN) & (s1 <= SMAX))

# ------------------------------------------------------------------ 2. controllers
class NoIrrigation:
    def __call__(self, m, s, t): return np.zeros_like(m)
class ConstantDose:
    def __call__(self, m, s, t): return np.full_like(m, CONST_DOSE)
class RBC:
    def __call__(self, m, s, t): return np.where(m < RBC_THRESHOLD, RBC_DOSE, 0.0)
class PID:
    def __init__(self, kp, ki, kd, target): self.k = (kp, ki, kd); self.target = target
    def reset(self, N): self.i = np.zeros(N); self.e_prev = None
    def __call__(self, m, s, t):
        e = s - self.target
        if self.e_prev is None: self.e_prev = e.copy()
        self.i += e; kp, ki, kd = self.k
        u = kp * e + ki * self.i + kd * (e - self.e_prev); self.e_prev = e.copy()
        return np.clip(u, 0, 1)

def run(policy, sc, cap=7.0, obs_noise=0.0, noise_seed=0):
    """Returns per-scenario BO (%), DE (%), W (L), J6, J2."""
    N = sc["N"]; m = sc["m0"].copy(); s = sc["s0"].copy(); g = np.random.default_rng(noise_seed)
    bo = np.zeros(N); de = np.zeros(N); W = np.zeros(N); J6 = np.zeros(N); J2 = np.zeros(N)
    if hasattr(policy, "reset"): policy.reset(N)
    for t in range(T):
        s_obs = np.clip(s + g.normal(0, obs_noise, N), 0, 1) if obs_noise > 0 else s
        a = np.clip(policy(m, s_obs, t), 0, 1)
        m, s = step(m, s, a, sc["R"][:, t], sc["H"][:, t], sc["RHO"][:, t], cap)
        bo += (s >= SMIN) & (s <= SMAX); de += s > SDIST; W += a * cap
        J6 += reward(a, s, 6.0); J2 += reward(a, s, 2.0)
    return dict(BO=100 * bo / T, DE=100 * de / T, W=W, J6=J6, J2=J2)

def ms(x): return f"{np.mean(x):.1f} ± {np.std(x, ddof=1):.1f}"
TUNE = make_scenarios(range(70000, 70010)); EVAL = make_scenarios(range(90000, 90030))
KP4, KP8 = (4, 0, 4, 0.55), (8, 0, 4, 0.55)

if __name__ == "__main__":
    t0 = time.time(); checks = []
    def check(label, got, want, tol=0.06):
        checks.append((label, got, want, abs(got - want) <= tol))

    # ---------------- 2. calibration check vs Table 9 -------------------------------
    print("== Calibration check against Table 9 (BO %, DE %, W L) ==")
    paper = {"No irrigation": (10.9, 74.2, 0.0), "Constant dose": (15.6, 36.9, 252.0), "RBC": (14.3, 65.5, 89.6), "PID Kp=4": (32.8, 12.4, 259.3)}
    pols = {"No irrigation": NoIrrigation(), "Constant dose": ConstantDose(), "RBC": RBC(), "PID Kp=4": PID(*KP4)}
    for k, p in pols.items():
        r = run(p, EVAL); print(f"{k:14s} paper {paper[k]}  here ({r['BO'].mean():.1f}, {r['DE'].mean():.1f}, {r['W'].mean():.1f})")
        if k == "No irrigation": check("No irrigation BO", r['BO'].mean(), 9.3); check("No irrigation DE", r['DE'].mean(), 77.5)
        if k == "Constant dose": check("Constant dose BO", r['BO'].mean(), 14.0); check("Constant dose DE", r['DE'].mean(), 41.8)
        if k == "RBC": check("RBC BO", r['BO'].mean(), 11.9); check("RBC DE", r['DE'].mean(), 72.1)

    # ---------------- 3. 180-configuration grid search on tuning scenarios ------------
    grid = list(itertools.product([0.5, 1, 2, 4, 8], [0, 0.01, 0.05], [0, 1, 4], [0.55, 0.60, 0.65]))
    rows = []
    for (kp, ki, kd, tg) in grid:
        r = run(PID(kp, ki, kd, tg), TUNE)
        rows.append(dict(Kp=kp, Ki=ki, Kd=kd, target=tg, BO=r["BO"].mean(), DE=r["DE"].mean(), W=r["W"].mean(), J6=r["J6"].mean(), J2=r["J2"].mean()))
    G = pd.DataFrame(rows); G["score_eq12"] = G.BO - 0.5 * G.DE; G.to_csv("pid_grid_tuning.csv", index=False)
    def winner(score):
        b = G.loc[score.idxmax()]; return (b.Kp, b.Ki, b.Kd, b.target)
    win = []
    for mu in (0, 0.5, 1, 2, 4, 8): win.append(dict(score=f"water-aware mu={mu}", gains=winner(G.score_eq12 - mu * G.W / 252)))
    win.append(dict(score="J (Eq. 3), lambda3=6", gains=winner(G.J6))); win.append(dict(score="J (Eq. 3), lambda3=2", gains=winner(G.J2)))
    Wn = pd.DataFrame(win); Wn.to_csv("pid_water_aware_winners.csv", index=False)
    print("\n== PID winners on tuning seeds 70000-70009 =="); print(Wn.to_string(index=False))
    for w in win: check(f"winner Kp ({w['score']})", float(w["gains"][0]), 8.0, 0.0)

    # ---------------- 4. provenance rows on the 30 evaluation scenarios ---------------
    e4, e8 = run(PID(*KP4), EVAL), run(PID(*KP8), EVAL)
    prov = pd.DataFrame([
        dict(Source="Table 9 (original code)", Gains="4,0,4,0.55", BO="32.8 ± 9.8", DE="12.4 ± 7.9", W="259.3 ± 82.1", J6="n/a", J2="n/a"),
        dict(Source="This script", Gains="4,0,4,0.55", BO=ms(e4["BO"]), DE=ms(e4["DE"]), W=ms(e4["W"]), J6=ms(e4["J6"]), J2=ms(e4["J2"])),
        dict(Source="This script", Gains="8,0,4,0.55", BO=ms(e8["BO"]), DE=ms(e8["DE"]), W=ms(e8["W"]), J6=ms(e8["J6"]), J2=ms(e8["J2"]))])
    prov.to_csv("pid_provenance.csv", index=False); print("\n== Provenance (nominal, 7 L, 30 evaluation scenarios) =="); print(prov.to_string(index=False))
    check("Kp4 BO", e4["BO"].mean(), 36.7); check("Kp4 DE", e4["DE"].mean(), 8.6); check("Kp4 W", e4["W"].mean(), 266.4)
    check("Kp8 BO", e8["BO"].mean(), 38.2); check("Kp8 DE", e8["DE"].mean(), 5.6); check("Kp8 W", e8["W"].mean(), 286.7)
    check("Kp4 J6", e4["J6"].mean(), -31.6); check("Kp8 J6", e8["J6"].mean(), -7.1)
    check("Kp4 J2", e4["J2"].mean(), 9.9); check("Kp8 J2", e8["J2"].mean(), 19.8)

    # ---------------- 5. 5 x 5 demand x capacity matrix -------------------------------
    caps = [4.0, 5.5, 7.0, 8.5, 10.0]; mat = []
    ctrls = {"no_irrigation": NoIrrigation(), "constant_dose": ConstantDose(), "rbc": RBC(), "pid_Kp4": PID(*KP4), "pid_Kp8": PID(*KP8)}
    for reg in REGIMES:
        sc = make_scenarios(range(90000, 90030), reg)
        for cap in caps:
            for name, p in ctrls.items():
                r = run(p, sc, cap)
                mat.append(dict(demand=reg, capacity_L=cap, controller=name, BO=r["BO"].mean(), BO_sd=r["BO"].std(ddof=1), DE=r["DE"].mean(), DE_sd=r["DE"].std(ddof=1), W=r["W"].mean()))
    M = pd.DataFrame(mat); M.to_csv("pid_matrix.csv", index=False)
    print("\n== Matrix, nominal regime, PID Kp=4 and Kp=8 =="); print(M[(M.demand == "nominal") & M.controller.isin(["pid_Kp4", "pid_Kp8"])].round(1).to_string(index=False))
    def cell(reg, cap, c, col): return float(M[(M.demand == reg) & (M.capacity_L == cap) & (M.controller == c)][col].iloc[0])
    for (reg, cap, c, bo, de) in [("nominal", 4.0, "pid_Kp4", 29.5, 29.1), ("nominal", 10.0, "pid_Kp4", 41.8, 3.2), ("nominal", 4.0, "pid_Kp8", 30.1, 26.6), ("nominal", 10.0, "pid_Kp8", 43.1, 2.8),
                                   ("heatwave", 7.0, "pid_Kp4", 15.5, 63.0), ("heatwave", 10.0, "pid_Kp4", 36.0, 11.5), ("heatwave", 7.0, "pid_Kp8", 27.2, 53.8), ("heatwave", 10.0, "pid_Kp8", 80.2, 10.7),
                                   ("beyond-heatwave", 7.0, "pid_Kp4", 2.7, 93.9), ("beyond-heatwave", 7.0, "pid_Kp8", 3.7, 92.8), ("beyond-heatwave", 10.0, "pid_Kp4", 19.4, 20.0)]:
        check(f"{reg} {cap:g}L {c} BO", cell(reg, cap, c, "BO"), bo); check(f"{reg} {cap:g}L {c} DE", cell(reg, cap, c, "DE"), de)
    rb = M[(M.demand == "nominal") & (M.controller == "rbc")]
    print(f"\nRBC nominal across 4-10 L: BO {rb.BO.min():.1f}-{rb.BO.max():.1f}, DE {rb.DE.min():.1f}-{rb.DE.max():.1f}")
    check("RBC BO min", rb.BO.min(), 11.0); check("RBC BO max", rb.BO.max(), 11.9); check("RBC DE min", rb.DE.min(), 71.9); check("RBC DE max", rb.DE.max(), 73.4)

    # ---------------- 6. observation-noise sensitivity --------------------------------
    nz = []
    for eps in (0.0, 0.02, 0.05, 0.10):
        for name, g in (("PID Kp4", KP4), ("PID Kp8", KP8)):
            r = run(PID(*g), EVAL, obs_noise=eps, noise_seed=1); nz.append(dict(controller=name, noise_sd=eps, BO=r["BO"].mean(), DE=r["DE"].mean(), W=r["W"].mean()))
    N = pd.DataFrame(nz); N.to_csv("noise_test.csv", index=False)
    print("\n== Observation-noise sensitivity (SD of noise on the stress estimate) =="); print(N.round({"BO": 1, "DE": 1, "W": 1}).to_string(index=False))

    # ---------------- 7. reproduction report ------------------------------------------
    bad = [c for c in checks if not c[3]]
    print(f"\n== Reproduction report: {len(checks) - len(bad)}/{len(checks)} values match the manuscript ==")
    for label, got, want, ok in bad: print(f"  MISMATCH {label}: got {got:.2f}, manuscript {want}")
    print(f"Done in {time.time() - t0:.0f} s. Files: pid_grid_tuning.csv, pid_water_aware_winners.csv, pid_provenance.csv, pid_matrix.csv, noise_test.csv")