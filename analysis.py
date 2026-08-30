!pip install lifelines -q

import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
import matplotlib.pyplot as plt
baseline = [(1000,1),(1200,1),(1200,1),(1200,1),(1000,1),(1200,1),(800,1),(1600,1),(1200,1),(800,1)]
suppress = [(1000,1),(1000,1),(1200,1),(1000,1),(1200,1),(1200,1),(1200,1),(1400,1),(1400,1),(1000,1)]
encourage = [(1800,1),(1800,1),(16000,0),(16000,0),(16000,0),(1400,1),(1800,1),(2600,1),(2200,1),(1000,1),
             (16000,0),(1600,1),(2600,1),(800,1),(1200,1),(1000,1),(16000,0),(16000,0),(2200,1),(1200,1),
             (2600,1),(16000,0),(1400,1),(800,1)]

def to_df(data, label):
    return pd.DataFrame({"step": [d[0] for d in data], "event": [d[1] for d in data], "condition": label})

df = pd.concat([to_df(baseline, "baseline"), to_df(suppress, "suppress"), to_df(encourage, "encourage")])

print("="*70)
print("KAPLAN-MEIER MEDIAN GROKKING TIME")
print("="*70)
fig, ax = plt.subplots(figsize=(8, 5.5))
kmfs = {}
for cond, color in zip(["baseline", "suppress", "encourage"], ["tab:blue", "tab:orange", "tab:green"]):
    sub = df[df.condition == cond]
    kmf = KaplanMeierFitter()
    kmf.fit(sub["step"], event_observed=sub["event"], label=cond)
    kmfs[cond] = kmf
    kmf.plot_survival_function(ax=ax, color=color)
    median = kmf.median_survival_time_
    print(f"{cond:10s}: median time-to-grok = {median} steps (n={len(sub)}, "
          f"{sub['event'].sum()} events observed, {(1-sub['event']).sum()} censored)")

ax.set_title("Kaplan-Meier: fraction not yet grokked")
ax.set_xlabel("step")
ax.set_ylabel("fraction not yet grokked")
plt.tight_layout()
plt.savefig("/kaggle/working/grok", dpi=140)

print("\n" + "="*70)
print("LOG-RANK TESTS")
print("="*70)

for cond in ["suppress", "encourage"]:
    sub_a = df[df.condition == "baseline"]
    sub_b = df[df.condition == cond]
    result = logrank_test(sub_a["step"], sub_b["step"],
                           event_observed_A=sub_a["event"], event_observed_B=sub_b["event"])
    print(f"baseline vs {cond:10s}: test statistic={result.test_statistic:.3f}, p={result.p_value:.4f}")

overall = multivariate_logrank_test(df["step"], df["condition"], df["event"])
print(f"\nOverall (all 3 groups): test statistic={overall.test_statistic:.3f}, p={overall.p_value:.4f}")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
