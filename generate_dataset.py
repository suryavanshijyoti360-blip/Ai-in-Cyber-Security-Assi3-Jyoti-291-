"""Generate the SYNTHETIC simple_cyber_activity.csv for Lab 3 (no real data).
100 rows, label 0 = Normal, 1 = Attack. Some Attack rows are deliberately
"stealthy" and some Normal rows "unusual", so the classes overlap."""
import os
import numpy as np, pandas as pd

SEED = int(os.environ.get("DS_SEED", 225))
rng = np.random.default_rng(SEED)

def normal(n):
    return pd.DataFrame({
        "failed_logins": rng.choice([0, 1, 2, 3], n, p=[0.5, 0.3, 0.15, 0.05]),
        "login_hour": np.clip(rng.normal(13, 4, n).round(), 0, 23).astype(int),
        "new_device": rng.choice([0, 1], n, p=[0.85, 0.15]),
        "data_transfer_mb": np.clip(rng.normal(70, 45, n).round(), 5, 300).astype(int),
        "label": 0})

def attack(n):
    return pd.DataFrame({
        "failed_logins": rng.integers(2, 12, n),
        "login_hour": rng.choice([0, 1, 2, 3, 4, 5, 22, 23, 14, 15, 10], n, p=[.1]*6 + [.08]*5),
        "new_device": rng.choice([0, 1], n, p=[0.3, 0.7]),
        "data_transfer_mb": rng.integers(60, 1000, n),
        "label": 1})

df = pd.concat([normal(70), attack(30)], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
df.to_csv("data/simple_cyber_activity.csv", index=False)
print(df.shape, df.label.value_counts().to_dict())
