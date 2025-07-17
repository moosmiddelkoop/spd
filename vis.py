from typing import Any

import matplotlib.pyplot as plt
import numpy as np

CMAP = plt.get_cmap("plasma")


# Helper utilities -------------------------------------------------
def logit(p: np.ndarray) -> np.ndarray:
    eps = 1e-7
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def sample_logistic(size: int) -> np.ndarray:
    u = np.random.rand(size)
    return logit(u)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def plot_heatmap(
    x_samples: np.ndarray, y_samples: np.ndarray, title: str, ax: Any, bins: int = 100
) -> None:
    # 2‑D histogram normalised to a density
    hist, _, _ = np.histogram2d(
        x_samples, y_samples, bins=[bins, bins], range=[[0.0, 1.0], [0.0, 1.0]], density=True
    )
    # normalise rows to sum to 1
    hist = hist / hist.sum(axis=1, keepdims=True)

    im = ax.imshow(
        hist.T,
        origin="lower",
        extent=(0, 1, 0, 1),  # Changed to tuple
        aspect="auto",
        cmap=CMAP,
        vmin=0,
        vmax=float(np.percentile(hist, 99)),  # Explicitly convert to float
    )
    # Set bad color to the minimum color of viridis colormap
    im.cmap.set_bad(color=CMAP(0))
    ax.set_title(title)
    ax.set_xlabel("causal importance")
    ax.set_ylabel("gate value")


# Monte-Carlo sample size
N = 1_000_000

# Create figure with subplots (2 rows, 4 columns)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
plot_idx = 0

# # -------- 1. Uniform-min (unchanged) ------------------------------
# x = np.random.rand(N)
# y = x + (1 - x) * np.random.rand(N)
# plot_heatmap(x, y, "Uniform-min", axes[plot_idx])
# plot_idx += 1
x = np.random.rand(N)#  + 0.5
# ci = np.clip(x, 0, 1)
gate = np.clip(x + np.random.randn(N) / 10, 0, 1) # add stochastic noise so there's signal even in the saturated regions
plot_heatmap(x, gate, "ReLU(logit) + 0.5", axes[plot_idx])
plot_idx += 1

# -------- 2. Concrete distributions -------------------------------
temperatures = [1, 2/3, 0.3]
for tau in temperatures:
    x = np.random.rand(N)  # original input on [0,1]
    p_rescaled = x * 0.5 + 0.5  # → [0.5,1]

    L = sample_logistic(N)
    y = sigmoid((logit(p_rescaled) + L) / tau)

    plot_heatmap(x, y, f"Concrete τ={tau:.2f}", axes[plot_idx])  # still plot x∈[0,1]
    plot_idx += 1

# -------- 3. Hard-Concrete distributions --------------------------
l, r = -0.1, 1.1
for tau in temperatures:
    x = np.random.rand(N)
    # p_rescaled = x * 0.5 + 0.5

    L = sample_logistic(N)
    s = sigmoid((logit(p_rescaled) + L) / tau)
    v = s * (r - l) + l
    y = np.clip(v, 0.0, 1.0)

    plot_heatmap(x, y, f"Hard Concrete τ={tau:.2f}, bounds={l}, {r}", axes[plot_idx])
    plot_idx += 1

# -------- 4. Bernoulli -------------------------------------------
x = np.random.rand(N)
p_rescaled = x * 0.5 + 0.5
y = (np.random.rand(N) < p_rescaled).astype(float)

# tiny jitter so the two point-masses are visible in the heat-map
y_jitter = np.clip(y + 0.005 * np.random.randn(N), 0.0, 1.0)
plot_heatmap(x, y_jitter, "Bernoulli", axes[plot_idx])

# -------- Finish --------------------------------------------------
plt.tight_layout()
plt.show()
