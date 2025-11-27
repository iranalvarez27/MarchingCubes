import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_FILE = "../results_marching_cubes_octree_omp.csv"
TITLE = "OMP Efficiency"

df = pd.read_csv(CSV_FILE, skipinitialspace=True)
df = df.sort_values(by=["resolution", "threads"])
RESOLUTION_values = sorted(df["resolution"].unique())

# -------------------------------------------------------------------

plt.figure(figsize=(10, 6))

# Use plasma colormap avoiding bright yellow values
colors = plt.cm.plasma(np.linspace(0, 0.75, len(RESOLUTION_values)))

for idx, res in enumerate(RESOLUTION_values):
  subset = df[df["resolution"] == res]
  th_vals = subset["threads"].astype(float).values
  tp_vals = subset["time_avg"].astype(float).values

  # Calculate efficiency: E(p) = S(p) / p = T(1) / (p * T(p))
  T1 = tp_vals[0]  # Time with 1 thread
  speedup = T1 / tp_vals
  efficiency = (speedup / th_vals) * 100  # Efficiency as percentage

  # Experimental efficiency curve
  plt.plot(th_vals, efficiency,
           marker="o",
           color=colors[idx],
           linewidth=2,
           markersize=8,
           label=f"Experimental Resolution={res}")

# Ideal efficiency (100%)
plt.axhline(y=100, linestyle="--", linewidth=2, color="black", label="Ideal Efficiency (100%)")

plt.xlabel("ths (#threads)")
plt.ylabel("Efficiency (%)")
plt.xscale("log", base=2)
plt.title(TITLE, fontsize=14, fontweight="bold")
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.ylim([0, 110])

# Legend outside the plot
plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=1, fontsize=9)
plt.tight_layout(rect=[0, 0, 0.75, 1])  # Leave space on the right for legend
plt.savefig("EFFICIENCY_marching_cubes_octree_omp.png", dpi=300, bbox_inches="tight")
plt.show()
