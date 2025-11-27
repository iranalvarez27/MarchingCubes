import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_FILE = "../results_marching_cubes_octree_omp.csv"
TITLE = "OMP Parallel Time Execution"

df = pd.read_csv(CSV_FILE, skipinitialspace=True)
df = df.sort_values(by=["resolution", "threads"])
RESOLUTION_values = sorted(df["resolution"].unique())

# -------------------------------------------------------------------

# Theoric Function (s)
#C = 2.5e-5
#T_theo = lambda p, n: C * ((n / p) * np.log(p))

plt.figure(figsize=(10, 6))

# Use plasma colormap avoiding bright yellow values
colors = plt.cm.plasma(np.linspace(0, 0.75, len(RESOLUTION_values)))

for idx, res in enumerate(RESOLUTION_values):
  subset = df[df["resolution"] == res]
  th_vals = subset["threads"].astype(float).values
  tp_vals = subset["time_avg"].astype(float).values

  # Experimental curve
  plt.plot(th_vals, tp_vals,
           marker="o",
           color=colors[idx],
           linewidth=2,
           markersize=8,
           label=f"Experimental Resolution={res}")

  # Theoric Curve
  #th_teo = np.array(th_vals, dtype=float)
  #tp_teo = T_theo(th_teo, n)

  #plt.plot(th_teo, tp_teo,
  #        linestyle="--",
  #        linewidth=2,
  #        color=colors[idx],
  #        label=f"Theo Func Resolution={res}: ...")

plt.xlabel("ths (#threads)")
plt.ylabel("Tp (seconds)")
plt.xscale("log", base=2)
plt.yscale("log")
plt.title(TITLE, fontsize=14, fontweight="bold")
plt.grid(True, which="both", linestyle="--", alpha=0.4)

# Legend outside the plot
plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=1, fontsize=9)
plt.tight_layout(rect=[0, 0, 0.75, 1])  # Leave space on the right for legend
plt.savefig("TP_marching_cubes_octree_omp.png", dpi=300, bbox_inches="tight")
plt.show()
