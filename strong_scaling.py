import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

df = pd.read_csv("tiempos_optimizado.csv")
grillas = sorted(df["Grilla"].unique())
df["Celdas"] = df["Grilla"] ** 3

FLOPS_PER_CELL = 200

df["FLOPS"] = df["Celdas"] * FLOPS_PER_CELL
df["GFLOPS"] = df["FLOPS"] / (df["Tiempo(s)"] * 1e9)


if not os.path.exists("plots_hpc"):
    os.mkdir("plots_hpc")

colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    "#3182bd", "#31a354", "#756bb1", "#636363", "#9c9ede"
]

color_map = {g: colors[i % len(colors)] for i, g in enumerate(grillas)}


# 1. TIME vs THREADS (log-log)

plt.figure(figsize=(10,7))

for g in grillas:
    sub = df[df["Grilla"]==g].sort_values("Threads")
    plt.plot(sub["Threads"], sub["Tiempo(s)"], 
             marker="o", linewidth=3, markersize=10,
             color=color_map[g], label=f"Resolution={g}")

plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("Threads (log₂)")
plt.ylabel("Tp (seconds)")
plt.title("OMP Parallel Time Execution")
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("plots_hpc/tiempo_loglog.png")
plt.close()


# 2. SPEEDUP

plt.figure(figsize=(10,7))

for g in grillas:
    sub = df[df["Grilla"]==g].sort_values("Threads")
    T1 = sub[sub["Threads"]==1]["Tiempo(s)"].values[0]
    speedup = T1 / sub["Tiempo(s)"]
    plt.plot(sub["Threads"], speedup,
             marker="o", linewidth=3, markersize=10,
             color=color_map[g], label=f"Resolution={g}")

# Ideal line
threads_sorted = sorted(df["Threads"].unique())
plt.plot(threads_sorted, threads_sorted, '--', color="black", linewidth=2,
         label="Ideal Linear Speedup")

plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("Threads (log₂)")
plt.ylabel("Speedup")
plt.title("OMP Speedup")
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("plots_hpc/speedup_loglog.png")
plt.close()


# 3. EFFICIENCY (cap at 100%)

plt.figure(figsize=(10,7))

for g in grillas:
    sub = df[df["Grilla"]==g].sort_values("Threads")
    T1 = sub[sub["Threads"]==1]["Tiempo(s)"].values[0]
    speedup = T1 / sub["Tiempo(s)"]

    efficiency = (100 * speedup / sub["Threads"])

    plt.plot(sub["Threads"], efficiency,
             marker="o", linewidth=3, markersize=10,
             color=color_map[g], label=f"Resolution={g}")

plt.xscale("log", base=2)
plt.xlabel("Threads (log₂)")
plt.ylabel("Efficiency (%)")
plt.title("OMP Efficiency")

plt.axhline(100, linestyle="--", linewidth=2, color="black")  # Ideal 100%

plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("plots_hpc/eficiencia_log.png")
plt.close()


p_fijo = 80
df_p = df[df["Threads"] == p_fijo].sort_values("Grilla")

plt.figure(figsize=(10,7))
plt.plot(df_p["Grilla"], df_p["Tiempo(s)"], marker="o", linewidth=3)
plt.xlabel("Resolution (n)")
plt.ylabel("Tiempo (s)")
plt.title(f"Tiempo vs Tamaño del problema (p = {p_fijo})")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots_hpc/tiempo_vs_n_p16.png")
plt.close()


# 4. FLOPs vs Threads (Performance) — LOG–LOG

plt.figure(figsize=(10,7))

for g in grillas:
    sub = df[df["Grilla"] == g].sort_values("Threads")
    plt.plot(sub["Threads"], sub["GFLOPS"],
             marker="o", linewidth=3, markersize=10,
             color=color_map[g], label=f"Resolution={g}")

plt.xscale("log", base=2)
plt.yscale("log")

plt.xlabel("Threads (log₂)")
plt.ylabel("GFLOPs/s (log)")
plt.title("Rendimiento FLOPs vs Threads (OMP)")
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("plots_hpc/flops_vs_p_loglog.png")
plt.close()


print("Gráficas HPC generadas en la carpeta 'plots_hpc/'")
