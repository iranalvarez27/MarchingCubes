import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D

CSV_FILE = "tiempos_grilla.csv"

df = pd.read_csv(CSV_FILE)

OUT_DIR = "weak"
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)

print("Datos cargados desde:", CSV_FILE)


pairs = [
    (1, 64),
    (2, 96),
    (4, 128),
    (6, 192),
    (8, 256),
    (12, 512),
    (16, 640),
    (32, 768),
    (48, 896),
    (64, 1024)
]

threads_ws = []
times_ws = []

for p, n in pairs:
    row = df[(df["Threads"] == p) & (df["Grilla"] == n)]
    if len(row) > 0:
        threads_ws.append(p)
        times_ws.append(row["Tiempo(s)"].values[0])
    else:
        print(f"Advertencia: No se encontró combinación (Threads={p}, Grilla={n})")


plt.figure(figsize=(10,6))
plt.plot(threads_ws, times_ws, marker='o', linewidth=3, color="#1f77b4")

plt.xscale("log", base=2)
plt.grid(True, which="both", linestyle="--", alpha=0.4)

plt.xlabel("Numero de Threads (p)")
plt.ylabel("Tiempo T(p) [s]")
plt.title("Weak Scaling — Tiempo paralelo T(p) vs Threads")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/weak_scaling_Tp_vs_p.png", dpi=300)
plt.close()

print("Figura generada:", f"{OUT_DIR}/weak_scaling_Tp_vs_p.png")


Sg = [(times_ws[0] / t) * p for p, t in zip(threads_ws, times_ws)]

plt.figure(figsize=(10,6))
plt.plot(threads_ws, Sg, marker='o', linewidth=3, color="purple", 
         label="Gustafson S_G(p)")

plt.plot(threads_ws, threads_ws, "--", color="black", label="Ideal S_G(p)=p")

plt.xscale("log", base=2)
plt.yscale("log")

plt.grid(True, which="both", linestyle="--", alpha=0.4)

plt.xlabel("Numero de Threads (p)")
plt.ylabel("S_G(p)")
plt.title("Escalabilidad de Gustafson — S_G(p) vs Threads")
plt.legend()

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/gustafson_scaling.png", dpi=300)
plt.close()

print("Figura generada:", f"{OUT_DIR}/gustafson_scaling.png")


grillas = sorted(df["Grilla"].unique())
threads = sorted(df["Threads"].unique())

Z = np.zeros((len(grillas), len(threads)))

for i, g in enumerate(grillas):
    for j, p in enumerate(threads):
        row = df[(df["Grilla"] == g) & (df["Threads"] == p)]
        Z[i, j] = row["Tiempo(s)"].values[0] if len(row) else np.nan

X, Y = np.meshgrid(threads, grillas)

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="k")

ax.set_xlabel("Número de Threads (p)")
ax.set_ylabel("Resolución (n)")
ax.set_zlabel("Tiempo T(n,p) [s]")
ax.set_title("3D Surface Plot — Weak Scaling Behavior")

fig.colorbar(surf, shrink=0.5, aspect=12)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/weak_scaling_surface.png", dpi=300)
plt.close()

print("Figura generada:", f"{OUT_DIR}/weak_scaling_surface.png")

Ew = [times_ws[0] / t for t in times_ws]

plt.figure(figsize=(10,6))
plt.plot(threads_ws, Ew, marker='o', linewidth=3, color="green",
         label="Eficiencia Weak Scaling")

plt.axhline(1.0, linestyle="--", color="black", label="Ideal = 1")

plt.xscale("log", base=2)
plt.ylim(0, 1.1)
plt.grid(True, which="both", linestyle="--", alpha=0.4)

plt.xlabel("Número de Threads (p)")
plt.ylabel("Eficiencia $E_{weak}(p)$")
plt.title("Eficiencia de Weak Scaling")
plt.legend()

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/weak_efficiency.png", dpi=300)
plt.close()

print("Figura generada:", f"{OUT_DIR}/weak_efficiency.png")

print("   Guardadas en la carpeta:", OUT_DIR)
