import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Crear carpeta images si no existe
if not os.path.exists("images"):
    os.makedirs("images")

# Cargar datos
df_strong = pd.read_csv("strong.csv")
df_weak   = pd.read_csv("weak.csv")

# Filtrar solo modo BASIC
df_strong = df_strong[df_strong["mode"] == "basic"]
df_weak   = df_weak[df_weak["mode"] == "basic"]

metodos = ["marching1_omp", "marching2_omp", "marching3_omp"]

def compute(df):
    df = df.sort_values("threads")
    p = df["threads"].values
    t = df["time"].values
    t1 = t[0]
    S = t1 / t
    E = S / p
    return p, t, S, E

def plot_method(method):

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"{method} - Modo BASIC (Memoria Compartida)", fontsize=16, fontweight="bold")

    # ----------- STRONG -----------
    d = df_strong[df_strong["exe"] == method]
    p, t, S, E = compute(d)

    # Tiempo Strong
    axs[0,0].plot(p, t, marker="o")
    axs[0,0].set_title("Strong: Tiempo vs hilos")
    axs[0,0].set_xlabel("Hilos")
    axs[0,0].set_ylabel("Tiempo (s)")
    axs[0,0].set_xticks(p)
    axs[0,0].grid(alpha=0.3)

    # Speedup Strong
    axs[0,1].plot(p, S, marker="o", label="Speedup")
    axs[0,1].plot(p, p, "--", label="Ideal")
    axs[0,1].set_title("Strong: Speedup vs hilos")
    axs[0,1].set_xlabel("Hilos")
    axs[0,1].set_ylabel("Speedup")
    axs[0,1].set_xticks(p)
    axs[0,1].legend()
    axs[0,1].grid(alpha=0.3)

    # Eficiencia Strong
    axs[0,2].plot(p, E, marker="o")
    axs[0,2].plot(p, [1]*len(p), "--", label="Ideal")
    axs[0,2].set_title("Strong: Eficiencia vs hilos")
    axs[0,2].set_xlabel("Hilos")
    axs[0,2].set_ylabel("Eficiencia")
    axs[0,2].set_xticks(p)
    axs[0,2].legend()
    axs[0,2].grid(alpha=0.3)

    # ----------- WEAK -----------
    d = df_weak[df_weak["exe"] == method]
    p, t, S, E = compute(d)

    # Tiempo Weak
    axs[1,0].plot(p, t, marker="o")
    axs[1,0].set_title("Weak: Tiempo vs hilos")
    axs[1,0].set_xlabel("Hilos")
    axs[1,0].set_ylabel("Tiempo (s)")
    axs[1,0].set_xticks(p)
    axs[1,0].grid(alpha=0.3)

    # Speedup Weak
    axs[1,1].plot(p, S, marker="o", label="Speedup")
    axs[1,1].plot(p, p, "--", label="Ideal")
    axs[1,1].set_title("Weak: Speedup vs hilos")
    axs[1,1].set_xlabel("Hilos")
    axs[1,1].set_ylabel("Speedup")
    axs[1,1].set_xticks(p)
    axs[1,1].legend()
    axs[1,1].grid(alpha=0.3)

    # Eficiencia Weak
    axs[1,2].plot(p, E, marker="o")
    axs[1,2].plot(p, [1]*len(p), "--", label="Ideal")
    axs[1,2].set_title("Weak: Eficiencia vs hilos")
    axs[1,2].set_xlabel("Hilos")
    axs[1,2].set_ylabel("Eficiencia")
    axs[1,2].set_xticks(p)
    axs[1,2].legend()
    axs[1,2].grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Guardar imagen
    filename = f"v1/{method}_basic_scaling.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Guardado: {filename}")

# Ejecutar para cada metodo
for m in metodos:
    plot_method(m)
