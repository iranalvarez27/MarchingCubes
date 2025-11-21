import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuración de estilo mejorada
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'

# CARGAR DATOS
print("="*70)
print("ANÁLISIS DE RENDIMIENTO - MARCHING CUBES PARALELO OPTIMIZADO")
print("="*70)
print("\nCargando datos de CSV...")

try:
    df_threads = pd.read_csv('resultados_threads.csv')
    df_escala = pd.read_csv('resultados_escalabilidad.csv')
    print("✓ Datos cargados correctamente\n")
except FileNotFoundError as e:
    print(f"ERROR: No se encontró el archivo {e.filename}")
    print("Asegúrate de ejecutar primero el programa C++ para generar los CSVs")
    exit(1)

print("Vista previa - Datos de Threads:")
print(df_threads.head())
print("\nVista previa - Datos de Escalabilidad:")
print(df_escala.head())
print("\n" + "="*70 + "\n")


# GRÁFICA 1: SPEEDUP vs NÚMERO DE HILOS

fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# Speedup real
ax.plot(df_threads['Threads'], df_threads['Speedup'], 
        marker='o', linewidth=3, markersize=12, 
        label='Speedup Real', color='#2E86AB', 
        markeredgecolor='white', markeredgewidth=2)

# Speedup ideal (lineal)
speedup_ideal = df_threads['Threads'].values
ax.plot(df_threads['Threads'], speedup_ideal, 
        '--', linewidth=2.5, label='Speedup Ideal (Lineal)', 
        color='#A23B72', alpha=0.8)

# Línea de speedup sub-lineal típico (Ley de Amdahl aproximada)
if len(df_threads) > 0:
    f_parallel = 0.95  # 95% paralelizable
    amdahl = df_threads['Threads'].apply(lambda p: 1 / ((1 - f_parallel) + f_parallel/p))
    ax.plot(df_threads['Threads'], amdahl, 
            '-.', linewidth=2, label='Ley de Amdahl (95% paralelo)', 
            color='#F18F01', alpha=0.7)

ax.set_xlabel('Número de Hilos (p)', fontsize=14, fontweight='bold')
ax.set_ylabel('Speedup = T(1) / T(p)', fontsize=14, fontweight='bold')
ax.set_title('Speedup vs Número de Hilos\nMarching Cubes Optimizado con OpenMP', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.4, linestyle='--')
ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
ax.set_xlim(left=0.5)
ax.set_ylim(bottom=0)

# Anotaciones en los puntos
for i, row in df_threads.iterrows():
    ax.annotate(f'{row["Speedup"]:.2f}x', 
                (row['Threads'], row['Speedup']),
                textcoords="offset points", xytext=(0,12), 
                ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('grafica_1_speedup.png', dpi=300, bbox_inches='tight')
print("✓ Gráfica 1 guardada: grafica_1_speedup.png")

# GRÁFICA 2: TIEMPO DE EJECUCIÓN vs HILOS

fig, ax = plt.subplots(1, 1, figsize=(10, 7))

ax.plot(df_threads['Threads'], df_threads['Tiempo(s)'], 
        marker='s', linewidth=3, markersize=12, 
        color='#06A77D', label='Tiempo Real',
        markeredgecolor='white', markeredgewidth=2)

# Tiempo ideal (T(1)/p)
if len(df_threads) > 0:
    t1 = df_threads[df_threads['Threads'] == 1]['Tiempo(s)'].values
    if len(t1) > 0:
        tiempo_ideal = df_threads['Threads'].apply(lambda p: t1[0] / p)
        ax.plot(df_threads['Threads'], tiempo_ideal, 
                '--', linewidth=2.5, label='Tiempo Ideal', 
                color='#A23B72', alpha=0.8)

ax.set_xlabel('Número de Hilos (p)', fontsize=14, fontweight='bold')
ax.set_ylabel('Tiempo de Ejecución (segundos)', fontsize=14, fontweight='bold')
ax.set_title('Tiempo de Ejecución vs Número de Hilos\nReducción del Tiempo con Paralelización', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.4, linestyle='--')
ax.legend(fontsize=11, framealpha=0.95)
ax.set_xlim(left=0.5)

# Anotaciones
for i, row in df_threads.iterrows():
    ax.annotate(f'{row["Tiempo(s)"]:.3f}s', 
                (row['Threads'], row['Tiempo(s)']),
                textcoords="offset points", xytext=(0,10), 
                ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('grafica_2_tiempo.png', dpi=300, bbox_inches='tight')
print("✓ Gráfica 2 guardada: grafica_2_tiempo.png")

# GRÁFICA 3: EFICIENCIA vs HILOS

fig, ax = plt.subplots(1, 1, figsize=(10, 7))

ax.plot(df_threads['Threads'], df_threads['Eficiencia(%)'], 
        marker='D', linewidth=3, markersize=12, 
        color='#F18F01', label='Eficiencia Real',
        markeredgecolor='white', markeredgewidth=2)

ax.axhline(y=100, color='#A23B72', linestyle='--', 
           linewidth=2.5, label='Eficiencia Ideal (100%)', alpha=0.8)

# Zona de buena eficiencia (>80%)
ax.axhspan(80, 110, alpha=0.1, color='green', label='Zona de alta eficiencia')

ax.set_xlabel('Número de Hilos (p)', fontsize=14, fontweight='bold')
ax.set_ylabel('Eficiencia (%)', fontsize=14, fontweight='bold')
ax.set_title('Eficiencia vs Número de Hilos\nUtilización de Recursos Paralelos', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.4, linestyle='--')
ax.legend(fontsize=11, framealpha=0.95)
ax.set_ylim(0, 110)
ax.set_xlim(left=0.5)

# Anotaciones
for i, row in df_threads.iterrows():
    color = 'green' if row['Eficiencia(%)'] >= 80 else 'orange' if row['Eficiencia(%)'] >= 60 else 'red'
    ax.annotate(f'{row["Eficiencia(%)"]:.1f}%', 
                (row['Threads'], row['Eficiencia(%)']),
                textcoords="offset points", xytext=(0,12), 
                ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.5))

plt.tight_layout()
plt.savefig('grafica_3_eficiencia.png', dpi=300, bbox_inches='tight')
print("Gráfica 3 guardada: grafica_3_eficiencia.png")

# GRÁFICA 4: ESCALABILIDAD - TIEMPO vs TAMAÑO

fig, ax = plt.subplots(1, 1, figsize=(11, 7))

resoluciones = df_escala['Resolucion'].values
tiempos = df_escala['Tiempo(s)'].values
grillas = df_escala['Grilla'].values

# Colores degradados
colores = plt.cm.viridis(np.linspace(0.2, 0.9, len(resoluciones)))
bars = ax.bar(range(len(resoluciones)), tiempos, 
              color=colores, alpha=0.85, 
              edgecolor='black', linewidth=1.5)

ax.set_xticks(range(len(resoluciones)))
ax.set_xticklabels(resoluciones, fontsize=12, fontweight='bold')
ax.set_xlabel('Resolución de Malla (celdas³)', fontsize=14, fontweight='bold')
ax.set_ylabel('Tiempo de Ejecución (segundos)', fontsize=14, fontweight='bold')
ax.set_title('Escalabilidad: Tiempo vs Tamaño del Problema\nMarching Cubes Optimizado (p=8 hilos)', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.4, axis='y', linestyle='--')

# Anotaciones encima de barras
for i, (bar, tiempo) in enumerate(zip(bars, tiempos)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{tiempo:.2f}s',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Número de celdas procesadas
    if 'Celdas' in df_escala.columns:
        celdas = df_escala.iloc[i]['Celdas']
        ax.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                f'{int(celdas):,}\nceldas',
                ha='center', va='center', fontsize=9, 
                style='italic', color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('grafica_4_escalabilidad.png', dpi=300, bbox_inches='tight')
print("Gráfica 4 guardada: grafica_4_escalabilidad.png")


# GRÁFICA 5: COMPLEJIDAD COMPUTACIONAL (Log-Log)

if 'Celdas' in df_escala.columns and len(df_escala) > 1:
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    celdas = df_escala['Celdas'].values
    tiempos_escala = df_escala['Tiempo(s)'].values
    
    ax.loglog(celdas, tiempos_escala, 
              marker='o', linewidth=3, markersize=12, 
              color='#2E86AB', label='Tiempo Medido',
              markeredgecolor='white', markeredgewidth=2)
    
    # Ajuste lineal en log-log (complejidad)
    if len(celdas) > 1:
        log_celdas = np.log10(celdas)
        log_tiempo = np.log10(tiempos_escala)
        coef = np.polyfit(log_celdas, log_tiempo, 1)
        fit_tiempo = 10**(coef[0] * log_celdas + coef[1])
        
        ax.loglog(celdas, fit_tiempo, '--', linewidth=2.5, 
                  color='#A23B72', alpha=0.8,
                  label=f'Ajuste: O(n^{coef[0]:.2f})')
    
    # Referencias de complejidad
    ref_celdas = np.array([celdas[0], celdas[-1]])
    ref_lineal = tiempos_escala[0] * (ref_celdas / celdas[0])
    ax.loglog(ref_celdas, ref_lineal, ':', linewidth=2, 
              color='green', alpha=0.6, label='O(n) lineal')
    
    ax.set_xlabel('Número de Celdas Procesadas', fontsize=14, fontweight='bold')
    ax.set_ylabel('Tiempo de Ejecución (segundos)', fontsize=14, fontweight='bold')
    ax.set_title('Análisis de Complejidad Computacional\nEscala Log-Log', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.4, which='both', linestyle='--')
    ax.legend(fontsize=11, framealpha=0.95)
    
    # Anotaciones
    for i, (c, t) in enumerate(zip(celdas, tiempos_escala)):
        ax.annotate(f'{t:.2f}s', (c, t),
                    textcoords="offset points", xytext=(10,5), 
                    ha='left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('grafica_5_complejidad.png', dpi=300, bbox_inches='tight')
    print("Gráfica 5 guardada: grafica_5_complejidad.png")

# GRÁFICA 6: PANEL COMPLETO (2x2)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Subplot 1: Speedup
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(df_threads['Threads'], df_threads['Speedup'], 
         marker='o', linewidth=2.5, markersize=10, 
         color='#2E86AB', label='Real', markeredgecolor='white', markeredgewidth=1.5)
ax1.plot(df_threads['Threads'], speedup_ideal, 
         '--', linewidth=2, color='#A23B72', alpha=0.7, label='Ideal')
ax1.set_xlabel('Threads (p)', fontweight='bold', fontsize=12)
ax1.set_ylabel('Speedup', fontweight='bold', fontsize=12)
ax1.set_title('(a) Speedup vs Hilos', fontweight='bold', fontsize=13)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_xlim(left=0.5)

# Subplot 2: Eficiencia
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(df_threads['Threads'], df_threads['Eficiencia(%)'], 
         marker='D', linewidth=2.5, markersize=10, color='#F18F01',
         markeredgecolor='white', markeredgewidth=1.5)
ax2.axhline(y=100, color='#A23B72', linestyle='--', linewidth=2, alpha=0.7)
ax2.axhspan(80, 110, alpha=0.1, color='green')
ax2.set_xlabel('Threads (p)', fontweight='bold', fontsize=12)
ax2.set_ylabel('Eficiencia (%)', fontweight='bold', fontsize=12)
ax2.set_title('(b) Eficiencia', fontweight='bold', fontsize=13)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 110)
ax2.set_xlim(left=0.5)

# Subplot 3: Tiempo
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(df_threads['Threads'], df_threads['Tiempo(s)'], 
         marker='s', linewidth=2.5, markersize=10, color='#06A77D',
         markeredgecolor='white', markeredgewidth=1.5)
ax3.set_xlabel('Threads (p)', fontweight='bold', fontsize=12)
ax3.set_ylabel('Tiempo (s)', fontweight='bold', fontsize=12)
ax3.set_title('(c) Tiempo de Ejecución', fontweight='bold', fontsize=13)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(left=0.5)

# Subplot 4: Escalabilidad
ax4 = fig.add_subplot(gs[1, 1])
colores_panel = plt.cm.viridis(np.linspace(0.2, 0.9, len(resoluciones)))
ax4.bar(range(len(resoluciones)), tiempos, 
        color=colores_panel, alpha=0.85, edgecolor='black', linewidth=1.5)
ax4.set_xticks(range(len(resoluciones)))
ax4.set_xticklabels(resoluciones, fontsize=10)
ax4.set_xlabel('Resolución', fontweight='bold', fontsize=12)
ax4.set_ylabel('Tiempo (s)', fontweight='bold', fontsize=12)
ax4.set_title('(d) Escalabilidad (p=8)', fontweight='bold', fontsize=13)
ax4.grid(True, alpha=0.3, axis='y')

plt.suptitle('Análisis Completo: Marching Cubes Paralelo Optimizado con OpenMP', 
             fontsize=17, fontweight='bold', y=0.995)
plt.savefig('grafica_6_panel_completo.png', dpi=300, bbox_inches='tight')
print("Gráfica 6 guardada: grafica_6_panel_completo.png")

# ANÁLISIS ESTADÍSTICO

print("\n" + "="*70)
print("ANÁLISIS ESTADÍSTICO DETALLADO")
print("="*70)

# Análisis de Speedup
if len(df_threads) > 0:
    max_speedup = df_threads['Speedup'].max()
    max_speedup_threads = df_threads[df_threads['Speedup'] == max_speedup]['Threads'].values[0]
    speedup_8 = df_threads[df_threads['Threads'] == 8]['Speedup'].values
    speedup_8 = speedup_8[0] if len(speedup_8) > 0 else 0
    
    print("\n SPEEDUP:")
    print(f"  Speedup máximo: {max_speedup:.2f}x (con {max_speedup_threads} hilos)")
    if speedup_8 > 0:
        print(f"  Speedup con 8 hilos: {speedup_8:.2f}x")
        print(f"  Eficiencia paralela: {(speedup_8/8)*100:.1f}%")

# Análisis de Eficiencia
if len(df_threads) > 0:
    eficiencia_promedio = df_threads['Eficiencia(%)'].mean()
    mejor_eficiencia = df_threads['Eficiencia(%)'].max()
    threads_mejor_eficiencia = df_threads[df_threads['Eficiencia(%)'] == mejor_eficiencia]['Threads'].values[0]
    
    print("\n EFICIENCIA:")
    print(f"   Eficiencia promedio: {eficiencia_promedio:.1f}%")
    print(f"   Mejor eficiencia: {mejor_eficiencia:.1f}% ({threads_mejor_eficiencia} hilos)")
    
    # Clasificar eficiencia
    eficiencias_altas = df_threads[df_threads['Eficiencia(%)'] >= 80]
    if len(eficiencias_altas) > 0:
        print(f"   Configuraciones con eficiencia >80%: {len(eficiencias_altas)}")

# Análisis de Escalabilidad
if len(df_escala) > 1 and 'Celdas' in df_escala.columns:
    factor_escala = df_escala['Grilla'].iloc[-1] / df_escala['Grilla'].iloc[0]
    factor_tiempo = df_escala['Tiempo(s)'].iloc[-1] / df_escala['Tiempo(s)'].iloc[0]
    factor_celdas = df_escala['Celdas'].iloc[-1] / df_escala['Celdas'].iloc[0]
    
    print("\n ESCALABILIDAD:")
    print(f"   Factor de escala en grilla: {factor_escala:.1f}x")
    print(f"   Factor de aumento en celdas: {factor_celdas:.1f}x")
    print(f"   Factor de aumento en tiempo: {factor_tiempo:.1f}x")
    print(f"   Complejidad aparente: O(n^{np.log(factor_tiempo)/np.log(factor_celdas):.2f})")

# TABLAS 

print("\n" + "="*70)
print("TABLA 1: ANÁLISIS DE PARALELIZACIÓN")
print("="*70)
print(df_threads.to_string(index=False))

print("\n" + "="*70)
print("TABLA 2: ANÁLISIS DE ESCALABILIDAD")
print("="*70)
print(df_escala.to_string(index=False))


print("\n" + "="*70)
print("CONCLUSIONES AUTOMÁTICAS")
print("="*70)

if len(df_threads) > 0:
    eficiencia_8 = df_threads[df_threads['Threads'] == 8]['Eficiencia(%)'].values
    if len(eficiencia_8) > 0 and eficiencia_8[0] >= 80:
        print("Excelente escalabilidad: Eficiencia >80% con 8 hilos")
    elif len(eficiencia_8) > 0 and eficiencia_8[0] >= 60:
        print("Escalabilidad moderada: Eficiencia entre 60-80%")
    else:
        print("Escalabilidad limitada: Revisar balance de carga y overhead")

print("\n" + "="*70)
print("ARCHIVOS GENERADOS")
print("="*70)
archivos = [
    "grafica_1_speedup.png",
    "grafica_2_tiempo.png",
    "grafica_3_eficiencia.png",
    "grafica_4_escalabilidad.png",
    "grafica_5_complejidad.png",
    "grafica_6_panel_completo.png"
]

for i, archivo in enumerate(archivos, 1):
    print(f"  {i}. {archivo}")

print("="*70)
print("ANÁLISIS COMPLETADO ✓")
print("="*70)