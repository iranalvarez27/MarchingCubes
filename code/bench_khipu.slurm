#!/bin/bash
#SBATCH --job-name=mc_bench
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --output=bench.out
#SBATCH --error=bench.err

cd $SLURM_SUBMIT_DIR

echo "Nodo asignado:"
hostname
echo "CPUs asignadas: $SLURM_CPUS_PER_TASK"
echo ""
echo "CPU info:"
lscpu | grep 'CPU(s)'

THREADS="1 2 4 8 16 32 40 64 80"

for exe in marching1_omp marching2_omp marching3_omp
do
    echo "============== Ejecutable: $exe =============="
    for t in $THREADS
    do
        export OMP_NUM_THREADS=$t
        echo -n "Hilos: $t  -> "
        ./$exe 0.01 2>/dev/null | grep "Tiempo" | awk '{print $5}'
    done
    echo ""
done
