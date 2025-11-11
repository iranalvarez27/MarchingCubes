#!/bin/bash
#SBATCH --job-name=mc_weak
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=weak.out
#SBATCH --error=weak.err

cd $SLURM_SUBMIT_DIR

echo "Weak Scaling Benchmark"
echo "Nodo: $(hostname)"
echo "CPUs asignadas: $SLURM_CPUS_PER_TASK"
echo ""

THREADS="1 2 4 8 16 32"
STEPS="0.02 0.016 0.0127 0.01 0.0079 0.0063"

i=1

for exe in marching1_omp marching2_omp marching3_omp
do
    echo "============== Ejecutable: $exe =============="
    
    idx=1
    for t in $THREADS
    do
        step=$(echo $STEPS | awk -v n=$idx '{print $n}')
        export OMP_NUM_THREADS=$t
        
        echo -n "Hilos: $t, step: $step -> "
        ./$exe $step 2>/dev/null | grep "Tiempo" | awk '{print $5}'
        idx=$((idx+1))
    done
    echo ""
done
