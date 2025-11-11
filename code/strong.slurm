#!/bin/bash
#SBATCH --job-name=mc_strong
#SBATCH --partition=standard
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=strong.out
#SBATCH --error=strong.err

module load gnu12
cd $SLURM_SUBMIT_DIR

EXES=("marching1_omp" "marching2_omp" "marching3_omp")
OUT=strong.csv

echo "exe,mode,threads,step,triangles,time" > $OUT

modes=("basic" "hard")
threads=(1 2 4 8 16 32)
step=0.01

for exe in "${EXES[@]}"; do
for mode in "${modes[@]}"; do
for t in "${threads[@]}"; do

    export OMP_NUM_THREADS=$t

    output=$(./$exe $step $mode)

    time=$(echo "$output" | grep TIME | cut -d'=' -f2)
    tris=$(echo "$output" | grep TRIANGLES | cut -d'=' -f2)

    echo "$exe,$mode,$t,$step,$tris,$time" >> $OUT

done
done
done
