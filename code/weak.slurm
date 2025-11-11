#!/bin/bash
#SBATCH --job-name=mc_weak
#SBATCH --partition=standard
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=weak.out
#SBATCH --error=weak.err

module load gnu12
cd $SLURM_SUBMIT_DIR

EXES=("marching1_omp" "marching2_omp" "marching3_omp")
OUT=weak.csv

echo "exe,mode,threads,step,triangles,time" > $OUT

modes=("basic" "hard")
threads=(1 2 4 8 16 32)

for exe in "${EXES[@]}"; do
for mode in "${modes[@]}"; do
for t in "${threads[@]}"; do

    case $t in
      1) step=0.02 ;;
      2) step=0.016 ;;
      4) step=0.0127 ;;
      8) step=0.01 ;;
      16) step=0.0079 ;;
      32) step=0.0063 ;;
    esac

    export OMP_NUM_THREADS=$t

    output=$(./$exe $step $mode)

    time=$(echo "$output" | grep TIME | cut -d'=' -f2)
    tris=$(echo "$output" | grep TRIANGLES | cut -d'=' -f2)

    echo "$exe,$mode,$t,$step,$tris,$time" >> $OUT

done
done
done
