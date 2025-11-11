#!/bin/bash

EXES=("marching1_omp" "marching2_omp" "marching3_omp")
OUT="local_strong.csv"

echo "exe,mode,threads,step,triangles,time,ram_MB" > $OUT

modes=("basic" "hard")
threads=(1 2 4 8 16)
step=0.01

for exe in "${EXES[@]}"; do
for mode in "${modes[@]}"; do
for t in "${threads[@]}"; do
    
    export OMP_NUM_THREADS=$t
    echo "Running $exe mode=$mode threads=$t"

    /usr/bin/env time -f "%M" -o mem.log ./$exe $step $mode > run.log

    time=$(grep TIME run.log | cut -d'=' -f2)
    tris=$(grep TRIANGLES run.log | cut -d'=' -f2)
    mem=$(cat mem.log)

    echo "$exe,$mode,$t,$step,$tris,$time,$mem" >> $OUT
done
done
done
