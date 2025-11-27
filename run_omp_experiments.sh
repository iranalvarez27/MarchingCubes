#!/bin/bash

EXEC=./marching_cubes_octree_omp
CSV=results_marching_cubes_octree_omp.csv
M=10  # number of repetitions for statistical average
THREADS=(1 2 4 8 16 32)  # number of OMP threads
RESOLUTIONS=(32 64 128 256 512)  # mesh resolutions (n = resolution^3 voxels)

# If CSV does not exist, create header
if [ ! -f "$CSV" ]; then
  echo "resolution,n_voxels,threads,vertices,faces,time_avg" >> $CSV
fi

echo ">> Running Marching Cubes Octree experiments with OMP..."

# ==========================
#   Main loop
# ==========================
for RES in "${RESOLUTIONS[@]}"; do
  # Calculate number of voxels: n = resolution^3
  N_VOXELS=$((RES * RES * RES))
  
  echo "=============================="
  echo ">>> Resolution = $RES (n_voxels = $N_VOXELS)"
  echo "=============================="

  for p in "${THREADS[@]}"; do
    echo ">> threads = $p"

    SUM_TIME=0
    VERTICES=0
    FACES=0

    for ((i=1; i<=M; i++)); do
      echo "  rep $i ..."

      # Execute program passing resolution and threads as arguments
      OUTPUT=$($EXEC $RES $p 2>&1)

      # Extract execution time
      TIME=$(echo "$OUTPUT" | grep "Execution time:" | awk '{print $3}')
      
      # Extract vertices and faces (only in first iteration)
      if [ $i -eq 1 ]; then
        VERTICES=$(echo "$OUTPUT" | grep "Generated mesh:" | awk '{print $3}')
        FACES=$(echo "$OUTPUT" | grep "Generated mesh:" | awk '{print $5}')
      fi

      echo "  Time: $TIME s"
      
      SUM_TIME=$(echo "$SUM_TIME + $TIME" | bc -l)
    done

    # Final average
    AVG_TIME=$(echo "$SUM_TIME / $M" | bc -l)

    # Save to CSV
    echo "$RES,$N_VOXELS,$p,$VERTICES,$FACES,$AVG_TIME" >> $CSV
    
    echo "  -> Average: $AVG_TIME s"
  done
done

echo ""
echo ">> Experiments finished. Data saved in $CSV."
echo ">> Summary:"
wc -l $CSV