#!/bin/bash
# Launch N parallel pathology-encoding workers, split across available GPUs.
#
# Subprocess-level parallelism deliberately, not Python multiprocessing:
# CUDA + fork()-based multiprocessing is a well-known source of hangs and
# crashes once any CUDA context has been initialized in the parent process.
# Independent OS-level subprocesses, each with its own clean interpreter and
# CUDA context, avoid this entirely and are also simpler to reason about --
# each worker is just a normal invocation of the same script already
# smoke-tested, run against 1/N of the cohort.
#
# Usage:
#   ./run_pathology_parallel.sh <n_workers> <n_gpus> [extra args passed through]
#
# Example, 8 workers split across 2 GPUs (4 workers/GPU), full TCGA cohort:
#   ./run_pathology_parallel.sh 8 2 \
#       --case-index /data/pduttapramanik/raresynth/data/manifests/case_index_tcga.csv \
#       --tcga-manifest /data/pduttapramanik/raresynth/data/manifests/tcga_manifest.json \
#       --raw-dir /data/pduttapramanik/raresynth/data/raw/tcga/slides \
#       --out /data/pduttapramanik/raresynth/data/embeddings/pathology_tcga_full
#
# Each worker writes to its own log file (pathology_worker_<i>.log in the
# current directory). Progress can be checked at any time with
# check_pathology_progress.py, without needing to inspect individual logs.
#
# Safe to re-run after a partial failure or interruption: completed cases
# are skipped automatically (each case's .npz is checked for existence
# before reprocessing), so restarting this script picks up only what is
# still missing across all shards.

set -e

N_WORKERS=$1
N_GPUS=$2
shift 2
EXTRA_ARGS="$@"

if [ -z "$N_WORKERS" ] || [ -z "$N_GPUS" ]; then
    echo "Usage: $0 <n_workers> <n_gpus> [extra args for run_pathology_encoder]"
    exit 1
fi

echo "Launching $N_WORKERS workers across $N_GPUS GPU(s)"
echo "Extra args: $EXTRA_ARGS"
echo ""

# Cap each worker's internal thread usage (OpenMP/MKL/OpenBLAS/NumExpr all
# default to spawning one thread PER SYSTEM CORE, per PROCESS, unless told
# otherwise -- with N_WORKERS processes all doing this simultaneously, they
# do not get N_WORKERS x the throughput, they mostly contend with each
# other for the same cores. Confirmed on this exact server: a single
# worker spawned 429 threads and system load hit 755 on a 344-core
# machine, with ZERO cases completed despite tens of CPU-minutes consumed.
# Splitting the core budget evenly across workers (with a sane floor) lets
# the 8 PROCESSES provide the parallelism, which is what --n-shards is
# for, rather than each process ALSO fighting internally for every core.
THREADS_PER_WORKER=$(( $(nproc) / N_WORKERS ))
if [ "$THREADS_PER_WORKER" -lt 1 ]; then THREADS_PER_WORKER=1; fi
if [ "$THREADS_PER_WORKER" -gt 8 ]; then THREADS_PER_WORKER=8; fi  # BLAS-style
    # ops see negligible benefit and rising overhead well before 8 threads
    # for typical per-tile operation sizes -- capping here leaves the rest
    # of the per-worker core share idle rather than actively harmful, which
    # is the safe direction to err on this shared server
echo "threads per worker: $THREADS_PER_WORKER (of $(nproc) total cores / $N_WORKERS workers)"
echo ""

PIDS=()
for ((i=0; i<N_WORKERS; i++)); do
    GPU=$((i % N_GPUS))
    LOG="pathology_worker_${i}.log"
    echo "  worker $i -> GPU $GPU -> $LOG"
    CUDA_VISIBLE_DEVICES=$GPU \
    OMP_NUM_THREADS=$THREADS_PER_WORKER \
    MKL_NUM_THREADS=$THREADS_PER_WORKER \
    OPENBLAS_NUM_THREADS=$THREADS_PER_WORKER \
    NUMEXPR_NUM_THREADS=$THREADS_PER_WORKER \
    nohup python -u -m raresynth.encoders.run_pathology_encoder \
        --shard-index $i --n-shards $N_WORKERS \
        $EXTRA_ARGS \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All $N_WORKERS workers launched. PIDs: ${PIDS[@]}"
echo "Check progress with: python check_pathology_progress.py <out_dir>"
echo "Or tail a specific worker's log: tail -f pathology_worker_0.log"
echo ""
echo "This script's own job is done (workers are detached background"
echo "processes) -- safe to let this shell exit; the workers keep running"
echo "under nohup regardless."
