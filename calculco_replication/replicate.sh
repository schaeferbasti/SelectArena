#!/usr/bin/env bash
# Rerun the Freiburg comparison's SelectArena jobs on calculco (OAR) and compare with Freiburg.
#
#   bash replicate.sh setup [--submit]   # prepares the run and prints (or submits) the oarsub command(s)
#   bash replicate.sh status             # results finished so far
#   bash replicate.sh eval               # when all subjobs are done: results.csv and the reports
#
# Run it on the front end. What runs is in benchmark.yaml (the Freiburg comparison's runs).
set -euo pipefail

# ---- Paths on calculco (edit if yours differ) ----------------------------------------------
SELECTARENA_REPO=$HOME/1_select_arena/SelectArena                          # SelectArena checkout (git clone)
SELECTARENA_PY=$HOME/venvs/selectarena/bin/python           # venv of scripts/oar/install_env.sh
DATA=/nfs/data/unprotected/lisic/bschafer                   # large storage seen by every node
CONTAINERS=                                                 # empty: download the containers from Hugging Face;
                                                            # or a local warehouse <dataset>/<uuid>
ENV_SCRIPT=/nfs/opt/env/env.sh
RUN=selectarena_run_hf                                      # this run's workspace under $DATA/fork_comparison
PREVIOUS=selectarena_run                                    # an earlier calculco run to compare with (optional)
# ---------------------------------------------------------------------------------------------

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG=$HERE/benchmark.yaml
WORK=$DATA/fork_comparison/$RUN              # this run's workspace, with its own OpenML cache
CLUSTER=$WORK/cluster.yaml
NAME=$(sed -n 's/^benchmark_name: *//p' "$CONFIG")
FREIBURG=$HERE/freiburg                      # results.csv files copied from Freiburg
cd "$HERE"

for path in "$SELECTARENA_REPO/scripts/run_benchmark.py" "$SELECTARENA_PY"; do
    [ -e "$path" ] || { echo "Not found: $path (edit the paths at the top of replicate.sh)"; exit 1; }
done

write_cluster_config() {
    mkdir -p "$WORK"
    cat > "$CLUSTER" <<EOF
# Written by replicate.sh; edit replicate.sh instead.
scheduler: oar
workspace: $WORK
python_path: $SELECTARENA_PY
caches:
  openml: $WORK/.openml-cache
  data_foundry: $WORK/.data_foundry_cache
  huggingface: null
  tabarena: null
data_foundry_warehouse: ${CONTAINERS:-null}
oar:
  cpu_resources: "/nodes=1/core={num_cpus}"
  cpu_properties: null
  queue: null
  besteffort: false
  env_script: $ENV_SCRIPT
  modules: []
  time_limit_overhead: 1
resources:
  num_cpus: 8
  memory_limit: 32
EOF
}

compare_with() {  # compare_with <freiburg results.csv> <label> <report dir>
    if [ -f "$1" ]; then
        "$SELECTARENA_PY" -P "$HERE/compare_runs.py" --fork "$1" --selectarena "$WORK/output/$NAME/results.csv" \
            --labels "$2,calculco" --out "$3"
    else
        echo "Skipped the comparison with $2: $1 not found (see README.md)."
    fi
}

case "${1:-}" in
    setup)
        write_cluster_config
        echo "SelectArena: $(git -C "$SELECTARENA_REPO" log -1 --format='%h %s')"
        "$SELECTARENA_PY" -P "$HERE/prepare_datasets.py" --config "$CONFIG" --cluster-config "$CLUSTER"
        "$SELECTARENA_PY" -P "$SELECTARENA_REPO/scripts/run_benchmark.py" setup \
            --config "$CONFIG" --cluster-config "$CLUSTER" "${@:2}"
        ;;
    status)
        echo "calculco: $(find "$WORK/output/$NAME/data" -name 'results.pkl*' 2>/dev/null | wc -l) of 456 results"
        echo "(oarstat -u $USER lists the subjobs still queued or running)"
        ;;
    eval)
        write_cluster_config
        "$SELECTARENA_PY" -P "$SELECTARENA_REPO/scripts/run_benchmark.py" eval \
            --config "$CONFIG" --cluster-config "$CLUSTER" --num-cpus 4
        compare_with "$FREIBURG/selectarena_results.csv" freiburg "$HERE/report_vs_freiburg_selectarena"
        compare_with "$FREIBURG/fork_results.csv" fork "$HERE/report_vs_fork"
        if [ -n "$PREVIOUS" ] && [ "$PREVIOUS" != "$RUN" ]; then
            compare_with "$DATA/fork_comparison/$PREVIOUS/output/$NAME/results.csv" "calculco_$PREVIOUS" \
                "$HERE/report_vs_$PREVIOUS"
        fi
        ;;
    *)
        sed -n '2,8p' "$0"
        exit 1
        ;;
esac
