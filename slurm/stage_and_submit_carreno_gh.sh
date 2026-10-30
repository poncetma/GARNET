#!/usr/bin/env bash
#
# Stage a GARNET snapshot for the C5G7 pin-resolved mesh study of the
# Carreno et al. (2024) V&V report (audit item O10) and submit one GH200 job
# per method. Run on a Merlin login node:
#
#   bash slurm/stage_and_submit_carreno_gh.sh
#
# Overrides:
#   GARNET_VV_METHODS      subset/order of methods (default: SP1 SP3 SP5 SP7 SDP1 SDP2 SDP3)
#   GARNET_VV_CELLS, GARNET_VV_ROBIN_CELLS   passed to the job (see run_garnet_carreno_gh.sh)
#   GARNET_VV_TIME         time limit per job (default 12:00:00)
#   SBATCH                 command used to submit (e.g. "echo" for a dry run)
#   GARNET_VV_STAGE_ROOT, GARNET_VV_WORK_ROOT  staging and results roots

set -euo pipefail

src_root="${GARNET_SRC_ROOT:-$HOME/GARNET}"
stage_root="${GARNET_VV_STAGE_ROOT:-/data/scratch/shared/poncet_m/garnet-carreno-vv-stage}"
work_root="${GARNET_VV_WORK_ROOT:-/data/scratch/shared/poncet_m/garnet-carreno-vv}"
read -r -a methods <<<"${GARNET_VV_METHODS:-SP1 SP3 SP5 SP7 SDP1 SDP2 SDP3}"
sbatch_cmd="${SBATCH:-sbatch}"

mkdir -p "$stage_root" "$work_root/results"
rsync -a --delete \
    --exclude '.git' \
    --exclude '.agents' \
    --exclude '.codex' \
    --exclude '__pycache__' \
    --exclude '.pytest_cache' \
    --exclude 'results' \
    --exclude 'benchmark-results' \
    "$src_root/" "$stage_root/repo/"

# .git is not staged, so record exactly which source state the jobs ran.
{
    printf 'staged: %s by %s on %s\n' "$(date -Is)" "$(id -un)" "$(hostname)"
    printf 'source: %s\n' "$src_root"
    git -C "$src_root" log -1 --format='commit: %H %ad %s' --date=iso
    printf '\n# git status --short\n'
    git -C "$src_root" status --short
    printf '\n# git diff --stat HEAD\n'
    git -C "$src_root" diff --stat HEAD
} >"$stage_root/repo/PROVENANCE.txt" 2>&1 || true

export GARNET_REPO="$stage_root/repo"
export GARNET_PYTHON_BIN="${GARNET_PYTHON_BIN:-/data/scratch/shared/poncet_m/garnet-gh-py313-v1/bin/python}"
export GARNET_VV_RESULTS="$work_root/results"
limit="${GARNET_VV_TIME:-12:00:00}"

for method in "${methods[@]}"; do
    args=(--parsable --partition=gh-daily --time="$limit" --job-name="garnet-vv-$method"
          --export="ALL,GARNET_VV_METHOD=$method")
    id="$($sbatch_cmd "${args[@]}" "$GARNET_REPO/slurm/run_garnet_carreno_gh.sh")"
    id="${id%%;*}"
    printf '%-5s job %s  (gh-daily, %s)\n' "$method" "$id" "$limit"
done
printf 'logs:    %s/garnet-carreno-vv-<job>.out\nresults: %s/results/<job>-<method>/\n' "$work_root" "$work_root"
