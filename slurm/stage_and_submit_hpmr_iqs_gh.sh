#!/usr/bin/env bash
#
# Stage a GARNET snapshot for the HP-MR IQS guard re-check (audit item O3) and
# submit its cases to GH200. Run on a Merlin login node:
#
#   bash slurm/stage_and_submit_hpmr_iqs_gh.sh
#
# The cases are independent and all go to gh-daily (24 h limit; gh-general
# maps to one node shared with higher-priority partitions). The full-diffusion
# reference needs about 16 h for the 200 s transient at the 2026-08 rate and
# appends t, P/P0 to progress.txt every step, so a time-limit cut keeps its
# history; the IQS cases take about 1 h each. The dedicated stage directory
# never touches snapshots used by other queued GARNET jobs.
#
# Overrides:
#   GARNET_IQS_CASES       subset/order of cases
#                          (default: reference guarded-0.005 guarded-0.01 guarded-0.02 fixed)
#   GARNET_IQS_PARTITION_<CASE>, GARNET_IQS_TIME_<CASE>
#                          partition/time per case, with '-' and '.' in the case
#                          name written as '_', e.g. GARNET_IQS_TIME_GUARDED_0_02=08:00:00
#   GARNET_SLURM_DEPENDENCY  dependency applied to every case
#   SBATCH                 command used to submit (e.g. "echo" for a dry run)
#   GARNET_IQS_STAGE_ROOT, GARNET_IQS_WORK_ROOT  staging and results roots

set -euo pipefail

src_root="${GARNET_SRC_ROOT:-$HOME/GARNET}"
stage_root="${GARNET_IQS_STAGE_ROOT:-/data/scratch/shared/poncet_m/garnet-hpmr-iqs-stage}"
work_root="${GARNET_IQS_WORK_ROOT:-/data/scratch/shared/poncet_m/garnet-hpmr-iqs}"
read -r -a cases <<<"${GARNET_IQS_CASES:-reference guarded-0.005 guarded-0.01 guarded-0.02 fixed}"
sbatch_cmd="${SBATCH:-sbatch}"

mkdir -p "$stage_root" "$work_root/results"
rsync -a --delete \
    --exclude '.git' \
    --exclude '.agents' \
    --exclude '.codex' \
    --exclude '__pycache__' \
    --exclude '.pytest_cache' \
    --exclude 'results' \
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
export GARNET_IQS_RESULTS="$work_root/results"

for case_name in "${cases[@]}"; do
    key="${case_name^^}"; key="${key//[-.]/_}"
    partition_var="GARNET_IQS_PARTITION_$key"; time_var="GARNET_IQS_TIME_$key"
    if [[ "$case_name" == reference ]]; then
        partition="${!partition_var:-gh-daily}"; limit="${!time_var:-23:59:00}"
    else
        partition="${!partition_var:-gh-daily}"; limit="${!time_var:-06:00:00}"
    fi
    args=(--parsable --partition="$partition" --time="$limit" --job-name="garnet-iqs-$case_name"
          --export="ALL,GARNET_IQS_CASE=$case_name")
    if [[ -n "${GARNET_SLURM_DEPENDENCY:-}" ]]; then
        args+=(--dependency="$GARNET_SLURM_DEPENDENCY")
    fi
    id="$($sbatch_cmd "${args[@]}" "$GARNET_REPO/slurm/run_garnet_hpmr_iqs_gh.sh")"
    id="${id%%;*}"
    printf '%-14s job %s  (%s, %s)\n' "$case_name" "$id" "$partition" "$limit"
done
printf 'logs:    %s/garnet-hpmr-iqs-<job>.out\nresults: %s/results/<job>-<case>/\n' "$work_root" "$work_root"
