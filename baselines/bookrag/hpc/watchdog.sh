#!/bin/bash
# Login-node watchdog — every INTERVAL, reap leftover zombie processes from
# timed-out sessions (bash/find/rsync/ssh) that exhaust the per-user shell quota.
#
# SCOPE / HAZARD: this reaps ONLY the allowlisted command types below (see
# sweep()), never "every non-ancestor process". A blanket sweep would also kill
# sbatch/squeue, python jobs, and a live sshd session driving an in-flight
# wcss/run_remote.sh poll. Even within the allowlist a bash/ssh/rsync belonging to
# an ACTIVE session can match, so keep INTERVAL coarse and prefer running this only
# when no run_remote poll is in flight.
INTERVAL="${WATCHDOG_INTERVAL:-300}"
SELF=$$
get_ancestors() {
    local pid=$SELF out="$pid" ppid
    while :; do
        ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
        [ -z "$ppid" ] && break; [ "$ppid" -le 1 ] && break
        out="$out $ppid"; pid=$ppid
    done
    echo "$out"
}
ANCESTORS=$(get_ancestors)
echo "$(date -Iseconds) WATCHDOG up: pid=$SELF ancestors='$ANCESTORS' interval=${INTERVAL}s"
sweep() {
    local KILLED=0 pid comm skip anc
    while read -r pid comm; do
        [ -z "$pid" ] && continue
        skip=0; for anc in $ANCESTORS; do [ "$pid" = "$anc" ] && skip=1 && break; done
        [ $skip -eq 1 ] && continue
        # ALLOWLIST: only reap the zombie command types this watchdog exists for.
        # Everything else (sbatch, squeue, python, sshd, systemd, ...) is left
        # alone so a running job or a live run_remote poll is never collateral.
        case "$comm" in
            bash|find|rsync|ssh|scp|sftp-server) ;;
            *) continue ;;
        esac
        kill -9 "$pid" 2>/dev/null && KILLED=$((KILLED+1))
    done < <(ps -u "$USER" -o pid=,comm= --no-headers)
    echo "$(date -Iseconds) sweep killed=$KILLED"
}
sweep    # immediate cleanup on start
while :; do sleep "$INTERVAL"; sweep; done
