#!/bin/bash
# Login-node watchdog — every INTERVAL, kill all $USER processes on the login
# node except self, ancestors, and systemd user-instance. Clears zombie
# bash/find/rsync/ssh from timed-out sessions that exhaust the shell quota.
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
        case "$comm" in systemd|"(sd-pam)"|sshd|watchdog.sh) continue ;; esac
        kill -9 "$pid" 2>/dev/null && KILLED=$((KILLED+1))
    done < <(ps -u "$USER" -o pid=,comm= --no-headers)
    echo "$(date -Iseconds) sweep killed=$KILLED"
}
sweep    # immediate cleanup on start
while :; do sleep "$INTERVAL"; sweep; done
