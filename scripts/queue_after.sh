#!/usr/bin/env bash
# Wait for a marker line in a log file, then run a command. Usage: queue_after.sh <logfile> <marker> <cmd...>
set -u
LOG=$1; MARK=$2; shift 2
until grep -q "$MARK" "$LOG" 2>/dev/null; do sleep 20; done
exec "$@"
