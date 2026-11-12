#!/usr/bin/env bash
# Synchronize source and optional private notes; never delete destination files.
set -euo pipefail
TASK_LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MAGSET_SSH_HOST:?Set MAGSET_SSH_HOST to your configured SSH alias}"
: "${MAGSET_REMOTE_ROOT:?Set MAGSET_REMOTE_ROOT to an absolute project directory}"
if [[ ! "$MAGSET_SSH_HOST" =~ ^[A-Za-z0-9_.@-]+$ || "$MAGSET_SSH_HOST" == -* ]]; then
  echo 'Invalid SSH host/alias' >&2; exit 2
fi
if [[ ! "$MAGSET_REMOTE_ROOT" =~ ^/[A-Za-z0-9_./-]+$ || "$MAGSET_REMOTE_ROOT" == *'/../'* ]]; then
  echo 'Use an absolute remote root without spaces or traversal components' >&2; exit 2
fi
ssh -o BatchMode=yes -o ConnectTimeout=15 "$MAGSET_SSH_HOST" \
  "mkdir -p '$MAGSET_REMOTE_ROOT/repo' '$MAGSET_REMOTE_ROOT/data' '$MAGSET_REMOTE_ROOT/cache/tmp' '$MAGSET_REMOTE_ROOT/envs' '$MAGSET_REMOTE_ROOT/runs' '$MAGSET_REMOTE_ROOT/logs'"
rsync -az --exclude-from="$TASK_LOCAL_ROOT/.gitignore" --exclude='.git/' \
  "$TASK_LOCAL_ROOT/" "$MAGSET_SSH_HOST:$MAGSET_REMOTE_ROOT/repo/"
if [[ -d "$TASK_LOCAL_ROOT/memory" ]]; then
  rsync -az "$TASK_LOCAL_ROOT/memory/" "$MAGSET_SSH_HOST:$MAGSET_REMOTE_ROOT/repo/memory/"
fi
