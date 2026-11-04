#!/usr/bin/env bash
# One-shot bootstrap: build image, create named volumes, start container,
# install SSH/gitconfig via docker cp (no host bind mount → ERM blind).
set -euo pipefail

CONTAINER="${CONTAINER:-hugarch-dev}"
IMAGE="${IMAGE:-hugarch-dev:latest}"
WORK_VOL="${WORK_VOL:-hugarch-work}"
CLAUDE_VOL="${CLAUDE_VOL:-hugarch-claude}"
HF_VOL="${HF_VOL:-hugarch-hf}"
REPO_URL="${REPO_URL:-git@github.com:shenh10/HuggingArch.git}"
BRANCH="${BRANCH:-main}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

say() { printf '\033[1;36m[bootstrap]\033[0m %s\n' "$*"; }

say "build image $IMAGE"
docker build -t "$IMAGE" "$SCRIPT_DIR"

say "ensure named volumes"
docker volume create "$WORK_VOL"   >/dev/null
docker volume create "$CLAUDE_VOL" >/dev/null
docker volume create "$HF_VOL"     >/dev/null

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  say "remove existing container"
  docker rm -f "$CONTAINER" >/dev/null
fi

say "start container"
docker run -d --name "$CONTAINER" \
  -v "$WORK_VOL:/workspace" \
  -v "$CLAUDE_VOL:/home/dev/.claude" \
  -v "$HF_VOL:/home/dev/.cache/huggingface" \
  --restart unless-stopped \
  "$IMAGE" sleep infinity >/dev/null

dex()  { docker exec       -u dev "$CONTAINER" "$@"; }
drex() { docker exec       -u root "$CONTAINER" "$@"; }
dex_w(){ docker exec -w /workspace/HuggingArch -u dev "$CONTAINER" "$@"; }

if [ -d "$HOME/.ssh" ]; then
  say "copy ~/.ssh into container (one-shot)"
  dex mkdir -p /home/dev/.ssh
  docker cp "$HOME/.ssh/." "$CONTAINER:/home/dev/.ssh/"
  drex chown -R dev:dev /home/dev/.ssh
  dex chmod 700 /home/dev/.ssh
  dex sh -c 'chmod 600 /home/dev/.ssh/id_* 2>/dev/null || true'
  dex sh -c 'ssh-keyscan -H github.com >> /home/dev/.ssh/known_hosts 2>/dev/null || true'
fi

if [ -f "$HOME/.gitconfig" ]; then
  say "copy ~/.gitconfig"
  docker cp "$HOME/.gitconfig" "$CONTAINER:/home/dev/.gitconfig"
  drex chown dev:dev /home/dev/.gitconfig
fi

say "clone $REPO_URL (branch=$BRANCH) if missing"
dex bash -c "
  set -e
  if [ ! -d /workspace/HuggingArch/.git ]; then
    cd /workspace
    git clone --branch '$BRANCH' '$REPO_URL' HuggingArch
  else
    echo 'repo already present, skipping clone'
  fi
"

say "uv sync --dev"
dex_w uv sync --dev || say "uv sync failed (non-fatal); fix inside container"

cat <<EOF

[bootstrap] done.

Enter the container:
  docker exec -it -u dev -w /workspace/HuggingArch $CONTAINER zsh

Inside, run Claude Code with bypass:
  claude --dangerously-skip-permissions
  # or use the alias
  claudey

Notes:
  - Code lives in named volume '$WORK_VOL' (Docker VM disk, not host FS).
  - First 'claude' run will prompt for /login; credentials persist in '$CLAUDE_VOL'.
  - To sync uncommitted host edits: commit + push from host first, then 'git pull' inside.
  - Tear down: docker rm -f $CONTAINER  (volumes survive; add 'docker volume rm ...' to wipe).
EOF
