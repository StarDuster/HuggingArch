# HuggingArch dev container — Python 3.13 + uv + Node 20 + Claude Code.
# Runtime image lives in Dockerfile at repo root; this is dev-only.

FROM python:3.13-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/workspace/.venv

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates openssh-client zsh \
        less vim ripgrep jq \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

RUN npm install -g @anthropic-ai/claude-code

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/zsh $USERNAME \
    && mkdir -p /workspace /home/$USERNAME/.claude /home/$USERNAME/.cache/huggingface \
    && chown -R $USERNAME:$USERNAME /workspace /home/$USERNAME

USER $USERNAME
WORKDIR /workspace

RUN printf '%s\n' \
    'export PATH=$HOME/.local/bin:/workspace/.venv/bin:$PATH' \
    'export HF_HOME=$HOME/.cache/huggingface' \
    'alias ll="ls -lah"' \
    'alias claudey="claude --dangerously-skip-permissions"' \
    'cd /workspace/HuggingArch 2>/dev/null || true' \
    > ~/.zshrc

CMD ["sleep", "infinity"]
