FROM ghcr.io/prefix-dev/pixi:latest

# Install git and Node.js (needed for clone, experiment loop, and Claude Code)
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install Claude Code
RUN npm install -g @anthropic-ai/claude-code

# Create a non-root user for sandboxing
RUN useradd --create-home --shell /bin/bash researcher

# Clone the repo (requires a GitHub personal access token for github.nrel.gov)
ARG GIT_TOKEN
RUN git clone https://${GIT_TOKEN}@github.nrel.gov/RouteE/routee-autoresearch.git /workspace

# Set working directory
WORKDIR /workspace

RUN pixi install

# Hand ownership to the non-root user
RUN chown -R researcher:researcher /workspace

USER researcher

# Default command: drop into a shell ready to run experiments
CMD ["bash"]
