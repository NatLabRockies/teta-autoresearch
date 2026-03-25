FROM ghcr.io/prefix-dev/pixi:latest

# Create a non-root user for sandboxing
RUN useradd --create-home --shell /bin/bash researcher

# Set working directory
WORKDIR /workspace

# Copy project files
COPY pyproject.toml pixi.lock ./
COPY prepare.py train.py program.md ./
COPY data/ data/

# Install dependencies via pixi
RUN pixi install

# Initialize git repo (needed for the experiment loop)
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN git init && git add -A && git commit -m "initial"

# Hand ownership to the non-root user
RUN chown -R researcher:researcher /workspace

USER researcher

# Default command: drop into a shell ready to run experiments
CMD ["bash"]
