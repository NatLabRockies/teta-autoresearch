# routee-autoresearch

Autonomous research swarm for RouteE Powertrain model architectures. See `program.md` for the full experiment protocol.

## Running in Docker

The Docker container provides a sandboxed environment with internet access, Claude Code, and all Python dependencies pre-installed. Code running inside the container cannot make changes outside of it.

### Build

```bash
docker build --build-arg GIT_TOKEN=your_token_here -t autoresearch .
```

### Run

```bash
docker run -it \
  -e ANTHROPIC_API_KEY=your_key_here \
  --security-opt no-new-privileges \
  --cap-drop ALL \
  --pids-limit 256 \
  --memory 8g \
  autoresearch
```

This drops you into a bash shell in `/workspace` with `claude` and `pixi` available.

### Sandbox properties

- **Internet access**: Enabled by default
- **Host isolation**: No host directories are mounted; the container cannot modify anything on the host
- **Dropped capabilities**: All Linux capabilities removed (no mounting, no raw sockets, etc.)
- **No privilege escalation**: Prevented via `no-new-privileges`
- **Resource limits**: 8 GB memory, 256 PIDs max
- **Non-root user**: Runs as `researcher`

### Extracting results

The container is ephemeral. To copy results out before stopping:

```bash
# Find the container ID
docker ps

# Copy results from the running container
docker cp <container_id>:/workspace/results.tsv .
```
