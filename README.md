# routee-autoresearch

Autonomous research swarm for RouteE Powertrain model architectures. See `program.md` for the full experiment protocol.

## Running in Docker

The Docker container provides a sandboxed environment with internet access, Claude Code, and all Python dependencies pre-installed. Code running inside the container cannot make changes outside of it.

### Build

```bash
docker build --build-arg GIT_TOKEN=your_token_here -t autoresearch .
```

#### Building on Linux Server

If on an internal linux server, you might need to inject the custom certificates. Add the `CA_CERT` build arg:

```bash
docker build --build-arg CA_CERT="$(cat /usr/local/share/ca-certificates/nrel-ca-bundle.crt)" --build-arg GIT_TOKEN=your_token_here -t autoresearch .
```

This can be combined with `--platform linux/amd64` if building on a Mac for a remote Linux server.

### Run

```bash
docker run -it \
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


## Claude

Run `claude` and then log in.

Exit and then run again with `claude --dangerously-skip-permissions`

Initiate the session with:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```
