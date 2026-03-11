# example-terminal-bench-env

RLVR environment for TerminalBench2-style tasks. The agent interacts with a sandboxed Linux container via bash to complete terminal-based challenges, scored by pytest verification.

### Overview
- **Environment ID**: `example-terminal-bench-env`
- **Type**: Multi-turn tool use (`SandboxEnv` subclass)
- **Reward**: Proportion of pytest tests passed + efficiency bonus for concise solutions

### Dataset
- **Source**: Bundled sample tasks in `sample_tasks/`
- **Format**: Each task has a `task.yaml` (description), `Dockerfile`, `test.sh`, and `tests/test_outputs.py`
- **Included tasks**: `nginx_setup` — configure nginx to serve a static page on port 8080

### Rubric

| Function | Weight | Description |
|:---|:---|:---|
| `test_pass_rate` | 1.0 | Proportion of pytest tests passed (from CTRF report) |
| `efficiency_bonus` | 0.2 | Bonus for solving in fewer turns (only when all tests pass) |

### Quickstart

Run an evaluation:

```bash
prime eval run example-terminal-bench-env -m gpt-4.1-mini
```

Run with custom settings:

```bash
prime eval run example-terminal-bench-env \
  -m gpt-4.1-mini \
  -n 1 -r 1 -t 4096 -T 0.7 \
  -a '{"max_turns": 50}'
```

### Environment Arguments

| Arg | Type | Default | Description |
|:---|:---|:---|:---|
| `max_turns` | int | `50` | Maximum bash interactions per rollout |
| `docker_image` | str | `"ubuntu:24.04"` | Default Docker image (overridden per-task) |

### Training

Push and train with Hosted Training:

```bash
prime env push example-terminal-bench-env
prime rl run @ configs/rl/terminal-bench.toml
```

See `configs/rl/terminal-bench.toml` for the GRPO training configuration.
