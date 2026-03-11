import base64
import json
import logging
from pathlib import Path
from typing import Any

import yaml
import verifiers as vf
from datasets import Dataset
from verifiers.envs.sandbox_env import SandboxEnv

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a terminal agent. You have access to a bash shell inside a Linux container. "
    "Your goal is to complete the task described below by executing shell commands. "
    "Work step by step. When you believe the task is complete, stop issuing commands."
)

DEFAULT_MAX_TURNS = 50
DEFAULT_TEST_TIMEOUT = 120


def parse_ctrf_results(ctrf_json_str: str) -> tuple[int, int]:
    """Parse a CTRF JSON test report and return (passed, total)."""
    try:
        report = json.loads(ctrf_json_str)
        summary = report["results"]["summary"]
        passed = summary.get("passed", 0)
        total = summary.get("tests", 0)
        return passed, total
    except (json.JSONDecodeError, KeyError, TypeError):
        return 0, 0


def load_sample_tasks() -> Dataset:
    """Load bundled sample tasks from the sample_tasks/ directory."""
    sample_dir = Path(__file__).parent / "sample_tasks"
    rows: list[dict[str, Any]] = []

    if not sample_dir.exists():
        raise RuntimeError(
            f"sample_tasks/ directory not found at {sample_dir}. "
            "Add at least one task with a task.yaml, test.sh, and tests/test_outputs.py."
        )

    for task_dir in sorted(sample_dir.iterdir()):
        if not task_dir.is_dir():
            continue

        task_yaml_path = task_dir / "task.yaml"
        if not task_yaml_path.exists():
            continue

        with open(task_yaml_path) as f:
            task_meta = yaml.safe_load(f)

        description = task_meta["descriptions"][0]["description"]

        test_sh_path = task_dir / "test.sh"
        test_outputs_path = task_dir / "tests" / "test_outputs.py"

        test_sh = test_sh_path.read_text() if test_sh_path.exists() else ""
        test_outputs_py = test_outputs_path.read_text() if test_outputs_path.exists() else ""

        docker_image = task_meta.get("docker_image", "ubuntu:24.04")

        rows.append({
            "question": description,
            "info": json.dumps({
                "task_id": task_dir.name,
                "docker_image": docker_image,
                "test_sh": test_sh,
                "test_outputs_py": test_outputs_py,
                "max_test_timeout": task_meta.get("max_test_timeout_sec", DEFAULT_TEST_TIMEOUT),
            }),
        })

    if not rows:
        raise RuntimeError("No valid tasks found in sample_tasks/.")

    return Dataset.from_list(rows)


class TerminalBenchEnv(SandboxEnv):
    """Verifiers environment for TerminalBench2-style tasks.

    Each rollout spins up a Prime Sandbox with the task's Docker image,
    lets the model interact via bash, then injects and runs the task's
    pytest test suite to compute rewards.
    """

    def __init__(
        self,
        max_turns: int = DEFAULT_MAX_TURNS,
        timeout_per_command_seconds: int = 60,
        **kwargs,
    ):
        super().__init__(
            max_turns=max_turns,
            timeout_per_command_seconds=timeout_per_command_seconds,
            **kwargs,
        )

    def get_sandbox_request(self, state: vf.State):
        """Configure the sandbox with the task-specific Docker image."""
        request = super().get_sandbox_request(state)
        info = state["info"]
        if isinstance(info, str):
            info = json.loads(info)
        request.docker_image = info["docker_image"]
        return request

    async def _exec(self, sandbox_id: str, command: str, timeout: int = 30):
        return await self.sandbox_client.execute_command(
            sandbox_id, command, timeout=timeout
        )

    async def _write_file(self, sandbox_id: str, path: str, content: str):
        """Write file content into the sandbox using base64 to avoid escaping issues."""
        encoded = base64.b64encode(content.encode()).decode()
        await self._exec(sandbox_id, f"echo '{encoded}' | base64 -d > {path}")

    async def post_rollout(self, state: vf.State):
        """Inject test files and run verification after the agent finishes."""
        sid = state["sandbox_id"]
        info = state["info"]
        if isinstance(info, str):
            info = json.loads(info)

        test_sh = info.get("test_sh", "")
        test_outputs_py = info.get("test_outputs_py", "")
        max_test_timeout = info.get("max_test_timeout", DEFAULT_TEST_TIMEOUT)

        if not test_sh and not test_outputs_py:
            state["tests_passed"] = 0
            state["tests_total"] = 0
            state["binary_reward"] = 0.0
            return

        try:
            await self._exec(sid, "mkdir -p /tests /logs/verifier")

            if test_outputs_py:
                await self._write_file(sid, "/tests/test_outputs.py", test_outputs_py)

            if test_sh:
                await self._write_file(sid, "/tests/test.sh", test_sh)
                await self._exec(
                    sid,
                    "chmod +x /tests/test.sh && cd /tests && bash test.sh",
                    timeout=max_test_timeout,
                )
            else:
                await self._exec(
                    sid,
                    "pip install pytest -q 2>/dev/null; "
                    "python -m pytest /tests/test_outputs.py -v --tb=short 2>&1",
                    timeout=max_test_timeout,
                )

            reward_result = await self._exec(
                sid, "cat /logs/verifier/reward.txt 2>/dev/null || echo 0"
            )
            state["binary_reward"] = float(reward_result.stdout.strip() or "0")

            ctrf_result = await self._exec(
                sid, "cat /logs/verifier/ctrf.json 2>/dev/null || echo '{}'"
            )
            passed, total = parse_ctrf_results(ctrf_result.stdout)
            state["tests_passed"] = passed
            state["tests_total"] = total

        except Exception as e:
            logger.warning(f"Test execution failed for sandbox {sid}: {e}")
            state["tests_passed"] = 0
            state["tests_total"] = 0
            state["binary_reward"] = 0.0


async def test_pass_rate(state: vf.State) -> float:
    """Proportion of pytest tests passed (primary reward signal)."""
    total = state.get("tests_total", 0)
    if total == 0:
        return state.get("binary_reward", 0.0)
    return state.get("tests_passed", 0) / total


async def efficiency_bonus(state: vf.State) -> float:
    """Bonus for solving the task in fewer turns."""
    passed = state.get("tests_passed", 0)
    total = state.get("tests_total", 0)
    if total == 0 or passed < total:
        return 0.0
    turns = len(state.get("trajectory", []))
    return max(0.0, 1.0 - (turns / DEFAULT_MAX_TURNS))


def load_environment(
    max_turns: int = DEFAULT_MAX_TURNS,
    docker_image: str = "ubuntu:24.04",
) -> vf.Environment:
    """Load the TerminalBench2 environment.

    Args:
        max_turns: Maximum bash interactions per rollout.
        docker_image: Default Docker image (overridden per-task via info).
    """
    dataset = load_sample_tasks()

    rubric = vf.Rubric(
        funcs=[test_pass_rate, efficiency_bonus],
        weights=[1.0, 0.2],
    )

    return TerminalBenchEnv(
        dataset=dataset,
        rubric=rubric,
        system_prompt=SYSTEM_PROMPT,
        max_turns=max_turns,
        docker_image=docker_image,
    )
