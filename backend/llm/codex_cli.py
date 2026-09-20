"""Stateless Codex CLI completions billed through the user's ChatGPT/Codex account.

No shared conversational thread, project instructions, tools, or API-key fallback.
Temperature and a hard output-token cap are not exposed by `codex exec`; the
requested reply budget is an instruction. Record/replay remains in LLMClient.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading

from backend.llm.client import Backend, NonRetryableProviderFailure, ProviderFailure
from backend.llm.environment import setting

DEFAULT_MODEL = "gpt-5.6-luna"

# Prevent accidental use of API billing or an inherited API endpoint. CODEX_HOME
# stays untouched: the CLI owns credentials and refresh; we never read auth.json.
_API_ENV = {"OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL"}
_DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "shell_snapshot", "apps", "plugins", "remote_plugin",
    "multi_agent", "multi_agent_v2", "memories", "hooks", "browser_use",
    "browser_use_external", "computer_use", "image_generation", "view_image",
    "code_mode_host", "goals", "sleep_tool", "skill_search", "skill_mcp_dependency_install",
    "workspace_dependencies", "tool_suggest", "unbounded_connection_retries",
)
_STARTUP_ADVISORIES = (
    "Under-development features enabled: skip_host_skill_discovery.",
    "Code Mode is unavailable because code-mode host is disabled.",
)


def executable():
    candidate = setting("CODEX_CLI_PATH") or "codex"
    found = shutil.which(candidate)
    if not found:
        raise NonRetryableProviderFailure(
            "Codex CLI was not found. Install Codex, restart this terminal/server, "
            "or set CODEX_CLI_PATH to its executable in .env.")
    return found


def process_options():
    return {"env": {k: v for k, v in os.environ.items() if k.upper() not in _API_ENV},
            "encoding": "utf-8", "errors": "replace", "text": True, "capture_output": True,
            "creationflags": subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0}


def check_login():
    """Local status only: do not launch login, reveal credentials, or make a model call."""
    try:
        result = subprocess.run([executable(), "login", "status"], timeout=15, **process_options())
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NonRetryableProviderFailure("Could not check Codex login. Run: codex login status") from exc
    status = (result.stdout + "\n" + result.stderr).lower()
    if result.returncode or "logged in" not in status or "chatgpt" not in status:
        raise NonRetryableProviderFailure(
            "Codex needs ChatGPT sign-in to use your plan/credits. Run codex login, "
            "then codex login status in the terminal that launches the experiment. "
            "API-key authentication is not used by codex_cli.")


def _failure(detail):
    # Diagnose internally, but never echo CLI output: it can contain prompts/tokens.
    lower = detail.lower()
    if any(s in lower for s in ("usage limit", "quota", "insufficient", "credits", "rate limit", "429")):
        return NonRetryableProviderFailure(
            "Codex usage/rate limit reached. Check your Codex allowance and reset time before resuming.")
    if any(s in lower for s in ("auth", "login", "logged in", "401", "403", "token expired")):
        return NonRetryableProviderFailure("Codex authentication failed. Run codex login and check your ChatGPT account.")
    if any(s in lower for s in ("model", "unsupported", "unexpected argument", "invalid value", "config", "feature")):
        return NonRetryableProviderFailure(
            "Codex rejected the model or CLI configuration. Update Codex and select a model available in /model.")
    return ProviderFailure("Codex CLI request failed; check connectivity and Codex service status.")


class CodexCLIBackend(Backend):
    name = "codex_cli"

    def __init__(self, model=None, timeout=180):
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout
        self._login_checked = False
        self._login_lock = threading.Lock()

    def generate(self, prompt, system, max_tokens, temperature):
        with self._login_lock:
            if not self._login_checked:
                check_login()
                self._login_checked = True
        binary = executable()
        # Each call gets an empty working directory and a new ephemeral session.
        # File paths/argv contain no prompt text (Windows command-line length).
        with tempfile.TemporaryDirectory(prefix="memeworld-codex-") as folder:
            root = Path(folder)
            instructions = root / "instructions.txt"
            instructions.write_text(
                (system or "You are a helpful assistant.")
                + "\nReturn only the requested response, respecting its exact format. "
                "Use only the supplied context; do not use tools or inspect files. "
                f"Keep the response within approximately {int(max_tokens)} tokens.", encoding="utf-8")
            output = root / "answer.txt"
            config = {
                "model_provider": "openai", "forced_login_method": "chatgpt",
                "model_instructions_file": str(instructions), "project_doc_max_bytes": 0,
                "model_reasoning_effort": "low", "model_reasoning_summary": "none",
                "model_verbosity": "low", "web_search": "disabled", "approval_policy": "never",
                "history.persistence": "none", "mcp_servers": {},
            }
            cmd = [binary, "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
                   "--sandbox", "read-only", "--model", self.model, "--json", "--color", "never",
                   "--output-last-message", str(output)]
            for key, value in config.items():
                # JSON strings/numbers are also valid TOML scalar values. Empty
                # mappings use TOML's empty inline-table syntax.
                cmd += ["-c", key + "=" + ("{}" if value == {} else json.dumps(value))]
            for feature in _DISABLED_FEATURES:
                cmd += ["--disable", feature]
            cmd += ["--enable", "skip_host_skill_discovery"]
            cmd.append("-")
            try:
                result = subprocess.run(cmd, input=prompt, cwd=root, timeout=self.timeout, **process_options())
            except subprocess.TimeoutExpired as exc:
                raise NonRetryableProviderFailure(
                    "Codex request timed out; the CLI process was stopped. Check login, limits, and connectivity before resuming.") from exc
            except OSError as exc:
                raise NonRetryableProviderFailure("Could not start Codex CLI; check CODEX_CLI_PATH and installation.") from exc
            if result.returncode:
                raise _failure(result.stderr + "\n" + result.stdout)
            completed = False
            started = False
            for line in result.stdout.splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                if event.get("type") in ("turn.failed", "error"):
                    raise _failure(json.dumps(event))
                started |= event.get("type") == "turn.started"
                if event.get("type", "").startswith("item."):
                    item = event.get("item") or {}
                    item_type = item.get("type")
                    if item_type == "error":
                        # This CLI version reports intentional disabled-tool startup
                        # notices as error items, even on successful text-only turns.
                        if not started and str(item.get("message", "")).startswith(_STARTUP_ADVISORIES):
                            continue
                        raise _failure(str(item.get("message", "")))
                    if item_type not in (None, "agent_message", "reasoning"):
                        raise NonRetryableProviderFailure(
                            "Codex emitted a tool/action event; this response was rejected to preserve experiment isolation.")
                completed |= event.get("type") == "turn.completed"
            if not completed:
                raise ProviderFailure("Codex did not report a completed turn; no response was recorded.")
            answer = output.read_text(encoding="utf-8").strip() if output.exists() else ""
            if not answer:
                raise ProviderFailure("Codex returned no final response.")
            return answer
