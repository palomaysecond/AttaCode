from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on Windows
    resource = None  # type: ignore[assignment]


def compiler_version(executable: str) -> str:
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"
    first_line = (result.stdout or result.stderr).splitlines()
    return first_line[0] if first_line else "unknown"


def _limit_resources(memory_limit_mb: int | None):
    def apply_limits() -> None:
        if memory_limit_mb and resource is not None:
            limit = int(memory_limit_mb) * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            except (ValueError, OSError):
                pass

    return apply_limits


def _trim(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n...[truncated {len(value) - limit} characters]"


def compile_and_run(program: str, compiler: dict[str, Any]) -> dict[str, Any]:
    executable = str(compiler.get("executable", "g++"))
    standard = str(compiler.get("standard", "gnu++17"))
    flags = [str(flag) for flag in compiler.get("flags", [])]
    compile_timeout = float(compiler.get("compile_timeout_seconds", 30))
    run_timeout = float(compiler.get("run_timeout_seconds", 10))
    memory_limit = compiler.get("memory_limit_mb")
    max_log = int(compiler.get("max_log_characters", 12000))

    with tempfile.TemporaryDirectory(prefix="rq5-humaneval-") as directory:
        work_dir = Path(directory)
        source_path = work_dir / "program.cpp"
        binary_path = work_dir / "program"
        source_path.write_text(program, encoding="utf-8")
        command = [executable, f"-std={standard}", *flags, str(source_path), "-o", str(binary_path)]

        compile_started = time.monotonic()
        try:
            compiled = subprocess.run(
                command,
                cwd=work_dir,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=compile_timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            return _failure("compiler_missing", str(exc), max_log)
        except subprocess.TimeoutExpired as exc:
            result = _failure("compile_timeout", str(exc), max_log)
            result["compile_seconds"] = round(time.monotonic() - compile_started, 6)
            return result

        compile_seconds = round(time.monotonic() - compile_started, 6)
        compile_stdout = _trim(compiled.stdout, max_log)
        compile_stderr = _trim(compiled.stderr, max_log)
        if compiled.returncode != 0:
            return {
                "compiled": False,
                "passed": False,
                "failure_kind": "compile_error",
                "compile_returncode": compiled.returncode,
                "compile_stdout": compile_stdout,
                "compile_stderr": compile_stderr,
                "compile_seconds": compile_seconds,
                "run_returncode": None,
                "run_stdout": "",
                "run_stderr": "",
                "run_seconds": None,
            }

        run_started = time.monotonic()
        try:
            executed = subprocess.run(
                [str(binary_path)],
                cwd=work_dir,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=run_timeout,
                check=False,
                preexec_fn=_limit_resources(int(memory_limit) if memory_limit else None) if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "compiled": True,
                "passed": False,
                "failure_kind": "runtime_timeout",
                "compile_returncode": compiled.returncode,
                "compile_stdout": compile_stdout,
                "compile_stderr": compile_stderr,
                "compile_seconds": compile_seconds,
                "run_returncode": None,
                "run_stdout": _trim(exc.stdout or "", max_log) if isinstance(exc.stdout, str) else "",
                "run_stderr": _trim(exc.stderr or "", max_log) if isinstance(exc.stderr, str) else "",
                "run_seconds": round(time.monotonic() - run_started, 6),
            }

        run_seconds = round(time.monotonic() - run_started, 6)
        passed = executed.returncode == 0
        if passed:
            failure_kind = None
        elif executed.returncode < 0:
            failure_kind = "runtime_signal"
        else:
            failure_kind = "test_failure"
        return {
            "compiled": True,
            "passed": passed,
            "failure_kind": failure_kind,
            "compile_returncode": compiled.returncode,
            "compile_stdout": compile_stdout,
            "compile_stderr": compile_stderr,
            "compile_seconds": compile_seconds,
            "run_returncode": executed.returncode,
            "run_stdout": _trim(executed.stdout, max_log),
            "run_stderr": _trim(executed.stderr, max_log),
            "run_seconds": run_seconds,
        }


def _failure(kind: str, message: str, max_log: int) -> dict[str, Any]:
    return {
        "compiled": False,
        "passed": False,
        "failure_kind": kind,
        "compile_returncode": None,
        "compile_stdout": "",
        "compile_stderr": _trim(message, max_log),
        "compile_seconds": None,
        "run_returncode": None,
        "run_stdout": "",
        "run_stderr": "",
        "run_seconds": None,
    }
