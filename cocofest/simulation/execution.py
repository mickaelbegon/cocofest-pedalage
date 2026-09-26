"""Subprocess lifecycle shared by the GUI, with no shell evaluation."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

from .launch import LaunchPlan, ROOT


def runtime_helpers(root: Path = ROOT):
    """Load the existing standard-library environment policy only when requested."""
    name = "_cocofest_simulation_runtime"
    path = Path(root) / ".github/scripts/run_benchmarks.py"
    module = sys.modules.get(name)
    if module is not None and Path(module.__file__) == path:
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SimulationProcess:
    """One process tree, streamed logs and graceful then forced cancellation.

    This class tracks process status only. Scientific conclusions must come
    from the fresh result JSON identified by the plan.
    """

    def __init__(self):
        self.process = None
        self.lines = queue.Queue()
        self.reader = None
        self.stop_requested_at = None
        self.returncode = None
        self.log_path = None

    @property
    def running(self):
        return self.process is not None and (
            self.process.poll() is None or (self.reader is not None and self.reader.is_alive()))

    def start(self, plan: LaunchPlan, environment: dict[str, str], log_path: Path):
        if self.running:
            raise RuntimeError("Une simulation est déjà en cours")
        if plan.result_json is not None and plan.result_json.exists():
            raise FileExistsError(f"Résultat existant : choisissez un nouveau dossier ({plan.result_json})")
        log_path = Path(log_path)
        if log_path.exists():
            raise FileExistsError(f"Journal existant : choisissez un nouveau dossier ({log_path})")
        plan.save_effective_configuration()
        plan.cwd.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("x", encoding="utf-8")
        try:
            process = subprocess.Popen(plan.argv, cwd=plan.cwd, env=environment, shell=False,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", bufsize=1,
                                       start_new_session=os.name == "posix")
        except BaseException:
            log.close()
            # This empty file belongs to the failed launch, so a retry can use
            # the same output without removing any pre-existing user file.
            log_path.unlink()
            raise
        self.process = process
        self.returncode = None
        self.stop_requested_at = None
        self.log_path = log_path
        self.lines = queue.Queue()
        def read_output():
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    self.lines.put(line)
            finally:
                process.stdout.close()
                log.close()
        self.reader = threading.Thread(target=read_output, name="simulation-log", daemon=True)
        self.reader.start()

    def drain(self, limit=1000):
        lines = []
        for _ in range(limit):
            try:
                lines.append(self.lines.get_nowait())
            except queue.Empty:
                break
        return lines

    def stop(self):
        if not self.running:
            return
        self.stop_requested_at = time.monotonic()
        self._signal(signal.SIGTERM)

    def _signal(self, sig):
        try:
            if os.name == "posix":
                os.killpg(self.process.pid, sig)
            elif sig == signal.SIGTERM:
                self.process.terminate()
            else:
                self.process.kill()
        except ProcessLookupError:
            pass

    def poll(self):
        if self.process is None:
            return None
        if self.stop_requested_at is not None and self.running and time.monotonic() - self.stop_requested_at > 5:
            self._signal(signal.SIGKILL)
        self.returncode = self.process.poll()
        return self.returncode
