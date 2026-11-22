"""Error and output handling that never shows data values.

Why: error messages and library warnings can carry report text, UIDs or DICOM
values, and the notebooks' output is read by people and by Claude. The
competition's data terms keep those values inside Kaggle.

Three tools:
1. install() replaces every place an error or warning is printed (IPython,
   sys.excepthook, threading, warnings, logging) with a line that names only
   the error type and where it happened (file, line, function).
2. step() runs one notebook step. While it runs, everything written to stdout
   and stderr (Python and C level) goes to a temporary file outside
   /kaggle/working, which is never saved. If the step fails, it raises
   StepFailed, whose message is only the safe description, so the notebook
   runner's own error report carries no data either.
3. describe() builds that safe description; workers use it too.
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import tempfile
import threading
import traceback
import warnings

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


class StepFailed(RuntimeError):
    """Raised by step(); its message holds only the error type and location."""


def _where(tb) -> tuple[str, int, str]:
    """Innermost frame inside this package, else the innermost frame."""
    frames = traceback.extract_tb(tb) if tb is not None else []
    if not frames:
        return "?", 0, "?"
    here = os.path.abspath(__file__)
    frames = [f for f in frames if os.path.abspath(f.filename) != here] or frames
    ours = [f for f in frames if os.path.abspath(f.filename).startswith(_PKG_DIR)]
    f = (ours or frames)[-1]
    return os.path.basename(f.filename), int(f.lineno or 0), f.name


def describe(etype: type, tb) -> str:
    """Safe one-line description: type and location, never the message."""
    name = getattr(etype, "__name__", "Exception")
    file, line, func = _where(tb)
    return f"{name} at {file}:{line} in {func}()"


def _excepthook(etype, value, tb):
    sys.__stderr__.write("ERROR " + describe(etype, tb) + "\n")


def _thread_hook(args):
    sys.__stderr__.write("ERROR (thread) " + describe(args.exc_type, args.exc_traceback) + "\n")


def _showwarning(message, category, filename, lineno, file=None, line=None):
    name = getattr(category, "__name__", "Warning")
    sys.__stderr__.write(f"WARNING {name} at {os.path.basename(str(filename))}:{lineno}\n")


class _NoMessageFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        return f"{record.levelname} {record.name} at {os.path.basename(record.pathname)}:{record.lineno}"


def _ipython_handler(shell, etype, value, tb, tb_offset=None):
    print("ERROR " + describe(etype, tb))
    return None


_installed = False


def install() -> None:
    """Install the safe handlers. Safe to call more than once."""
    global _installed
    sys.excepthook = _excepthook
    threading.excepthook = _thread_hook
    warnings.showwarning = _showwarning
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.__stderr__)
    handler.setFormatter(_NoMessageFormatter())
    root.addHandler(handler)
    root.setLevel(logging.WARNING)
    for name in ("pydicom", "PIL", "timm", "lightgbm", "optuna"):
        lg = logging.getLogger(name)
        for h in list(lg.handlers):
            lg.removeHandler(h)
        lg.propagate = True
    try:  # inside a notebook
        from IPython import get_ipython  # type: ignore

        ip = get_ipython()
        if ip is not None:
            ip.set_custom_exc((BaseException,), _ipython_handler)
    except Exception:  # noqa: BLE001 - no IPython is fine
        pass
    _installed = True


def worker_init() -> None:
    """Initializer for worker processes (ProcessPoolExecutor, multiprocessing)."""
    install()


@contextlib.contextmanager
def _redirect_all(path: str):
    """Send fd 1, fd 2, sys.stdout and sys.stderr to a file, then restore."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    py_out, py_err = sys.stdout, sys.stderr
    f = open(path, "a", buffering=1, encoding="utf-8", errors="replace")
    try:
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)
        sys.stdout = f
        sys.stderr = f
        yield
    finally:
        try:
            f.flush()
        except Exception:  # noqa: BLE001
            pass
        sys.stdout, sys.stderr = py_out, py_err
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        f.close()


@contextlib.contextmanager
def step(name: str, quiet: bool = True):
    """Run one step safely.

    quiet=True hides all output of the step (it goes to a temp file that is
    never saved). On failure, prints and raises only the safe description.
    """
    install()
    log_path = os.path.join(tempfile.gettempdir(), f"knee_{name}_quiet.log")
    ctx = _redirect_all(log_path) if quiet else contextlib.nullcontext()
    failure: str | None = None
    try:
        with ctx:
            try:
                yield
            except BaseException as e:  # noqa: BLE001 - we re-raise a safe one
                failure = describe(type(e), e.__traceback__)
    finally:
        pass
    if failure is not None:
        print(f"STEP {name}: FAILED - {failure}")
        raise StepFailed(failure) from None
    print(f"STEP {name}: OK")
