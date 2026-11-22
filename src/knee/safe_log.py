"""Error and output handling that never shows data values.

Why: error messages, warnings and library output can carry report text, UIDs
or DICOM values, and the notebooks' output is read by people and by Claude.
The competition's data terms keep those values inside Kaggle.

1. install() replaces every place an error or warning is printed (IPython,
   sys.excepthook, threading, unraisable exceptions, warnings, logging) with a
   line that names only the error type and where it happened.
2. step() runs one notebook step. While it runs, everything written to stdout
   and stderr (Python and C level, workers included) goes to a temporary file
   outside /kaggle/working, which is never saved. On failure it copies that
   file and the full traceback to a private error log (for Ablay only), keeps
   the exception for show_last(), and raises StepFailed, whose message is only
   the safe description, so the notebook runner's error report carries no data.
   A watchdog interrupts the step after watchdog_s seconds: it sends a real
   SIGINT to the main thread (which also breaks a blocking wait on a worker),
   and if the step has not ended WATCHDOG_GRACE_S later, it terminates this
   process's worker processes, because a pool cannot shut down while a worker
   hangs.
3. block() prints text between fixed markers; Claude reads only these blocks.
"""
from __future__ import annotations

import _thread
import contextlib
import logging
import os
import signal
import sys
import tempfile
import threading
import traceback
import warnings
from pathlib import Path

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_HERE = os.path.abspath(__file__)
BEGIN = "=== KNEE SUMMARY BEGIN ==="
END = "=== KNEE SUMMARY END ==="
WATCHDOG_GRACE_S = 5.0

_last: tuple | None = None  # (type, value, traceback) of the last failed step


class StepFailed(RuntimeError):
    """Raised by step(); its message holds only the error type and location."""


def _where(tb) -> tuple[str, int, str]:
    """Innermost frame inside this package (not this file), else the innermost frame."""
    frames = traceback.extract_tb(tb) if tb is not None else []
    if not frames:
        return "?", 0, "?"
    frames = [f for f in frames if os.path.abspath(f.filename) != _HERE] or frames
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


def _unraisable_hook(u):
    sys.__stderr__.write("ERROR (unraisable) " + describe(u.exc_type, u.exc_traceback) + "\n")


def _showwarning(message, category, filename, lineno, file=None, line=None):
    name = getattr(category, "__name__", "Warning")
    sys.__stderr__.write(f"WARNING {name} at {os.path.basename(str(filename))}:{lineno}\n")


class _NoMessageFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        return f"{record.levelname} {record.name} at {os.path.basename(record.pathname)}:{record.lineno}"


def _ipython_handler(shell, etype, value, tb, tb_offset=None):
    print("ERROR " + describe(etype, tb))
    return None


def _strip_loggers() -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.__stderr__)
    handler.setFormatter(_NoMessageFormatter())
    root.addHandler(handler)
    root.setLevel(logging.WARNING)
    for name in list(logging.root.manager.loggerDict):
        lg = logging.getLogger(name)
        for h in list(lg.handlers):
            lg.removeHandler(h)
        lg.propagate = True
    logging.raiseExceptions = False  # logging's own error report prints the message and args


def install() -> None:
    """Install the safe handlers. Safe to call more than once."""
    sys.excepthook = _excepthook
    threading.excepthook = _thread_hook
    sys.unraisablehook = _unraisable_hook
    warnings.showwarning = _showwarning
    _strip_loggers()
    try:  # inside a notebook
        from IPython import get_ipython  # type: ignore

        ip = get_ipython()
        if ip is not None:
            ip.set_custom_exc((BaseException,), _ipython_handler)
    except Exception:  # noqa: BLE001 - no IPython is fine
        pass


def worker_init() -> None:
    """Initializer for worker processes (ProcessPoolExecutor, multiprocessing)."""
    install()


def block(text: str) -> None:
    """Print text between the fixed markers (the only output Claude reads)."""
    print(f"{BEGIN}\n{text}\n{END}", flush=True)


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


def _interrupt_main() -> None:
    """Raise KeyboardInterrupt in the main thread, also when it is blocked in a wait.

    _thread.interrupt_main() alone acts only between bytecodes, so a main thread
    waiting on a hung worker's result never sees it; a real SIGINT breaks the wait.
    """
    try:
        if callable(signal.getsignal(signal.SIGINT)):
            signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)
            return
    except Exception:  # noqa: BLE001 - fall back to the simulated interrupt
        pass
    _thread.interrupt_main()


def _stop_workers() -> None:
    """Terminate this process's multiprocessing children (pool workers)."""
    try:
        import multiprocessing

        for child in multiprocessing.active_children():
            with contextlib.suppress(Exception):
                child.terminate()
    except Exception:  # noqa: BLE001 - best effort
        pass


@contextlib.contextmanager
def step(name: str, quiet: bool = True, error_log: Path | None = None, watchdog_s: float | None = None):
    """Run one step safely (see the module docstring)."""
    global _last
    install()
    log_path = os.path.join(tempfile.gettempdir(), f"knee_{name}_quiet.log")
    with contextlib.suppress(FileNotFoundError):
        os.remove(log_path)
    fired = threading.Event()
    ended = threading.Event()  # set as soon as the step body has ended, either way

    def _fire():
        if ended.is_set():
            return
        fired.set()
        _interrupt_main()
        if not ended.wait(WATCHDOG_GRACE_S):
            _stop_workers()

    timer = None
    if watchdog_s:
        timer = threading.Timer(watchdog_s, _fire)
        timer.daemon = True
        timer.start()
    ctx = _redirect_all(log_path) if quiet else contextlib.nullcontext()
    failure: str | None = None
    try:
        with ctx:
            try:
                yield
            except BaseException as e:  # noqa: BLE001 - we re-raise a safe one
                failure = ("watchdog: " if fired.is_set() else "") + describe(type(e), e.__traceback__)
                _last = (type(e), e, e.__traceback__)
            finally:
                ended.set()
    finally:
        if timer is not None:
            timer.cancel()
    if failure is not None:
        if error_log is not None:
            try:
                error_log = Path(error_log)
                error_log.parent.mkdir(parents=True, exist_ok=True)
                with open(error_log, "w", encoding="utf-8", errors="replace") as out:
                    if os.path.exists(log_path):
                        with open(log_path, encoding="utf-8", errors="replace") as src:
                            out.write(src.read())
                    out.write("\n--- traceback ---\n")
                    out.write("".join(traceback.format_exception(*_last)))
            except Exception:  # noqa: BLE001 - the error log is best effort
                pass
        block(f"STEP {name}: FAILED - {failure}")
        raise StepFailed(failure) from None
    block(f"STEP {name}: OK")


def show_last() -> None:
    """Print the full traceback of the last failed step. For Ablay only, in a session only he watches."""
    if _last is None:
        print("no failed step")
        return
    traceback.print_exception(*_last, file=sys.__stdout__)
