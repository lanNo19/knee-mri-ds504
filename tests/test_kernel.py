"""Safe logging inside a real Jupyter kernel, run the way Kaggle runs a saved version (papermill).

A fake UID is printed at Python and file-descriptor level, by the parent and by
worker processes, and carried in exception messages. It must not appear in the
output notebook or in the runner's own output, except through no route at all.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

nbformat = pytest.importorskip("nbformat")
pytest.importorskip("papermill")

SECRET = "1.2.826.0.1.3680043.8.498.777777"
CELL1 = "import knee.safe_log as sl; sl.install()"
WORKER = f'''
import logging, os, warnings
from concurrent.futures import ProcessPoolExecutor
from knee import safe_log
SECRET = "{SECRET[:12]}" + "{SECRET[12:]}"  # built at run time, so the source never holds it

def worker(i):
    print("worker-py " + SECRET, flush=True)
    os.write(1, ("worker-fd1 " + SECRET + "\\n").encode())
    os.write(2, ("worker-fd2 " + SECRET + "\\n").encode())
    warnings.warn("worker-warn " + SECRET)
    logging.getLogger("x").warning("worker-log %s", SECRET)
    if i == 1:
        raise ValueError("worker-exc " + SECRET)
    return i
'''

CASES = {
    "quiet_workers": WORKER + '''
with safe_log.step("probe", quiet=True):
    print("parent " + SECRET)
    os.write(2, ("parent-fd2 " + SECRET + "\\n").encode())
    with ProcessPoolExecutor(2, initializer=safe_log.worker_init) as ex:
        res = []
        for f in [ex.submit(worker, i) for i in range(3)]:
            try:
                res.append(f.result())
            except Exception as e:
                res.append(type(e).__name__)
''',
    "worker_uncaught": WORKER + '''
with safe_log.step("probe", quiet=True):
    with ProcessPoolExecutor(2, initializer=safe_log.worker_init) as ex:
        list(ex.map(worker, range(3)))
''',
    "watchdog_hung_worker": WORKER + '''
import time
def hang(i):
    time.sleep(600)
with safe_log.step("probe", quiet=True, watchdog_s=2):
    with ProcessPoolExecutor(2, initializer=safe_log.worker_init) as ex:
        list(ex.map(hang, range(2)))
''',
    "fail_in_step": f'''
from knee import safe_log
with safe_log.step("probe", quiet=True):
    raise KeyError("step-exc " + "{SECRET[:12]}" + "{SECRET[12:]}")
''',
}


def _run(tmp_path, name, code):
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(CELL1), nbformat.v4.new_code_cell(code)]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    src, out = tmp_path / f"{name}.ipynb", tmp_path / f"{name}.out.ipynb"
    nbformat.write(nb, src)
    p = subprocess.run([sys.executable, "-m", "papermill", str(src), str(out), "--log-output", "-k", "python3"],
                       capture_output=True, text=True, timeout=300)
    return p, (out.read_text() if out.exists() else "")


@pytest.mark.parametrize("name", list(CASES))
def test_no_secret_reaches_notebook_or_runner(tmp_path, name):
    p, nbtxt = _run(tmp_path, name, CASES[name])
    assert nbtxt, p.stderr[-500:]
    assert SECRET not in nbtxt
    assert SECRET not in p.stdout and SECRET not in p.stderr
    if name != "quiet_workers":
        assert p.returncode != 0  # the failure still fails the run
        outs = [o for c in json.loads(nbtxt)["cells"] for o in c.get("outputs", [])]
        assert any("FAILED" in "".join(o.get("text", "")) for o in outs if o.get("output_type") == "stream")
