import os
import sys
from importlib import import_module

from pyatb.timer import timer as Timer

'''
This file is used to initialize the pyatb package, 
including the creation of a workspace folder, and some global variables.
'''

def _should_skip_runtime_init() -> bool:
    argv0 = os.path.basename(sys.argv[0])
    return os.environ.get("PYATB_SKIP_RUNTIME_INIT") == "1" or argv0.startswith("pyatb_input") or "input_generator" in argv0


class _NoOpTimer:
    def program_start(self):
        return None

    def program_end(self):
        return None

    def print_all(self):
        return None


# workspace folder
INPUT_PATH = os.getcwd()
OUTPUT_PATH = os.path.join(INPUT_PATH, 'Out')
RUNNING_LOG = None

if _should_skip_runtime_init():
    COMM = None
    SIZE = 1
    RANK = 0
else:
    from pyatb.parallel import COMM, SIZE, RANK

timer = _NoOpTimer()


def initialize_runtime(input_path: str | None = None) -> None:
    global INPUT_PATH, OUTPUT_PATH, RUNNING_LOG, timer

    if _should_skip_runtime_init():
        return
    if not isinstance(timer, _NoOpTimer):
        return

    INPUT_PATH = os.path.abspath(os.getcwd() if input_path is None else input_path)
    OUTPUT_PATH = os.path.join(INPUT_PATH, 'Out')
    if RANK == 0:
        if not os.path.exists(OUTPUT_PATH):
            os.mkdir(OUTPUT_PATH)
        RUNNING_LOG = os.path.join(OUTPUT_PATH, 'running.log')
    else:
        RUNNING_LOG = os.path.join(OUTPUT_PATH, 'running-' + str(RANK) + '.log')

    timer = Timer(RUNNING_LOG, RANK, SIZE)
    timer.program_start()


def __getattr__(name: str):
    if name == "io":
        module = import_module("pyatb.io")
        globals()[name] = module
        return module
    if name == "init_tb":
        function = import_module("pyatb.init_tb").init_tb
        globals()[name] = function
        return function
    raise AttributeError(f"module 'pyatb' has no attribute {name!r}")

# if 'OMP_NUM_THREADS' not in os.environ:
#     os.environ['OMP_NUM_THREADS'] = '1'
