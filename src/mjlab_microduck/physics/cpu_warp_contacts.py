"""Opt-in CPU replay of the pinned Warp foot/plane collision algorithm.

Install on the physics owner's main thread before starting physics workers.
MuJoCo's collision table is process-global; the native dispatcher only changes
explicitly registered models and named feet. Other models/pairs stay native.
This aligns an algorithm, not a calibrated hardware contact model.
"""

import ctypes
import fcntl
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import weakref
from importlib.metadata import version
from pathlib import Path

import mujoco

_LIBRARY = None
_LIBRARY_LOCK = threading.Lock()
SOURCE_PATHS = (Path(__file__), Path(__file__).with_name("_cpu_warp_contacts.cpp"))


def _library():
    global _LIBRARY
    with _LIBRARY_LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        if sys.platform != "linux":
            raise RuntimeError(
                "CPU Warp contact compatibility is currently validated on Linux"
            )
        if mujoco.__version__ != "3.13.0" or version("mujoco-warp") != "3.8.1":
            raise RuntimeError(
                "CPU contact compatibility requires MuJoCo 3.13.0 / mujoco-warp 3.8.1"
            )
        compiler = shutil.which("c++")
        if compiler is None:
            raise RuntimeError(
                "CPU contact compatibility requires a C++17 compiler (c++)"
            )
        package = Path(mujoco.__file__).resolve().parent
        include = package / "include"
        library = package / "libmujoco.so.3.13.0"
        source = SOURCE_PATHS[1]
        digest = hashlib.sha256(source.read_bytes())
        # Invalidate cached code when build flags/loader logic or the linked
        # binary change, even if their filenames and version stay the same.
        digest.update(SOURCE_PATHS[0].read_bytes())
        digest.update(library.read_bytes())
        digest.update(str(library).encode())
        for path in sorted((include / "mujoco").glob("*.h")):
            digest.update(path.read_bytes())
        cache_root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        cache = cache_root / "microduck_rl/cpu_contacts" / digest.hexdigest()
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / "contacts.so"
        # A second evaluator may start while the first compiles this exact code.
        # Publish a complete library atomically, never a partly linked .so.
        with (cache / "build.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not target.is_file():
                with tempfile.TemporaryDirectory(dir=cache) as temporary:
                    output = Path(temporary) / "contacts.so"
                    command = [
                        compiler,
                        "-shared",
                        "-fPIC",
                        "-O2",
                        "-std=c++17",
                        "-pthread",
                        "-ffp-contract=fast",
                        "-mfma",
                        "-I" + str(include),
                        str(source),
                        str(library),
                        "-Wl,-rpath," + str(package),
                        "-o",
                        str(output),
                    ]
                    result = subprocess.run(
                        command, capture_output=True, text=True, check=False
                    )
                    if result.returncode:
                        raise RuntimeError(
                            "CPU contact extension compilation failed:\n"
                            + result.stderr
                        )
                    output.replace(target)
        lib = ctypes.CDLL(str(target))
        lib.register_model.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.register_model.restype = ctypes.c_int
        lib.unregister_model.argtypes = [ctypes.c_void_p]
        lib.unregister_model.restype = None
        # Hold the shared object for the process lifetime: its dispatcher remains
        # in MuJoCo's collision table even when the registration map is empty.
        _LIBRARY = lib
        return lib


def _unregister(lib, model):
    lib.unregister_model(model._address)


class CpuWarpFootContacts:
    """Lifetime handle for one model; close only after its physics workers stop."""

    def __init__(self, model: mujoco.MjModel):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError(
                "Install CPU contact compatibility on the main thread before physics workers start"
            )
        feet = [
            i
            for i in range(model.ngeom)
            if model.geom(i).name.split("/")[-1]
            in ("left_foot_collision", "right_foot_collision")
        ]
        if len(feet) != 2 or any(
            model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH for g in feet
        ):
            raise ValueError("Expected two named foot mesh geoms")
        lib = _library()
        if lib.register_model(model._address, *feet):
            raise RuntimeError(
                "Model already registered or another collision dispatcher replaced this one"
            )
        # The finalizer holds the model until unregistration, preventing stale
        # pointers from surviving model destruction or address reuse.
        self._finalizer = weakref.finalize(self, _unregister, lib, model)
        self.model = model

    def close(self):
        self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
