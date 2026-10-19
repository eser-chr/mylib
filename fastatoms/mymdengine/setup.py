"""Build the mymd extension (CPU always, GPU if nvcc is found).

    pip install -e .              # or: python setup.py build_ext --inplace

Environment variables:
    MYMD_NO_CUDA=1      build the CPU engine only
    CUDA_HOME=/path     CUDA installation (default: from `which nvcc`)
    CUDA_ARCH=sm_89     GPU architecture (default: native = the GPU in this machine)
    NVCC_CCBIN=g++-12   host compiler for nvcc (default: auto, picks g++-12 if the
                        default g++ is newer than nvcc supports)
"""
import os
import re
import shutil
import subprocess
import sys

import pybind11
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

HERE = os.path.dirname(os.path.abspath(__file__))


def find_cuda():
    if os.environ.get("MYMD_NO_CUDA"):
        return None
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = os.path.join(home, "bin", "nvcc") if home else shutil.which("nvcc")
    if not nvcc or not os.path.exists(nvcc):
        return None
    home = home or os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
    libdirs = [d for d in (os.path.join(home, "lib64"), os.path.join(home, "lib"),
                           "/usr/lib/x86_64-linux-gnu")
               if os.path.exists(os.path.join(d, "libcudart.so"))]
    return {"nvcc": nvcc, "home": home, "libdirs": libdirs[:1]}


def nvcc_host_compiler(nvcc):
    """nvcc refuses host compilers newer than it supports; pick an older g++ if needed."""
    if os.environ.get("NVCC_CCBIN"):
        return os.environ["NVCC_CCBIN"]
    out = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
    m = re.search(r"release (\d+)\.(\d+)", out)
    cuda = (int(m.group(1)), int(m.group(2))) if m else (99, 0)
    # Newest GCC supported by each CUDA version (from the CUDA installation guides).
    max_gcc = 12 if cuda < (12, 4) else 13 if cuda < (12, 8) else 14
    try:
        gcc = int(subprocess.run(["g++", "-dumpversion"], capture_output=True, text=True).stdout.split(".")[0])
    except (OSError, ValueError):
        return None
    if gcc <= max_gcc:
        return None
    for v in range(max_gcc, 8, -1):
        if shutil.which(f"g++-{v}"):
            return f"g++-{v}"
    return None


CUDA = find_cuda()

CXX_FLAGS = ["-O3", "-march=native", "-std=c++17"]
NVCC_FLAGS = ["-O3", "-std=c++17", "-Xcompiler", "-fPIC,-O3",
              f"-arch={os.environ.get('CUDA_ARCH', 'native')}"]

sources = ["src/bindings.cpp"]
macros, libs, libdirs = [], [], []
if CUDA:
    sources.append("src/atoms_gpu.cu")
    macros.append(("WITH_CUDA", None))
    libs.append("cudart")
    libdirs = CUDA["libdirs"]
    ccbin = nvcc_host_compiler(CUDA["nvcc"])
    if ccbin:
        NVCC_FLAGS += ["-ccbin", ccbin]


class BuildExt(build_ext):
    """Teach the default compiler to hand .cu files to nvcc."""

    def build_extensions(self):
        self.compiler.src_extensions.append(".cu")
        default_compile = self.compiler._compile

        def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src.endswith(".cu"):
                self.spawn([CUDA["nvcc"], "-c", src, "-o", obj, *pp_opts, *NVCC_FLAGS])
            else:
                default_compile(obj, src, ext, cc_args, CXX_FLAGS, pp_opts)

        self.compiler._compile = _compile
        super().build_extensions()


ext = Extension(
    "mymd._mymd",
    sources=sources,
    include_dirs=[pybind11.get_include(), os.path.join(HERE, "src")],
    define_macros=macros,
    libraries=libs,
    library_dirs=libdirs,
    runtime_library_dirs=libdirs,
    language="c++",
)

print(f"mymd: building with CUDA = {bool(CUDA)}", file=sys.stderr)

setup(
    name="mymd",
    version="0.1.0",
    description="Small Lennard-Jones MD engine with CPU and CUDA backends",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExt},
    install_requires=["numpy"],
    zip_safe=False,
)
