import os
import re
import subprocess
import sys
from setuptools import setup, find_packages

# Commands that only collect metadata or package sources and never compile: they may run
# without a CUDA toolchain. Every other command (wheel builds, installs) needs the extension.
METADATA_ONLY_COMMANDS = {"sdist", "egg_info", "dist_info", "clean"}

# Set WARPGBM_SKIP_CUDA=1 to build without the CUDA extension anyway (the result cannot train
# or predict; only useful for packaging or docs tooling)
SKIP_CUDA = os.environ.get("WARPGBM_SKIP_CUDA", "0") == "1"

INSTALL_HELP = """
How to fix:
  * Install a PyTorch build whose CUDA version matches your toolkit, e.g. for CUDA 11.8:
        pip install torch --index-url https://download.pytorch.org/whl/cu118
  * or install a CUDA toolkit matching your PyTorch and point CUDA_HOME at it, e.g.
        export CUDA_HOME=/usr/local/cuda-12.8
  then reinstall WarpGBM with: pip install --no-build-isolation <warpgbm source>
"""

try:
    import torch
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME
    torch_available = True
except ImportError:
    torch_available = False
    CUDA_HOME = None


def needs_extension():
    # The first positional argument is the command: bdist_wheel/build_ext/install/develop/...
    # (PEP 517 backends call setup.py the same way). No command, e.g. `setup.py --version`.
    command = next((arg for arg in sys.argv[1:] if not arg.startswith("-")), None)
    return command is not None and command not in METADATA_ONLY_COMMANDS


def nvcc_version(cuda_home):
    """Returns the 'major.minor' release of the nvcc in cuda_home, or None if it cannot be run."""
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    try:
        out = subprocess.check_output([nvcc, "--version"], stderr=subprocess.STDOUT, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"release (\d+\.\d+)", out)
    return match.group(1) if match else None


def cuda_problem():
    """Returns None if the CUDA extension can be built, else a description of what is wrong."""
    if not torch_available:
        return "PyTorch is not installed in the build environment."
    torch_cuda = torch.version.cuda
    if torch_cuda is None:
        return (
            f"The installed PyTorch ({torch.__version__}) is a CPU-only build; WarpGBM needs a "
            "CUDA-enabled PyTorch."
        )
    if CUDA_HOME is None:
        return (
            f"No CUDA toolkit (nvcc) found, but PyTorch {torch.__version__} was built for CUDA "
            f"{torch_cuda}. Install the CUDA {torch_cuda.split('.')[0]}.x toolkit and set CUDA_HOME."
        )
    nvcc = nvcc_version(CUDA_HOME)
    print(f"WarpGBM build: PyTorch {torch.__version__} (CUDA {torch_cuda}), nvcc {nvcc or 'unknown'} at {CUDA_HOME}")
    if nvcc is not None and nvcc.split(".")[0] != torch_cuda.split(".")[0]:
        return (
            f"CUDA major version mismatch: nvcc at {CUDA_HOME} is CUDA {nvcc}, but PyTorch "
            f"{torch.__version__} was built for CUDA {torch_cuda}. Both must have the same major version."
        )
    return None


def get_extensions():
    if SKIP_CUDA or not needs_extension():
        return []
    problem = cuda_problem()
    if problem is not None:
        # WarpGBM has no CPU code path: a package installed without the extension cannot train
        # or predict, so fail now rather than produce a broken install
        raise RuntimeError(f"\n\nCannot build the WarpGBM CUDA extension.\n{problem}\n{INSTALL_HELP}")
    return [
        CUDAExtension(
            name="warpgbm.cuda.node_kernel",
            sources=[
                "warpgbm/cuda/histogram_kernel.cu",
                "warpgbm/cuda/best_split_kernel.cu",
                "warpgbm/cuda/binner.cu",
                "warpgbm/cuda/predict.cu",
                "warpgbm/cuda/node_kernel.cpp",
            ],
        )
    ]

# Read version
with open("version.txt") as f:
    version = f.read().strip()

setup(
    name="warpgbm",
    version=version,
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension} if torch_available else {},
    install_requires=[
        "torch",
        "numpy",
        "scikit-learn",
        "tqdm",
    ],
)
