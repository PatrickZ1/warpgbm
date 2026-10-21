import importlib

_NODE_KERNEL = None
_NODE_KERNEL_IMPORT_ERROR = None

_IMPORT_HINT = (
    "WarpGBM CUDA extension 'warpgbm.cuda.node_kernel' is not available. "
    "Build/install with CUDA-enabled PyTorch and CUDA toolkit, then run "
    "`python setup.py build_ext --inplace`."
)


def _get_node_kernel():
    """The compiled CUDA extension, imported on first use."""
    global _NODE_KERNEL
    global _NODE_KERNEL_IMPORT_ERROR

    if _NODE_KERNEL is not None:
        return _NODE_KERNEL

    if _NODE_KERNEL_IMPORT_ERROR is not None:
        raise ImportError(_IMPORT_HINT) from _NODE_KERNEL_IMPORT_ERROR

    try:
        _NODE_KERNEL = importlib.import_module("warpgbm.cuda.node_kernel")
    except ImportError as e:
        _NODE_KERNEL_IMPORT_ERROR = e
        raise ImportError(_IMPORT_HINT) from e

    return _NODE_KERNEL
