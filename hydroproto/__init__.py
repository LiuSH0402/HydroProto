"""HydroProto: prototype-guided bacterial hydrolase classification."""

from importlib import import_module

__all__ = [
    "SequenceMLP",
    "HydroLocalHead",
    "HydroProtoHead",
    "HydroMPC650MHead",
]

__version__ = "0.1.0"


_LAZY_EXPORTS = {
    "SequenceMLP": ("hydroproto.esm2_mlp", "SequenceMLP"),
    "HydroLocalHead": ("hydroproto.hydrolocal", "HydroLocalHead"),
    "HydroProtoHead": ("hydroproto.hydroproto", "HydroProtoHead"),
    "HydroMPC650MHead": ("hydroproto.hydrompc", "HydroMPC650MHead"),
}


def __getattr__(name: str):
    """Import model classes only when they are requested."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
