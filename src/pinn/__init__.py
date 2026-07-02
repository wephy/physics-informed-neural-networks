"""Physics-informed neural networks in JAX/Equinox.

A problem is defined by its residual functions and matching samplers (see
:class:`Problem` and the ``sampling`` / ``operators`` helpers). ``train32`` and
``train64`` run the sketched trust-region optimiser in single or double
precision. See ``examples/`` for complete, self-contained problems.
"""

from __future__ import annotations

from . import operators, sampling
from .config import RunConfig
from .networks import MLP, SIREN, GaborNet, SPINN
from .problem import Problem
from .reference import Reference, load_reference, reference_path, save_reference
from .trainer32 import precompile as precompile32
from .trainer32 import train as train32
from .trainer64 import precompile as precompile64
from .trainer64 import train as train64

__all__ = [
    "RunConfig",
    "train32",
    "train64",
    "precompile32",
    "precompile64",
    "MLP",
    "SIREN",
    "GaborNet",
    "SPINN",
    "Problem",
    "operators",
    "sampling",
    "Reference",
    "save_reference",
    "load_reference",
    "reference_path",
]
