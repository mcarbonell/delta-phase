from .model import DeltaPhaseConfig, DeltaPhaseModel
from .layers import DeltaPhaseHolographicBlock, LearnableSubstrateLerpFFN, ShortCausalConv1D, LogicPhaseCore
from .spectral import create_hadamard_matrix, create_dct2_matrix, create_haar_matrix

__version__ = "1.1.0"
__all__ = [
    "DeltaPhaseConfig",
    "DeltaPhaseModel",
    "DeltaPhaseHolographicBlock",
    "LearnableSubstrateLerpFFN",
    "ShortCausalConv1D",
    "LogicPhaseCore",
    "create_hadamard_matrix",
    "create_dct2_matrix",
    "create_haar_matrix",
]
