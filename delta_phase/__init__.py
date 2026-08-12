from .model import DeltaPhaseConfig, DeltaPhaseModel
from .layers import DeltaPhaseHolographicBlock, LearnableSubstrateLerpFFN, ShortCausalConv1D, LogicPhaseCore, LaplacePhaseCore
from .spectral import create_hadamard_matrix, create_dct2_matrix, create_haar_matrix

__version__ = "1.2.0"
__all__ = [
    "DeltaPhaseConfig",
    "DeltaPhaseModel",
    "DeltaPhaseHolographicBlock",
    "LearnableSubstrateLerpFFN",
    "ShortCausalConv1D",
    "LogicPhaseCore",
    "LaplacePhaseCore",
    "create_hadamard_matrix",
    "create_dct2_matrix",
    "create_haar_matrix",
]
