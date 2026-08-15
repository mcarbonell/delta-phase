from .model import DeltaPhaseConfig, DeltaPhaseModel
from .layers import DeltaPhaseHolographicBlock, LearnableSubstrateLerpFFN, ShortCausalConv1D, LogicPhaseCore, LaplacePhaseCore
from .spectral import create_hadamard_matrix, create_dct2_matrix, create_haar_matrix
from .kernels import delta_phase_chunkwise_fused, triton_available

__version__ = "1.3.0"
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
    "delta_phase_chunkwise_fused",
    "triton_available",
]

