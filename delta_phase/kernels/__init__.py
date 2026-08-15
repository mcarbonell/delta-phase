"""
delta_phase.kernels
===================
High-performance fused GPU Triton kernels and optimized PyTorch fallbacks for DeltaPhase.
"""

from .triton_chunk_delta import (
    delta_phase_chunkwise_fused,
    triton_available,
    DeltaPhaseTritonFunction
)

__all__ = [
    'delta_phase_chunkwise_fused',
    'triton_available',
    'DeltaPhaseTritonFunction'
]
