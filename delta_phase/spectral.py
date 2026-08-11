import math
import torch

def create_hadamard_matrix(n: int) -> torch.Tensor:
    """Genera la matriz ortogonal discreta binaria de Fast Walsh-Hadamard (±1 / sqrt(n))"""
    H = torch.tensor([[1.0]], dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)
    return H / math.sqrt(n)

def create_dct2_matrix(n: int) -> torch.Tensor:
    """Genera la matriz ortogonal armónica de cosenos continuos Discrete Cosine Transform (DCT-II)"""
    C = torch.zeros((n, n), dtype=torch.float32)
    for k in range(n):
        for i in range(n):
            if k == 0:
                C[k, i] = 1.0 / math.sqrt(n)
            else:
                C[k, i] = math.sqrt(2.0 / n) * math.cos(math.pi * k * (2 * i + 1) / (2.0 * n))
    return C

def create_haar_matrix(n: int) -> torch.Tensor:
    """Genera la matriz ortogonal de ondículas multi-resolución Discrete Wavelet Transform (DWT Haar)"""
    if n == 1:
        return torch.tensor([[1.0]], dtype=torch.float32)
    H_sub = create_haar_matrix(n // 2)
    low = torch.cat([H_sub, H_sub], dim=1) / math.sqrt(2)
    high = torch.zeros((n // 2, n), dtype=torch.float32)
    for i in range(n // 2):
        high[i, 2 * i] = 1.0 / math.sqrt(2)
        high[i, 2 * i + 1] = -1.0 / math.sqrt(2)
    return torch.cat([low, high], dim=0)
