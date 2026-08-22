import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple
from .layers import DeltaPhaseHolographicBlock

@dataclass
class DeltaPhaseConfig:
    dim: int = 512
    emb_dim: int = 128
    n_layers: int = 6
    n_heads: int = 8
    vocab_size: int = 32768
    max_seq_len: int = 2048
    chunk_size: int = 128
    conv_kernel_size: int = 4
    num_banks: int = 4
    weight_tying: bool = True
    beta_mode: str = "learned"  # "learned": data-dependent beta_t | "fixed": beta_t = 1.0 (control arm)

class DeltaPhaseModel(nn.Module):
    def __init__(self, config: DeltaPhaseConfig):
        super().__init__()
        self.config = config
        
        if config.emb_dim and config.emb_dim > 0:
            self.embed = nn.Embedding(config.vocab_size, config.emb_dim)
            self.embed_proj = nn.Linear(config.emb_dim, config.dim, bias=False)
            self.use_factorized = True
        else:
            self.embed = nn.Embedding(config.vocab_size, config.dim)
            self.embed_proj = None
            self.use_factorized = False
            
        self.blocks = nn.ModuleList([
            DeltaPhaseHolographicBlock(
                d_model=config.dim,
                n_heads=config.n_heads,
                conv_kernel_size=config.conv_kernel_size,
                chunk_size=config.chunk_size,
                num_banks=config.num_banks,
                beta_mode=config.beta_mode
            ) for _ in range(config.n_layers)
        ])
        
        self.norm_f = nn.LayerNorm(config.dim)
        
        if self.use_factorized:
            self.head_proj = nn.Linear(config.dim, config.emb_dim, bias=False)
            self.head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
        else:
            self.head_proj = None
            self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
            
        if config.weight_tying:
            self.head.weight = self.embed.weight
            
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, states: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """Full sequence parallel forward pass"""
        if self.use_factorized:
            e = self.embed(input_ids)
            h = self.embed_proj(e)
        else:
            h = self.embed(input_ids)
            
        if states is None:
            states = [None] * len(self.blocks)
            
        for i, block in enumerate(self.blocks):
            h, _ = block(h, states[i])
            
        h_norm = self.norm_f(h)
        if self.head_proj is not None:
            h_out = self.head_proj(h_norm)
        else:
            h_out = h_norm
            
        logits = self.head(h_out)
        return logits

    def step(self, input_id_t: torch.Tensor, states: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Streaming single-token O(1) step during autoregressive decoding"""
        if self.use_factorized:
            e = self.embed(input_id_t)
            h = self.embed_proj(e)
        else:
            h = self.embed(input_id_t)
            
        if states is None:
            states = [None] * len(self.blocks)
            
        new_states = []
        for i, block in enumerate(self.blocks):
            h, next_s = block.step(h, states[i])
            new_states.append(next_s)
            
        h_norm = self.norm_f(h)
        if self.head_proj is not None:
            h_out = self.head_proj(h_norm)
        else:
            h_out = h_norm
            
        logits = self.head(h_out)
        return logits, new_states

    def print_substrate_report(self):
        """Imprime el reporte transparente de sustratos espectrales sintonizados"""
        print("\n" + "="*85)
        print("REPORTE TRANSPARENTE DE SUSTRATOS ESPECTRALES ELEGIDOS (DELTAPHASE LERP ROUTER)")
        print("="*85)
        print(f"{'Capa':<15} | {'% FWHT (Binario)':<20} | {'% DCT-II (Cosenos)':<20} | {'% DWT Haar (Ondículas)':<22}")
        print("-" * 85)
        for idx, block in enumerate(self.blocks):
            p_fwht, p_dct, p_haar = block.ffn.get_substrate_probabilities()
            print(f"Capa {idx+1:<10} | {p_fwht*100:<20.2f}% | {p_dct*100:<20.2f}% | {p_haar*100:<22.2f}%")
        print("="*85)
