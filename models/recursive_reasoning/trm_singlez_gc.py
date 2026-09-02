"""SRM(single-z)+ Segmented Gradient Checkpointing

對應實驗:B6(SRM 深層可行性)、D3/D4(深層靜態連接基準,計畫書比較組 2)

繼承自 trm_singlez.py(SRM:單一狀態 + 靜態 skip connection),
只在帶梯度的最後一輪遞迴加入分段 checkpoint,遞迴規則不變。
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.recursive_reasoning.trm_singlez import (
    TinyRecursiveReasoningModel_ACTV1InnerCarry as SingleZ_InnerCarry,
    TinyRecursiveReasoningModel_ACTV1Config as SingleZ_Config,
    TinyRecursiveReasoningModel_ACTV1_Inner as SingleZ_Inner,
    TinyRecursiveReasoningModel_ACTV1 as SingleZ_ACT,
)
from models.recursive_reasoning.trm_gc import split_segments


class SRM_GC_Config(SingleZ_Config):
    gc_segments: int = 0


class SRM_GC_Inner(SingleZ_Inner):
    def forward(self, carry: SingleZ_InnerCarry, batch):
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # 原版 SRM 每輪:n 步帶 input 注入 + 1 步不帶注入
        def run_injected(z, inj, k: int):
            for _ in range(k):
                z = self.L_level(z + inj, **seq_info)
            return z

        def run_plain(z):
            return self.L_level(z, **seq_info)

        z_L = carry.z_L
        n = self.config.L_cycles

        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                z_L = run_injected(z_L, input_embeddings, n)
                z_L = run_plain(z_L)

        if torch.is_grad_enabled():
            for k in split_segments(n, self.config.gc_segments):
                z_L = checkpoint(run_injected, z_L, input_embeddings, k, use_reentrant=False)
            z_L = checkpoint(run_plain, z_L, use_reentrant=False)
        else:
            z_L = run_injected(z_L, input_embeddings, n)
            z_L = run_plain(z_L)

        z_out = z_L
        new_carry = SingleZ_InnerCarry(z_L=z_L.detach())
        output = self.lm_head(z_out)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_out[:, 0]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class SRM_GC(SingleZ_ACT):
    def __init__(self, config_dict: dict):
        nn.Module.__init__(self)
        self.config = SRM_GC_Config(**config_dict)
        self.inner = SRM_GC_Inner(self.config)
