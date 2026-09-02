"""TRM + GC + Latent Damped Residual

對應實驗:C1/C2/C3(α 掃描)、D5/D6(完整訓練)

計畫書 4.1 公式:z_{t+1} = (1-α)·z_t + α·Net(x, y_t, z_t),α ∈ (0,1]
  α = 1.0 → 退化回原始 TRM
機制作用於潛在狀態 z(z_L)的每一步更新,包含不帶梯度的前 H_cycles-1 輪
(阻尼是模型動力學的一部分,必須在所有遞迴步生效,不只在帶梯度那輪)。
答案狀態 y(z_H)的更新維持原樣,與計畫書公式一致。
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
    TinyRecursiveReasoningModel_ACTV1_Inner,
    TinyRecursiveReasoningModel_ACTV1,
)
from models.recursive_reasoning.trm_gc import (
    TRM_GC_Config,
    log_z_delta,
    reset_z_delta_log,
    split_segments,
)


class TRM_GC_Damped_Config(TRM_GC_Config):
    alpha: float = 0.7


class TRM_GC_Damped_Inner(TinyRecursiveReasoningModel_ACTV1_Inner):
    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch):
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        alpha = self.config.alpha

        def run_zL(z, injection, k: int):
            for _ in range(k):
                z_new = (1.0 - alpha) * z + alpha * self.L_level(z, injection, **seq_info)
                if not self.training:
                    log_z_delta(self, z, z_new)
                z = z_new
            return z

        if not self.training:
            reset_z_delta_log(self)

        z_H, z_L = carry.z_H, carry.z_L
        n = self.config.L_cycles

        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                z_L = run_zL(z_L, z_H + input_embeddings, n)
                z_H = self.L_level(z_H, z_L, **seq_info)

        injection = z_H + input_embeddings
        if torch.is_grad_enabled():
            for k in split_segments(n, self.config.gc_segments):
                z_L = checkpoint(run_zL, z_L, injection, k, use_reentrant=False)

            def update_zH(z, inj):
                return self.L_level(z, inj, **seq_info)

            z_H = checkpoint(update_zH, z_H, z_L, use_reentrant=False)
        else:
            z_L = run_zL(z_L, injection, n)
            z_H = self.L_level(z_H, z_L, **seq_info)

        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TRM_GC_Damped(TinyRecursiveReasoningModel_ACTV1):
    def __init__(self, config_dict: dict):
        nn.Module.__init__(self)
        self.config = TRM_GC_Damped_Config(**config_dict)
        self.inner = TRM_GC_Damped_Inner(self.config)
