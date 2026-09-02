"""TRM-G 完整版:TRM + GC + Damped Residual + Learnable Gating

對應實驗:C5(小規模驗證)、D9/D10/D11(完整訓練與極限測試)

兩機制的組合方式:先做 Gating 融合得到候選狀態,再做阻尼更新——
  net_out   = Net(x, y_t, z_t)
  g_t       = σ(W_g · [z_t, net_out])
  candidate = g_t ⊙ z_t + (1 - g_t) ⊙ net_out
  z_{t+1}   = (1-α) · z_t + α · candidate
α = 1.0 時退化為純 Gating(C4/D7),gate_init_bias 很負時趨近純 Damped。
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
    TinyRecursiveReasoningModel_ACTV1,
)
from models.recursive_reasoning.trm_gc import split_segments
from models.recursive_reasoning.trm_gc_gating import (
    TRM_GC_Gating_Config,
    TRM_GC_Gating_Inner,
)


class TRM_GC_DampedGating_Config(TRM_GC_Gating_Config):
    alpha: float = 0.7


class TRM_GC_DampedGating_Inner(TRM_GC_Gating_Inner):
    # __init__ 繼承自 Gating 版本(含 gate_L 與 bias 初始化)

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch):
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        alpha = self.config.alpha

        def damped_gated_step(z, injection):
            net_out = self.L_level(z, injection, **seq_info)
            g = torch.sigmoid(self.gate_L(torch.cat([z, net_out], dim=-1)))
            if not self.training:
                self._gate_log.append(g.mean().detach())
            candidate = g * z + (1.0 - g) * net_out
            return (1.0 - alpha) * z + alpha * candidate

        def run_zL(z, injection, k: int):
            for _ in range(k):
                z = damped_gated_step(z, injection)
            return z

        if not self.training:
            self._gate_log = []

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


class TRM_GC_DampedGating(TinyRecursiveReasoningModel_ACTV1):
    def __init__(self, config_dict: dict):
        nn.Module.__init__(self)
        self.config = TRM_GC_DampedGating_Config(**config_dict)
        self.inner = TRM_GC_DampedGating_Inner(self.config)
