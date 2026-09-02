"""TRM + GC + Learnable Gating

對應實驗:C4(小規模驗證)、D7/D8(完整訓練,計畫書比較組 4「Ours」)

計畫書 4.1 公式:
  net_out = Net(x, y_t, z_t)                     ← 只前向一次
  g_t     = σ(W_g · [z_t, net_out])
  z_{t+1} = g_t ⊙ z_t + (1 - g_t) ⊙ net_out
g ≈ 1 → 保留舊狀態(梯度直通,Gradient Highway)
g ≈ 0 → 完全接受新輸出(等同原始 TRM)

gate bias 初始化為 gate_init_bias(預設 -2.0,g ≈ 0.12),
讓訓練起點行為接近原始 TRM,再由模型自行學習調節。
評估模式下(model.eval())會把每步的 g 平均值記錄到 inner._gate_log,
供之後視覺化 gate 行為(計畫書 5.2 質性分析)。
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.layers import CastedLinear
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
    TinyRecursiveReasoningModel_ACTV1_Inner,
    TinyRecursiveReasoningModel_ACTV1,
)
from models.recursive_reasoning.trm_gc import TRM_GC_Config, split_segments


class TRM_GC_Gating_Config(TRM_GC_Config):
    gate_init_bias: float = -2.0


class TRM_GC_Gating_Inner(TinyRecursiveReasoningModel_ACTV1_Inner):
    def __init__(self, config: TRM_GC_Gating_Config):
        super().__init__(config)
        # W_g:輸入 [z_t, net_out] 拼接(2·hidden),輸出逐維 gate(hidden)
        self.gate_L = CastedLinear(config.hidden_size * 2, config.hidden_size, bias=True)
        with torch.no_grad():
            self.gate_L.bias.fill_(config.gate_init_bias)

        self._gate_log = []

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch):
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        def gated_step(z, injection):
            net_out = self.L_level(z, injection, **seq_info)
            g = torch.sigmoid(self.gate_L(torch.cat([z, net_out], dim=-1)))
            if not self.training:
                self._gate_log.append(g.mean().detach())
            return g * z + (1.0 - g) * net_out

        def run_zL(z, injection, k: int):
            for _ in range(k):
                z = gated_step(z, injection)
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


class TRM_GC_Gating(TinyRecursiveReasoningModel_ACTV1):
    def __init__(self, config_dict: dict):
        nn.Module.__init__(self)
        self.config = TRM_GC_Gating_Config(**config_dict)
        self.inner = TRM_GC_Gating_Inner(self.config)
