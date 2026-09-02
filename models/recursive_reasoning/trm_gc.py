"""TRM + Segmented Gradient Checkpointing

對應實驗:B2-B5(GC 壓力測試)、D1/D2(深層 TRM 無穩定機制基準)

不修改原始 trm.py,以繼承方式只覆寫 forward。
遞迴深度 n 由 arch.L_cycles 控制(命令列覆蓋)。
分段數由 arch.gc_segments 控制(計畫書 4.2 的分段策略):
  gc_segments = 0  →  自動取 √n 段
  gc_segments = 1  →  整輪一段(儲存邊界最少、重算最多)
  gc_segments = n  →  每步一段
"""
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1_Inner,
    TinyRecursiveReasoningModel_ACTV1,
)


class TRM_GC_Config(TinyRecursiveReasoningModel_ACTV1Config):
    gc_segments: int = 0  # 0 = 自動 √n


def split_segments(n: int, gc_segments: int):
    """把 n 步切成若干段,回傳每段步數,例如 n=16 → [4, 4, 4, 4]"""
    num = int(round(math.sqrt(n))) if gc_segments <= 0 else max(1, min(gc_segments, n))
    base, rem = divmod(n, num)
    return [base + (1 if i < rem else 0) for i in range(num)]


# ---------------------------------------------------------------------------
# 潛在狀態逐步變化量的記錄(數值穩定性診斷)
#
# 為何不用變異數:TRM 的每個 Block 結尾都是 rms_norm(trm.py:96,103),
# z_L 是 L_level 的輸出,每一步都被強制正規化,變異數在結構上被釘在 1.0 附近
# (實測 smoke_bs256:z_L_var 全程 0.9975~1.0000)。
# 因此變異數無法區分「穩定」與「震盪」——正規化把尺度資訊抹掉了。
#
# 改用相對位移量:  Δ_t = ||z_{t+1} − z_t|| / ||z_t||
#   Δ 大 → 狀態每步大幅跳動,即計畫書所指的「數值震盪」
#   Δ 小 → 推論軌跡平滑
# 這個量直接對應 Damped Residual 的作用:z_{t+1} = (1−α)·z_t + α·Net(...)
# 會把位移縮放 α 倍,所以阻尼版本的 Δ 應顯著小於原版——可驗證、可證偽。
# 同時也是計畫書 5.2-2「推論軌跡平滑化」的量化版本。
#
# 與 _gate_log 相同,只在 eval(not self.training)記錄:訓練時帶梯度那輪走
# checkpoint,反向傳播會重算一次,在訓練時記錄會產生重複條目。
# ---------------------------------------------------------------------------

def reset_z_delta_log(module):
    """每次 forward 開頭重置,eval 結束後留存的是最後一個 forward 的軌跡快照"""
    module._z_delta_log = []


def log_z_delta(module, z_old, z_new):
    """記錄一個遞迴步的相對位移量 ||z_new − z_old|| / ||z_old||"""
    log = getattr(module, "_z_delta_log", None)
    if log is None:
        log = module._z_delta_log = []
    denom = z_old.float().norm()
    if denom > 0:
        log.append(((z_new - z_old).float().norm() / denom).detach())


class TRM_GC_Inner(TinyRecursiveReasoningModel_ACTV1_Inner):
    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch):
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        def run_zL(z, injection, k: int):
            for _ in range(k):
                z_new = self.L_level(z, injection, **seq_info)
                if not self.training:
                    log_z_delta(self, z, z_new)
                z = z_new
            return z

        if not self.training:
            reset_z_delta_log(self)

        z_H, z_L = carry.z_H, carry.z_L
        n = self.config.L_cycles

        # 前 H_cycles-1 輪不帶梯度(與原版相同,不保存 activation,無需 checkpoint)
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                z_L = run_zL(z_L, z_H + input_embeddings, n)
                z_H = self.L_level(z_H, z_L, **seq_info)

        # 最後一輪帶梯度:分段 checkpoint,以重算換記憶體
        injection = z_H + input_embeddings
        if torch.is_grad_enabled():
            for k in split_segments(n, self.config.gc_segments):
                z_L = checkpoint(run_zL, z_L, injection, k, use_reentrant=False)
            z_H = checkpoint(run_zL, z_H, z_L, 1, use_reentrant=False)
        else:
            # 評估時不需 checkpoint
            z_L = run_zL(z_L, injection, n)
            z_H = self.L_level(z_H, z_L, **seq_info)

        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TRM_GC(TinyRecursiveReasoningModel_ACTV1):
    def __init__(self, config_dict: dict):
        nn.Module.__init__(self)
        self.config = TRM_GC_Config(**config_dict)
        self.inner = TRM_GC_Inner(self.config)
