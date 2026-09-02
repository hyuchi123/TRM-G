"""帶量測記錄的訓練入口(不修改 pretrain.py)

以 monkey-patch 方式只覆蓋 pretrain.train_batch 與 pretrain.evaluate,
其餘(dataloader、optimizer、EMA、checkpoint、wandb)完全沿用原始碼。

新增的 WandB 記錄:
  train/grad_norm : 反向傳播後、optimizer step 前的全模型梯度 L2 範數
                    (對應計畫書評估指標 3:驗證梯度消失是否緩解)
  train/z_L_var   : 潛在狀態 z_L 的變異數(偵測數值震盪 / 發散 / 坍塌)
  train/z_H_var   : 答案狀態 z_H 的變異數(TRM 雙狀態才有,SRM 自動略過)
  train/vram_peak_gb     : torch.cuda.max_memory_allocated(),自訓練開始的累計峰值
                           (對應 Experiments.md 評估指標「VRAM 使用量」,驗證 GC 效果)
  train/vram_reserved_gb : torch.cuda.max_memory_reserved(),含 allocator 保留量,
                           與 nvidia-smi 顯示的數字較接近
  eval/gate_mean  : Gating 變體在 eval 時的 g 值平均(非 Gating 模型自動略過)
  eval/gate_std   : 同上,標準差

用法:指令與 pretrain.py 完全相同,只把入口換成本檔:
  python pretrain_instrumented.py --config-name cfg_sudoku arch=trm_gc_sudoku ...

環境變數 INSTRUMENT_EVERY 控制訓練端記錄頻率(預設每 50 步一次,設 0 關閉)。
"""
import os

import torch
import torch.distributed as dist

import pretrain
from pretrain import compute_lr

INSTRUMENT_EVERY = int(os.environ.get("INSTRUMENT_EVERY", "50"))


def _unwrap_inner(model):
    """拆掉 torch.compile 與 ACTLossHead 的包裝,拿到最內層的 inner 模型"""
    m = model
    if hasattr(m, "_orig_mod"):   # torch.compile 包裝
        m = m._orig_mod
    m = getattr(m, "model", m)    # ACTLossHead 包裝
    return getattr(m, "inner", None)


def train_batch(config, train_state, batch, global_batch_size, rank, world_size):
    # 複製自 pretrain.train_batch,僅插入「量測」段;其餘逐行相同
    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return

    batch = {k: v.cuda() for k, v in batch.items()}

    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)

    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    ((1 / global_batch_size) * loss).backward()

    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)

    # ---- 量測:必須在 optimizer step 之前,step 後梯度會被 zero_grad 清空 ----
    extra_metrics = {}
    if INSTRUMENT_EVERY and (train_state.step % INSTRUMENT_EVERY == 0):
        with torch.no_grad():
            grad_sq = torch.zeros((), device="cuda", dtype=torch.float32)
            for p in train_state.model.parameters():
                if p.grad is not None:
                    grad_sq += p.grad.float().pow(2).sum()
            extra_metrics["train/grad_norm"] = grad_sq.sqrt().item()

            # carry.inner_carry 是本次 forward 結束後 detach 的最新潛在狀態
            inner_carry = train_state.carry.inner_carry
            if hasattr(inner_carry, "z_L"):
                extra_metrics["train/z_L_var"] = inner_carry.z_L.float().var().item()
            if hasattr(inner_carry, "z_H"):
                extra_metrics["train/z_H_var"] = inner_carry.z_H.float().var().item()

            # VRAM 峰值:max_memory_allocated 是「自程式啟動(或上次 reset)以來」的
            # 單調累計峰值,所以每次記到的都是目前為止的最大值,取最後一筆即為該實驗的 peak。
            _GB = 1024 ** 3
            extra_metrics["train/vram_peak_gb"] = torch.cuda.max_memory_allocated() / _GB
            extra_metrics["train/vram_reserved_gb"] = torch.cuda.max_memory_reserved() / _GB
    # ----------------------------------------------------------------------

    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step

        optim.step()
        optim.zero_grad()

    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}

            count = max(reduced_metrics["count"], 1)
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            reduced_metrics.update(extra_metrics)  # 量測結果併入
            return reduced_metrics


_orig_evaluate = pretrain.evaluate


def evaluate(config, train_state, eval_loader, eval_metadata, evaluators, rank, world_size, cpu_group):
    # 先跑原始 evaluate,結束後補記 Gating 的 g 值統計
    reduced_metrics = _orig_evaluate(
        config, train_state, eval_loader, eval_metadata, evaluators,
        rank=rank, world_size=world_size, cpu_group=cpu_group,
    )

    inner = _unwrap_inner(train_state.model)
    gate_log = getattr(inner, "_gate_log", None) if inner is not None else None
    if gate_log:
        # _gate_log 每次 forward 都會重置,此處拿到的是 eval 最後一個
        # forward(最後一批、最後一個 ACT step)的逐遞迴步 g 平均值快照
        gates = torch.stack([g.float() for g in gate_log])
        if rank == 0:
            if reduced_metrics is None:
                reduced_metrics = {}
            reduced_metrics["eval/gate_mean"] = gates.mean().item()
            reduced_metrics["eval/gate_std"] = gates.std().item()

    return reduced_metrics


# 覆蓋 pretrain 模組內的同名函數;launch() 內部呼叫時會解析到這裡的版本
pretrain.train_batch = train_batch
pretrain.evaluate = evaluate


if __name__ == "__main__":
    try:
        pretrain.launch()
    finally:
        # 不論正常結束或 OOM 崩潰,都把 VRAM 峰值印出來
        # (B 組要記錄 OOM 當下的用量,此時 wandb 可能來不及上傳最後一筆)
        if torch.cuda.is_available():
            _GB = 1024 ** 3
            print(
                f"\n[VRAM] peak allocated = {torch.cuda.max_memory_allocated() / _GB:.2f} GB, "
                f"peak reserved = {torch.cuda.max_memory_reserved() / _GB:.2f} GB"
            )
