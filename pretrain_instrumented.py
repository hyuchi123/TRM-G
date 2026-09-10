"""帶量測記錄的訓練入口(不修改 pretrain.py)

以 monkey-patch 方式只覆蓋 pretrain.train_batch 與 pretrain.evaluate,
其餘(dataloader、optimizer、EMA、checkpoint、wandb)完全沿用原始碼。

新增的 WandB 記錄:
  train/grad_norm : 反向傳播後、optimizer step 前的全模型梯度 L2 範數
                    (對應計畫書評估指標 3:驗證梯度消失是否緩解)
  train/z_L_var   : 潛在狀態 z_L 的變異數
  train/z_H_var   : 答案狀態 z_H 的變異數(TRM 雙狀態才有,SRM 自動略過)
                    ⚠️ 這兩項因架構每步 rms_norm 而恆等於 1.0(實測 0.9975~1.0000),
                       無法診斷數值震盪。保留僅為記錄完整性,實際診斷請看下面的 z_delta。
  eval/z_delta_mean / _max / _last :
                    潛在狀態逐步位移量 Δ_t = ||z_{t+1} − z_t|| / ||z_t||,
                    取代變異數作為「數值震盪」的主要診斷指標。
                    對應計畫書評估指標「Latent z 變異數」與 5.2-2「推論軌跡平滑化」。
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

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig

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
    # ---- 隔離量測「單次 eval 本身」的 VRAM 峰值 ----
    # 背景:A1-repro 與 A3 兩次崩潰都發生在 eval 批次迴圈中間,且 train/vram_peak_gb
    # (自訓練開始的累計峰值)在兩次崩潰的 run 都出現遠高於短測(跳過 eval)的數字
    # (例如 A3 的 n=8 記到 17.97GB,遠高於 n=16 含 eval 的短測 7.97GB)。
    # 懷疑來源:EMA 切換時 `copy.deepcopy(train_state)` + `ema_helper.ema_copy(...)`
    # (pretrain.py launch() 內,SWITCH TO EMA 那段)會短暫疊加模型/optimizer 記憶體,
    # 但那段不在本檔案的攔截範圍內,故改用此處在 eval 迴圈本身起訖時重設/量測峰值,
    # 隔離出「單純跑完整個 eval batch 迴圈」實際需要多少 VRAM。
    # ⚠️ 副作用:此重設之後,train/vram_peak_gb 不再是「自訓練開始」的累計峰值,
    # 而變成「自上次 eval 以來」的峰值——對長 run 而言此資訊反而更有用(能看出
    # 是哪個 iter 的峰值特別高),故保留此副作用,不另外復原。
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 先跑原始 evaluate,結束後補記 Gating 的 g 值統計
    reduced_metrics = _orig_evaluate(
        config, train_state, eval_loader, eval_metadata, evaluators,
        rank=rank, world_size=world_size, cpu_group=cpu_group,
    )

    if torch.cuda.is_available() and rank == 0:
        _GB = 1024 ** 3
        eval_peak = torch.cuda.max_memory_allocated() / _GB
        eval_reserved = torch.cuda.max_memory_reserved() / _GB
        print(f"[EVAL VRAM] peak allocated = {eval_peak:.2f} GB, peak reserved = {eval_reserved:.2f} GB", flush=True)
        if reduced_metrics is None:
            reduced_metrics = {}
        reduced_metrics["eval/vram_peak_gb"] = eval_peak
        reduced_metrics["eval/vram_reserved_gb"] = eval_reserved

    inner = _unwrap_inner(train_state.model)

    # 潛在狀態逐步位移量 Δ_t = ||z_{t+1} − z_t|| / ||z_t||(數值震盪的主要診斷指標)
    # 取代 z_L_var——後者因架構每步 rms_norm 而恆等於 1,無法區分穩定與震盪。
    # 詳見 models/recursive_reasoning/trm_gc.py 的說明。
    z_delta_log = getattr(inner, "_z_delta_log", None) if inner is not None else None
    if z_delta_log and rank == 0:
        deltas = torch.stack([d.float() for d in z_delta_log])
        if reduced_metrics is None:
            reduced_metrics = {}
        reduced_metrics["eval/z_delta_mean"] = deltas.mean().item()
        reduced_metrics["eval/z_delta_max"] = deltas.max().item()
        reduced_metrics["eval/z_delta_last"] = deltas[-1].item()

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


# Hydra 的 config_path 是相對於「定義 @hydra.main 的那個模組」解析的。
# 直接呼叫 pretrain.launch() 時,該函數的 __module__ 是 "pretrain" 而非 "__main__",
# Hydra 會改用 Python 套件的方式去找名為 config 的模組,報:
#   Primary config module 'config' not found. Check that it's ... contains an __init__.py
# 解法:在本檔(即 __main__)重新宣告一次 @hydra.main,參數與 pretrain.py 完全相同,
# 讓 Hydra 以本檔所在目錄為基準找到 config/,再把解析好的 config 交給原始 launch 執行。
# 命令列的 --config-name cfg_sudoku 仍會正常覆蓋這裡的預設值。
_orig_launch = getattr(pretrain.launch, "__wrapped__", None)

# 用絕對路徑,不依賴 Hydra 去推測「呼叫端檔案在哪」——那個推測正是原本出錯的環節。
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")


@hydra.main(config_path=_CONFIG_DIR, config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    if _orig_launch is not None:
        # functools.wraps 保留的未裝飾原函數,直接執行其內容
        return _orig_launch(hydra_config)
    # 後備路徑:Hydra 的 cfg_passthrough——傳入既有 config 時會跳過 Hydra 直接執行
    return pretrain.launch(hydra_config)


if __name__ == "__main__":
    try:
        launch()
    finally:
        # 不論正常結束或 OOM 崩潰,都把 VRAM 峰值印出來
        # (B 組要記錄 OOM 當下的用量,此時 wandb 可能來不及上傳最後一筆)
        if torch.cuda.is_available():
            _GB = 1024 ** 3
            print(
                f"\n[VRAM] peak allocated = {torch.cuda.max_memory_allocated() / _GB:.2f} GB, "
                f"peak reserved = {torch.cuda.max_memory_reserved() / _GB:.2f} GB"
            )
