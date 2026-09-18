"""在訓練集上、用 EMA 權重、固定隨機抽樣做 eval

背景:原始碼的 `train/exact_accuracy` 是「原始(未 EMA)權重」在「單一訓練 batch」上算的,
`all.exact_accuracy` 是「EMA 權重」在「完整測試集」上算的——兩者不對等,不能直接比較來
判斷是否 overfitting。本檔補一個公平的比較對象:用同一套 EMA 權重、在訓練集的固定隨機
子集上跑 eval,方法與測試集 eval 完全一致,只是資料換成訓練集。

抽樣理由:訓練集含 1000 倍資料增強、約 100 萬筆,若全部評估會讓每個 run 的 eval 時間大增。
改為固定隨機種子抽樣 50,000 筆,標準誤差 < 0.2 個百分點(見對話記錄的統計估算),
足以分辨深度之間的差異(通常達數個百分點),且所有深度點與機制組共用同一組樣本,
確保彼此可比。

不呼叫 pretrain.evaluate():pretrain_instrumented.py 會把 pretrain.evaluate 替換成自己的
版本,若這裡再呼叫 pretrain.evaluate 會遞迴呼叫到替換後的版本,造成無窮遞迴。改為自行
實作精簡版推論迴圈(邏輯逐行對照 pretrain.py 的 evaluate(),只保留本檔需要的部分:
不含 evaluators、不存 save_preds)。

兩種用法:
1. 獨立腳本:補算已完成 run 的訓練集準確率(載入存好的 checkpoint,只做推論)
     python eval_train_sample.py --config-name cfg_sudoku arch=trm_gc_sudoku \\
       arch.L_cycles=6 global_batch_size=256 \\
       load_checkpoint="checkpoints/TRM-G-Sudoku/A1_TRMGC_n6_bs256_v2/step_XXXXX"

2. 供 pretrain_instrumented.py 的 evaluate() 覆寫呼叫,在每次正式 eval 時一併計算
   (使用當次已經完成 EMA 切換的模型),見該檔案的 evaluate_on_train_sample() 呼叫。
"""
import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

import pretrain
from pretrain import PretrainConfig, TrainState, create_model, load_synced_config
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")

# 固定種子與樣本數:所有深度點、所有機制比較組都用同一組訓練集樣本,確保互相可比。
SAMPLE_SEED = 42
SAMPLE_SIZE = 50_000


def _iter_train_sample(dataset: PuzzleDataset, sample_size: int, seed: int, global_batch_size: int):
    """仿照 PuzzleDataset._iter_test() 的輸出格式(set_name, batch, batch_size),
    但只走固定隨機抽樣出來的 sample_size 筆,不走全部訓練集。

    Sudoku-Extreme 的 puzzle_identifiers 全部固定為 0(見 dataset/build_sudoku_dataset.py:109,
    `results["puzzle_identifiers"].append(0)`),抽樣時不需要處理 puzzle 邊界對應問題,
    直接補 0 即可,不用像 _iter_test() 那樣用 puzzle_indices 做 searchsorted 查找。
    """
    dataset._lazy_load_dataset()
    rng = np.random.default_rng(seed)

    for set_name, data in dataset._data.items():
        n_examples = len(data["inputs"])
        n_sample = min(sample_size, n_examples)
        idx = rng.choice(n_examples, size=n_sample, replace=False)
        idx.sort()

        for start in range(0, len(idx), global_batch_size):
            batch_idx = idx[start:start + global_batch_size]
            batch = dataset._collate_batch({
                "inputs": data["inputs"][batch_idx],
                "labels": data["labels"][batch_idx],
                "puzzle_identifiers": np.zeros(len(batch_idx), dtype=np.int32),
            })
            yield set_name, batch, len(batch_idx)


def _run_inference_and_aggregate(model, loader, rank: int):
    """精簡版推論迴圈,邏輯對照 pretrain.py 的 evaluate():
    跑到 ACT all_finish、把每個 batch 的 metrics 加總,最後除以 count 正規化。
    不含 evaluators、不存 save_preds——這兩者本檔用不到。
    """
    metric_keys = None
    metric_sum = None
    processed = 0

    with torch.inference_mode():
        for set_name, batch, _ in loader:
            processed += 1
            if rank == 0 and processed % 20 == 0:
                print(f"  [train sample eval] batch {processed}", flush=True)

            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = model.initial_carry(batch)

            while True:
                carry, loss, metrics, preds, all_finish = model(carry=carry, batch=batch, return_keys=[])
                if all_finish:
                    break

            if metric_keys is None:
                metric_keys = sorted(metrics.keys())
                metric_sum = torch.zeros(len(metric_keys), dtype=torch.float32, device="cuda")
            metric_sum += torch.stack([metrics[k] for k in metric_keys])

            del carry, loss, preds, metrics, batch, all_finish

    if metric_sum is None:
        return None

    values = metric_sum.cpu().numpy()
    result = {k: float(values[i]) for i, k in enumerate(metric_keys)}
    count = max(result.pop("count"), 1)
    return {k: v / count for k, v in result.items()}


def evaluate_on_train_sample(model, config: PretrainConfig, rank: int, world_size: int,
                              sample_size: int = SAMPLE_SIZE, sample_seed: int = SAMPLE_SEED):
    """在訓練集的固定隨機子集上評估,回傳 {"exact_accuracy":..., "accuracy":..., "n_sampled":...}。
    model 應已是 EMA 權重(訓練中呼叫時傳入已完成 EMA 切換的模型;獨立腳本模式下,
    checkpoint 存的本來就是 EMA 權重,create_model() 載入後不需額外處理)。
    """
    ds_config = PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths,
        global_batch_size=config.global_batch_size,
        test_set_mode=True,   # 只借用其 padding/collate 邏輯,實際走訪由 _iter_train_sample 控制
        epochs_per_iter=1,
        rank=rank,
        num_replicas=world_size,
    )
    dataset = PuzzleDataset(ds_config, split="train")

    # 提前載入,才能從實際陣列長度算出真正的抽樣數(total_groups 是基礎題目數,
    # 不是資料增強後的總筆數——之前的版本誤用 total_groups × mean_puzzle_examples 算出
    # 1000,但實際抽樣邏輯用的是正確的 len(data["inputs"])≈100萬筆,抽樣本身沒有問題,
    # 只有這裡回報的數字算錯)
    dataset._lazy_load_dataset()
    n_examples = sum(len(d["inputs"]) for d in dataset._data.values())

    was_training = model.training
    model.eval()
    loader = _iter_train_sample(dataset, sample_size, sample_seed, config.global_batch_size)
    result = _run_inference_and_aggregate(model, loader, rank)
    if was_training:
        model.train()

    if result is None:
        return None
    result["n_sampled"] = min(sample_size, n_examples)
    return result


# ---------------------------------------------------------------------------
# 獨立腳本模式:載入已存的 checkpoint,補算已完成 run 的訓練集準確率
# ---------------------------------------------------------------------------
@hydra.main(config_path=_CONFIG_DIR, config_name="cfg_pretrain", version_base=None)
def main(hydra_config: DictConfig):
    config = load_synced_config(hydra_config, rank=0, world_size=1)
    assert config.load_checkpoint is not None, (
        "必須指定 load_checkpoint=<存好的 checkpoint 路徑>,例如:\n"
        '  load_checkpoint="checkpoints/TRM-G-Sudoku/A1_TRMGC_n6_bs256_v2/step_195312"'
    )

    # 只需要 metadata(vocab_size、seq_len 等)來建模型,不需要真的建立 DataLoader
    meta_probe = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed, dataset_paths=config.data_paths,
        global_batch_size=config.global_batch_size, test_set_mode=True,
        epochs_per_iter=1, rank=0, num_replicas=1,
    ), split="train")

    model, _, _ = create_model(config, meta_probe.metadata, rank=0, world_size=1)
    model.eval()

    print(f"\n[訓練集抽樣評估] run={config.run_name}, n(L_cycles)={config.arch.L_cycles}, "
          f"checkpoint={config.load_checkpoint}")
    result = evaluate_on_train_sample(model, config, rank=0, world_size=1)

    print(f"\n===== 結果 =====")
    print(f"抽樣數:          {result['n_sampled']}")
    print(f"train_sample exact_accuracy: {result['exact_accuracy']:.4f}")
    print(f"train_sample accuracy:       {result['accuracy']:.4f}")


if __name__ == "__main__":
    main()
