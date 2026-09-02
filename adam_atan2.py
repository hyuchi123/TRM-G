"""adam_atan2 相容墊片(shim)—— Windows 專用

原始 repo 依賴 PyPI 上的 `adam-atan2`(CUDA extension),官方明示僅支援 Unix,
Windows 無法安裝。本檔以純 PyTorch 的 `adam-atan2-pytorch`(lucidrains)頂替,
**檔名刻意取為 `adam_atan2`,使 `pretrain.py:20` 的

    from adam_atan2 import AdamATan2

原封不動即可解析到本檔**,維持「原始程式碼一行都不改」的原則。

安裝:  pip install adam-atan2-pytorch

---------------------------------------------------------------------------
為何不能直接 `from adam_atan2_pytorch import AdamAtan2 as AdamATan2`
---------------------------------------------------------------------------
lucidrains 的實作有兩處與 `pretrain.py` 的用法不相容:

1. `assert lr > 0.` 在 __init__ 內。
   但 pretrain.py 建構 optimizer 時刻意傳 `lr=0`(見 pretrain.py:149-153),
   真正的 lr 由 scheduler 在每個 step 前寫進 param_group(pretrain.py:82-83)。
   直接 alias 會在建構當下就 AssertionError。
   → 本檔以正數 placeholder 建構,建構後再把 param_group['lr'] 還原成原值。
     由於 lr 每步都被 scheduler 覆寫,placeholder 不影響任何一次更新。

2. `decoupled_wd` 的語意與直覺相反,且會與 `weight_decay=1.0` 產生災難性交互作用。
   lucidrains 的 `decoupled_wd=True` 意思是「把 wd 從 lr 中解耦」,
   實作為在 __init__ 內 `group['weight_decay'] /= lr`,
   step 時再 `p.mul_(1 - lr * wd)`。
   若在此設 True:__init__ 的 lr 是 placeholder(≠ 論文的 1e-4),
   wd 會被除以一個錯誤的基準;而當 scheduler 把 lr 調到 1e-4 時,
   有效衰減量變成 `1e-4 * (1.0 / placeholder)`,可能直接把權重歸零。
   → 本檔強制 `decoupled_wd=False`,此時 step 為 `p.mul_(1 - lr * wd)`,
     即標準 AdamW 式的解耦權重衰減:lr=1e-4、wd=1.0 → 每步乘 0.9999,
     這才是 `config/cfg_sudoku.yaml` 的 `weight_decay: 1.0` 所預期的語意。

     (命名很容易誤導:我們要的「AdamW 行為」對應的是 decoupled_wd=False。)

---------------------------------------------------------------------------
數值等價性
---------------------------------------------------------------------------
更新規則兩邊一致:
    update = a * atan2(m_hat, b * sqrt(v_hat)),  預設 a = 1.27, b = 1.0
lucidrains 版與原版 CUDA kernel 使用相同的預設常數,差別僅在 CUDA 融合實作
vs. 純 PyTorch(本模型僅約 5M 參數,速度差異可忽略)。

**此等價性由實驗 A1 驗證**:若 TRM n=6 能重現論文的 ~87.4%,
即證明本墊片在數值上與原版等價。詳見 MODIFICATIONS.md「七、Windows 環境調整」。

---------------------------------------------------------------------------
⚠️ 移植到 Linux / WSL 時
---------------------------------------------------------------------------
本檔會**遮蔽**(shadow)同名的真實套件。若日後在 Linux 上裝了原版 `adam-atan2`,
請刪除或改名本檔,否則仍會走墊片。本檔在偵測到原版已安裝時會印出警告。
"""
import warnings

from adam_atan2_pytorch import AdamAtan2 as _AdamAtan2

__all__ = ["AdamATan2"]

# 建構期用的 lr placeholder,僅為通過 `assert lr > 0.`;
# 每個 optimizer.step() 之前 scheduler 都會覆寫 param_group['lr'],故不影響更新。
_LR_PLACEHOLDER = 1e-4


def _warn_if_real_package_installed():
    """本檔遮蔽了真實套件時提出警告(移植到 Linux 後最容易踩的坑)"""
    try:
        from importlib.metadata import version

        real = version("adam-atan2")
    except Exception:
        return
    warnings.warn(
        f"偵測到已安裝原版 adam-atan2 (v{real}),但它被 repo 根目錄的 "
        f"adam_atan2.py 墊片遮蔽了。若要改用原版 CUDA 實作,請刪除或改名此墊片。",
        RuntimeWarning,
        stacklevel=2,
    )


class AdamATan2(_AdamAtan2):
    """介面相容於原版 CUDA `adam_atan2.AdamATan2`,內部用純 PyTorch 實作。"""

    def __init__(self, params, lr=0.0, betas=(0.9, 0.99), weight_decay=0.0, **kwargs):
        _warn_if_real_package_installed()

        # 見檔頭第 2 點:必須為 False 才是標準 AdamW 式解耦權重衰減
        if kwargs.get("decoupled_wd", False):
            raise ValueError(
                "decoupled_wd=True 會把 weight_decay 除以建構期的 lr placeholder,"
                "與 cfg_sudoku.yaml 的 weight_decay=1.0 產生錯誤的衰減量。"
                "請維持 decoupled_wd=False(標準 AdamW 語意)。"
            )
        kwargs["decoupled_wd"] = False

        requested_lr = lr
        super().__init__(
            params,
            lr=lr if lr > 0 else _LR_PLACEHOLDER,
            betas=betas,
            weight_decay=weight_decay,
            **kwargs,
        )

        # 還原呼叫端原本要求的 lr(pretrain.py 傳 0,由 scheduler 每步接管)
        for group in self.param_groups:
            group["lr"] = requested_lr

        print(
            f"[adam_atan2 shim] 使用純 PyTorch AdamAtan2 "
            f"(decoupled_wd=False, betas={betas}, weight_decay={weight_decay})"
        )
