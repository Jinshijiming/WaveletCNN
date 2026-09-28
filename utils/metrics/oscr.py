import torch
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat
from typing import Optional, Tuple
import matplotlib.pyplot as plt


def compute_ccr_fpr_auto(pred_labels: torch.Tensor, targets: torch.Tensor, known_mask: torch.Tensor,
                         score: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算 CCR/FPR
    pred_labels: [N] 预测标签（整数）
    score: [N] 开集分数（越大越像 known）
    targets: [N] 真实标签
    known_mask: [N] 已知/未知掩码
    """

    idx = torch.argsort(score, descending=True)
    score = score[idx]
    pred_labels = pred_labels[idx]
    targets = targets[idx]
    known_mask = known_mask[idx]

    # 前缀累积
    is_known = known_mask.float()
    is_unknown = (~known_mask).float()
    correct = (pred_labels == targets).float() * is_known
    cum_correct_known = torch.cumsum(correct, dim=0)
    cum_fp_unknown = torch.cumsum(is_unknown, dim=0)

    total_known = is_known.sum().float() + 1e-8
    total_unknown = is_unknown.sum().float() + 1e-8

    diff_indices = torch.where(score[1:] != score[:-1])[0]
    thr_idx = torch.nn.functional.pad(diff_indices, [0, 1], value=len(score)-1)

    thresholds = score[thr_idx]
    ccr = cum_correct_known[thr_idx] / total_known
    fpr = cum_fp_unknown[thr_idx] / total_unknown

    return thresholds, ccr, fpr


class OSCR(Metric):
    """
    Open-Set Classification Rate (OSCR) Metric
    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.add_state("known_mask", default=[], dist_reduce_fx="cat")
        self.add_state("score", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, targets: torch.Tensor, known_mask: torch.Tensor,
               score: torch.Tensor):
        """
        preds: [N] 已预测标签（整数）
        score: logits 或概率，用于排序（越大越像 known）
        targets: [N] 真实标签
        known_mask: [N] 已知/未知掩码，或已知/未知类标签(0,1)
        """
        known_mask = known_mask.bool()
        
        if score is None:
            raise ValueError("OSCR.update() 需要提供 'score'（越大越像已知类）")

        if score.ndim > 1:
            # 多分类 logits -> softmax -> max
            if score.max() > 1.0 or score.min() < 0.0:
                score = F.softmax(score, dim=1)
            score = score.max(dim=1)[0]
        else:
            # 二分类 logits -> sigmoid
            if score.max() > 1.0 or score.min() < 0.0:
                score = score.sigmoid()

        self.preds.append(preds)
        self.targets.append(targets)
        self.known_mask.append(known_mask)
        self.score.append(score)

    def compute(self):
        preds = dim_zero_cat(self.preds)
        targets = dim_zero_cat(self.targets)
        known_mask = dim_zero_cat(self.known_mask)
        score = dim_zero_cat(self.score)

        if score.numel() == 0:
            raise ValueError("OSCR.compute() 没有收到任何 score。update() 必须提供 score。")

        _, ccr, fpr = compute_ccr_fpr_auto(preds, targets, known_mask, score)
        return torch.trapz(ccr, fpr)

    def plot(self, curve=None, show_auc=True, ax=None):
        """
        绘制 OSCR 曲线
        curve: (thresholds, ccr, fpr)，None 时自动计算
        show_auc: 是否在标签显示 AUC
        """
        if curve is None:
            preds = dim_zero_cat(self.preds)
            targets = dim_zero_cat(self.targets)
            known_mask = dim_zero_cat(self.known_mask)
            score_tensor = dim_zero_cat(self.score)
            thresholds, ccr, fpr = compute_ccr_fpr_auto(preds, targets, known_mask, score_tensor)
        else:
            thresholds, ccr, fpr = curve

        if show_auc:
            auc = torch.trapz(ccr, fpr).item()
            label = f"OSCR AUC={auc:.4f}"
        else:
            label = "OSCR curve"

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig,ax = (None,ax)

        ax.plot(fpr.cpu(), ccr.cpu(), label=label)
        ax.set_xlabel("FPR (Unknown)")
        ax.set_ylabel("CCR (Known)")
        ax.set_title("OSCR Curve")
        ax.grid(True)
        ax.legend()

        return fig, ax




if __name__ == "__main__":
    # 简单测试
    metric = OSCR()

    preds = torch.tensor([0, 0, 2, 0, 1, 2, 3, 4, 5, 6])
    targets = torch.tensor([0, 1, 2, 0, 1, 2, 99, 99, 99, 99])
    known_mask = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    score = torch.tensor([0.9, 0.8, 0.95, 0.7, 0.6, 0.85, 0.4, 0.3, 0.2, 0.1])

    metric.update(preds, targets, known_mask, score)
    oscr_value = metric.compute()
    print(f"OSCR: {oscr_value:.4f}")

    fig, ax = metric.plot()
    plt.show()
