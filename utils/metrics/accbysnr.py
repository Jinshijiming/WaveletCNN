import torch
from torchmetrics import Metric
import matplotlib.pyplot as plt


class AccuracyBySNR(Metric):
    def __init__(self, snrs, dist_sync_on_step=False):
        """
        snrs: list of snr
        
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.snr2idx = {snr: i for i, snr in enumerate(snrs)}
        n = len(snrs)
        
        self.register_buffer("snrs", torch.tensor(snrs))
        self.add_state(
            "correct",
            default=torch.zeros(n, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total",
            default=torch.zeros(n, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, snr: torch.Tensor, label: torch.Tensor):
        """
        preds: (N,) 或 (N, num_classes) 的预测 logits/probs
        snr: (N,) 样本对应的 SNR，可以是整数或浮点数
        label: (N,) 真实标签
        """
        device = preds.device

        # 将 preds 转为类别索引
        if preds.ndim > 1:
            preds = preds.argmax(dim=1)
        else:
            preds = preds.long()

        # 计算每个样本的 SNR 与 self.snrs 的差值
        diff = (snr.unsqueeze(1) - self.snrs.unsqueeze(0)).abs()  # shape [N, num_snrs]
        snr_idx = diff.argmin(dim=1)  # 最近的 SNR 对应索引

        # 计算每个样本是否预测正确
        correct_mask = (preds == label).long()

        # 按 snr_idx 累加 correct 和 total
        self.correct.scatter_add_(0, snr_idx, correct_mask)
        self.total.scatter_add_(0, snr_idx, torch.ones_like(snr_idx, dtype=torch.long))

    def compute(self):
        # total > 0 的位置
        mask = self.total > 0

        # 先整体计算 acc（避免除 0）
        acc_all = torch.zeros_like(self.total, dtype=torch.float)
        acc_all[mask] = self.correct[mask] / self.total[mask]

        # 推导式构建字典
        # { expression
        # for target in iterable
        # if condition }
        acc = {
            snr: acc_all[idx].item()
            for snr, idx in self.snr2idx.items()
            if mask[idx]
        }
        return acc

    def plot(self, curve=None,ax=None, **plot_kwargs):
        """
        
        curve:(snrs,accs),None时计算curve
        ax: matplotlib.axes.Axes对象，None时新建
        """
        if curve is not None:
            snrs,accs=curve
        else:
            acc = self.compute()
            if len(acc) == 0:
                raise RuntimeError("No data to plot. Did you call update()?")

        snrs = sorted(acc.keys())
        accs = [acc[s] for s in snrs]

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(snrs, accs, marker='o', **plot_kwargs)
        ax.set_xticks(snrs)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs SNR")
        ax.grid(True)

        return fig,ax
