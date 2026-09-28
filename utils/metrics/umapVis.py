import torch
import numpy as np
from torch import Tensor
from torchmetrics import Metric
import umap
import matplotlib.pyplot as plt
from typing import Any, Tuple, Optional

class UmapVis(Metric):
    is_differentiable = False
    higher_is_better = None
    full_state_update = False

    def __init__(
        self, 
        n_components: int = 2, 
        n_neighbors: int = 15,      
        min_dist: float = 0.5,      
        metric: str = 'euclidean',  
        max_samples: int = 20000,   
        random_state: int = 42, 
        umap_kwargs: Optional[dict] = None, 
        **kwargs: Any               
    ) -> None:
        """
        n_components: UMAP 降维后的维度，默认为 2 以便可视化
        n_neighbors: UMAP 的邻居数量，控制局部结构的保留程度
        min_dist: UMAP 中的最小距离参数，控制嵌入空间中点之间的最小距离，影响聚类的紧密程度
        metric: UMAP 使用的距离度量，默认为欧氏距离
        max_samples: 在 update 中允许的最大样本数量，超过后会随机采样以控制内存使用
        random_state: UMAP 的随机种子，确保结果可复现
        umap_kwargs: 其他传递给 UMAP 构造函数的参数，可以
        """
        
        super().__init__(**kwargs)
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.umap_metric = metric
        self.max_samples = max_samples
        self.random_state = random_state
        
        # 优化点 1: 默认加上 init="pca"，大幅提升高维特征降维速度和稳定性
        self.umap_kwargs = {'init': 'pca'}
        if umap_kwargs is not None:
            self.umap_kwargs.update(umap_kwargs)

        self.mapper_ = None

        # 状态依然需要用 Tensor 保存，因为只有 Tensor 才能参与 DDP 多卡的状态同步 (dist_reduce_fx="cat")
        self.add_state("features", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, features: Tensor, labels: Tensor) -> None:
        if features.ndim != 2 or labels.ndim != 1:
            raise ValueError("维度错误：需传入 2D features 和 1D labels")

        batch_size = features.shape[0]
        feat_cpu = features.detach().cpu()
        lab_cpu = labels.detach().cpu()

        if self.total_samples < self.max_samples:
            remaining = self.max_samples - self.total_samples.item()
            if batch_size > remaining:
                indices = torch.randperm(batch_size)[:remaining]
                feat_cpu = feat_cpu[indices]
                lab_cpu = lab_cpu[indices]
            
            self.features.append(feat_cpu)
            self.labels.append(lab_cpu)
            self.total_samples += feat_cpu.shape[0]

    # 优化点 2: 返回值直接改为 np.ndarray
    def compute(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.features:
            raise ValueError("未收到任何数据，无法 compute。")

        features_all = torch.cat(self.features, dim=0).numpy()
        labels_all = torch.cat(self.labels, dim=0).numpy() # 直接转为 numpy

        print(f"正在使用 {features_all.shape[0]} 个样本建立 UMAP 基准流形...")
        
        self.mapper_ = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.umap_metric,
            random_state=self.random_state,
            **self.umap_kwargs       
        ).fit(features_all)
        
        return self.mapper_.embedding_, labels_all

    # 优化点 3: 直接返回 np.ndarray
    def transform(self, new_features: Tensor) -> np.ndarray:
        if self.mapper_ is None:
            raise RuntimeError("UMAP 模型尚未拟合。请先执行 compute() 建立基准空间。")
        
        feat_cpu = new_features.detach().cpu().numpy()
        print(f"正在将 {feat_cpu.shape[0]} 个新样本映射到基准空间...")
        
        # UMAP transform 输出的就是 numpy 数组
        return self.mapper_.transform(feat_cpu)

    def plot(
        self, 
        base_embeddings: Optional[np.ndarray] = None,
        base_labels: Optional[np.ndarray] = None, 
        new_embeddings: Optional[np.ndarray] = None,
        new_labels: Optional[np.ndarray] = None,
        ax: Optional[plt.Axes] = None, 
        title: str = "UMAP Feature Space",
        classes: Optional[list] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure
        

        # 1. 绘制基准数据：直接把等长的 base_labels 传给 hue 即可！
        if base_embeddings is not None and base_labels is not None:
            import seaborn as sns
            base_labels=[classes[lbl] for lbl in base_labels] if classes is not None else base_labels
            sns.scatterplot(
                x=base_embeddings[:, 0],
                y=base_embeddings[:, 1],
                hue=base_labels,       # 重点：Seaborn 会自动根据这个列表切分已知类别
                hue_order=classes,
                palette='tab10',       # 指定调色盘
                alpha=0.8,
                s=4,
                marker='o',
                ax=ax                  # 绘制到指定的子图上
            )

        # 2. 绘制新数据（如 OSR 未知样本）
        if new_embeddings is not None:
            import seaborn as sns
            if new_labels is not None:
                # 传入带前缀的等长列表，防止新类别的标签跟已知类别在图例中冲突
                new_labels=[classes[lbl] for lbl in new_labels] if classes is not None else new_labels
                sns.scatterplot(
                    x=new_embeddings[:, 0],
                    y=new_embeddings[:, 1],
                    hue=new_labels,    # 重点：同样直接传列表
                    hue_order=classes,  # 确保新类别在图例中有独立的颜色
                    palette='Set1',
                    alpha=0.9,
                    s=40,
                    marker='*',
                    edgecolors='black',
                    linewidths=0.5,
                    ax=ax
                )
            else:
                # 如果新数据没有细分标签，作为一个整体特殊层传入单字符串 label
                sns.scatterplot(
                    x=new_embeddings[:, 0],
                    y=new_embeddings[:, 1],
                    color='red',
                    alpha=0.9,
                    s=40,
                    marker='*',
                    edgecolors='black',
                    linewidths=0.5,
                    label='New Samples',
                    ax=ax
                )

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='best', fontsize='small', markerscale=4)
        return fig, ax
    
if __name__ == "__main__":
    umap_vis = UmapVis(n_neighbors=15, min_dist=0.1)

    print("=== Phase 1: Fit Baseline ===")
    for _ in range(3):
        train_features = torch.randn(128, 256) 
        train_labels = torch.randint(0, 4, (128,))
        umap_vis.update(train_features, train_labels)

    base_emb, base_labels = umap_vis.compute()

    print("=== Phase 2: Transform New Data ===")
    test_features_unknown = torch.randn(50, 256) + 3.0  
    # 注意这里：为了在 plot 时保持逻辑顺畅，模拟的 labels 转成了 numpy
    test_labels_unknown = np.full((50,), -1)          

    new_emb = umap_vis.transform(test_features_unknown)

    print("=== Phase 3: Plotting ===")
    fig = umap_vis.plot(
        base_embeddings=base_emb, 
        base_labels=base_labels, 
        new_embeddings=new_emb,       
        new_labels=test_labels_unknown,
        title="UMAP Open Set Recognition Space"
    )
    plt.show()