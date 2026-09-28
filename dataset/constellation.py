import numpy as np
import torch

# def iq_to_mcd(iq_signal, K=1, image_size=32):
#     """
#     将 IQ 信号转换为 MCD 张量表示：每个 offset 的星座图为二维栅格图
#     不绘图，适合 Dataset 中 __getitem__ 调用
#     Args:
#         iq_signal: np.ndarray, shape [L, 2]，float32
#         K: 采样因子（一个 symbol 有 K 个采样）
#         image_size: 星座图大小
#
#     Returns:
#         mcd_tensor: torch.Tensor, shape [K, 1, H, W]
#     """
#
#     i = iq_signal[:, 0]
#     q = iq_signal[:, 1]
#     i = (i - i.min()) / (i.max() - i.min())
#     q = (q - q.min()) / (q.max() - q.min())
#
#     mcd = []
#     for k in range(K):
#         dec_i = i[k::K]
#         dec_q = q[k::K]
#
#         x = np.clip((dec_i * image_size).astype(np.int32), 0, image_size - 1)
#         y = np.clip((dec_q * image_size).astype(np.int32), 0, image_size - 1)
#         # 用 bincount 构造密度图
#         idx_flat = y * image_size + x
#         heatmap = np.bincount(idx_flat, minlength=image_size ** 2).reshape(image_size, image_size)
#         heatmap = heatmap.astype(np.float32)
#         if heatmap.max() > 0:
#             heatmap /= heatmap.max()
#
#         mcd.append(heatmap)
#
#     mcd = np.stack(mcd, axis=0)  # [K, H, W]
#     return np.expand_dims(mcd, axis=1)  # [K, 1, H, W]

def iq_to_mcd(iq_signal, K=1, image_size=32):
    """
    使用 np.histogram2d 生成星座图张量
    """
    iq_signal=iq_signal.numpy()
    i = iq_signal[0]
    q = iq_signal[1]

    # 归一化到 [0, 1]
    # i = (i - i.min()) / (i.max() - i.min())
    # q = (q - q.min()) / (q.max() - q.min())

    mcd = []
    for k in range(K):
        dec_i = i[k::K]
        dec_q = q[k::K]

        # 使用 histogram2d 构造密度图
        heatmap, _, _ = np.histogram2d(
            dec_q, dec_i,  # 注意坐标顺序是 (y, x)
            bins=[image_size, image_size],
            # range=[[0, 1], [0, 1]]
        )

        # heatmap = heatmap.astype(np.float32)
        # if heatmap.max() > 0:
        #     heatmap /= heatmap.max()

        mcd.append(heatmap)

    mcd = np.stack(mcd, axis=0)  # [K, H, W]
    mcd=np.expand_dims(mcd, axis=1) # [K, 1, H, W]
    return  torch.tensor(mcd)
