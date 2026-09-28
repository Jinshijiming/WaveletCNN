import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    '''
    定义可覆盖的train_step
    '''

    def __init__(self):
        super().__init__()
        self._device = None
        self.criterion = F.cross_entropy

    def train_step(self, batch: torch.Tensor):
        """
        输入 batch，返回：
        - total_loss: 用于反向传播的总损失
        - loss_dict: 记录每个子损失（便于 log）
        - outputs: 预测结果（可选）
        """
        x, y = batch
        x = x.to(self.device)
        y = y[:, 1].to(self.device)
        logits = self(x)
        loss = self.criterion(logits, y)
        return {
            'loss': loss,
            'logits': logits,
            'pred': torch.argmax(logits, dim=1),
            'target': y
        }

    def infer_step(self, batch: torch.Tensor):
        """
        infer_step 的 Docstring
        
        :param self: 说明
        :param batch: x 或 x,y
        """
        if isinstance(batch, list):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
        else:
            x = batch
            x = x.to(self.device)
            y = None

        logits = self(x)

        return {
            'logits': logits,
            'pred': torch.argmax(logits, dim=1),
            'target': y
        }

    @property
    def device(self):
        if self._device is None:
            try:
                self._device = next(self.parameters()).device
            except StopIteration:
                self._device = next(self.buffers()).device
        return self._device

    @classmethod
    def from_config(cls, config: dict):
        '''
        通过配置字典创建模型实例
        只保留构造函数需要的参数，忽略多余参数
        :param config:
        :return:
        '''
        import inspect
        sig = inspect.signature(cls.__init__)
        # 去掉 self 参数
        params = list(sig.parameters.keys())[1:]

        # 只保留构造函数需要的参数
        filtered = {k: config[k] for k in params if k in config}
        return cls(**filtered)
