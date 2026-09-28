import shutil

from omegaconf import OmegaConf
import os, subprocess
from os.path import join
from collections import Counter
from functools import reduce

import numpy as np
import torch
from sklearn.model_selection import train_test_split
# from accelerate import Accelerator

_accelerator = None


def get_all_dataset(data_set):
    """
    获取整个数据集
    避免使用dataset[:]切片式获取整个数据集，导致的getitem形状操作错误
    :param data_set:
    :return:
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(data_set, batch_size=len(data_set), shuffle=False)
    return next(iter(loader))


def get_accelerator():
    global _accelerator
    if _accelerator is None:
        _accelerator = Accelerator()
    return _accelerator


def init_conf(cfg):
    device = choose_cuda(cfg.general.device)
    
    # --------------
    # 处理版本指定
    # --------------
    # 检查是否指定版本;指定且文件存在的话，加载该版本的配置文件
    if not OmegaConf.is_interpolation(cfg, 'version'):
        mode=cfg.general.mode
        output_dir = cfg.path['output_dir']
        version=cfg.version
        
        cfg = OmegaConf.load(join(cfg.path['output_dir'], 'config.yaml'))
        # 覆盖旧配置文件的设备、模式和输出目录
        cfg.general.device = device  
        cfg.general.mode = mode
        cfg.version = version 
        cfg.path['output_dir'] = output_dir 
    else:
        # --------------
        # 处理输出目录
        # --------------
        # debug模式下时,对目录进行另外处理
        if cfg.general['debug_size']:
            # hydra自动建的目录，debug时不需要
            if os.path.exists(cfg.path['output_dir']):
                shutil.rmtree(cfg.path['output_dir'])

            cfg.path['output_dir'] = './outputs/debug'
            cfg.name['file_name_compare'] = 'debug'

        os.makedirs(cfg.path['output_dir'], exist_ok=True)  # 创建输出目录，不用hydra创建
        
        # 不指定，则保存配置文件
        cfg.general.device = device
        OmegaConf.save(cfg, join(cfg.path['output_dir'], 'config.yaml'), resolve=True)

    print(
        f'************** mode:{cfg.general["mode"]} version:{cfg.version} model:{cfg.model["name"]} 调试：{cfg.general["debug_size"]}****************')

    # --------------
    # 将插值引用解析为固定配置
    # --------------
    OmegaConf.resolve(cfg)

    return cfg


def dir_increment(dir, prefix: str = None, suffix: str = 'version'):
    """
    根据给定的 pattern，匹配当前目录下的文件名，返回 pattern_num，其中 num 按照递增顺序变化。
    :param dir: 扫描的目录
    :param prefix: 前缀
    :param pattern: 模式字符串，包含数字部分
    :return: 更新后的模式字符串
    """
    if prefix:
        pattern = f"{prefix}__" + suffix
    else:
        pattern = suffix

    os.makedirs(join('./', dir), exist_ok=True)
    # 获取当前目录下所有文件
    files = os.listdir(join('./', dir))
    # 存储匹配到的数字
    nums = []

    # 遍历文件，检查文件名是否符合模式（即模式+下划线+数字）
    for file in files:
        if file.startswith(pattern + "_") and file[len(pattern) + 1:].isdigit():
            # 提取文件名中的数字部分，并加入 nums 列表
            num = int(file[len(pattern) + 1:])
            nums.append(num)

    # 如果 nums 列表为空，说明没有匹配到文件，返回模式加上下划线和数字1
    if not nums:
        next_num = 1
    else:
        # 找到最大数字并递增
        next_num = max(nums) + 1
    # 返回更新后的模式字符串
    return pattern + f"_{next_num}"


def choose_cuda(devices: str):
    if isinstance(devices, int):
        devices = str(devices)
        
    if devices == 'cpu':
        return 'cpu'

    if torch.cuda.is_available():
        if 'cuda' in devices:
            devices = '0,1,2,3,4,5,6,7'

        # 使用 nvidia-smi 命令获取 GPU 使用情况和内存信息
        # nvidia-smi不受CUDA_VISIBLE_DEVICES环境变量的影响
        gpu_info = subprocess.check_output(
            ['nvidia-smi', f'--id={devices}', '--query-gpu=index,utilization.gpu,memory.used',
             '--format=csv,noheader,nounits'],
            universal_newlines=True)

        # 按行分割使用情况和内存信息
        gpu_usage = gpu_info.strip().split('\n')
        # print('gpu_usage', gpu_usage)
        # 构建 GPU 使用率和内存的元组列表
        usage_memory_tuples = [(int(line.split(',')[0]), int(line.split(',')[1]), int(line.split(',')[2])) for line
                               in
                               gpu_usage]
        # print('usage_memory_tuples',usage_memory_tuples)
        # 按使用率由低到高排序
        sorted_by_usage = sorted(usage_memory_tuples, key=lambda x: x[1])

        # 取使用率前三个
        top_three_by_usage = sorted_by_usage[:3]

        # 按内存由低到高排序
        sorted_top_three = sorted(top_three_by_usage, key=lambda x: x[2])

        # 输出排序后的 GPU 信息
        # print("按照使用率由低到高排序，使用率前三个按照内存由低到高排序的 GPU：")
        # for index, usage, memory in sorted_top_three:
        #     print(f"GPU {index}: 使用率={usage}%, 内存={memory} MB")
        print(f"GPU {sorted_top_three[0][0]}: 使用率={sorted_top_three[0][1]}%, 内存={sorted_top_three[0][2]} MB")
        cuda = 'cuda:' + str(sorted_top_three[0][0])
        return cuda
    elif torch.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def calculate_probability(tensor, target: str = None):
    lst = tensor.cpu().numpy().tolist() if isinstance(tensor, torch.Tensor) else tensor
    # 使用Counter计算每个元素的出现次数
    counts = Counter(lst)
    if target not in counts:
        counts[target] = 0
    # 计算总元素个数
    total_elements = len(lst)

    # 计算每个元素的出现概率
    probabilities = {key: count / total_elements for key, count in counts.items()}

    return probabilities


class EarlyStopping:
    def __init__(self, patience=5, mode='min', min_delta=0.):
        """
        Args:
            patience (int): 如果在多少个epoch后性能指标没有改善就停止训练。
            min_delta (float): 用于定义性能指标变化的最小值（绝对值），即多少才算做改善。
            mode (str): 'min'表示性能指标下降是好的（如损失），'max'表示性能指标上升是好的（如准确率）。
            monitor (str): 要监控的性能指标名称，如'loss'、'accuracy'等。
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        if self.mode not in ['min', 'max']:
            raise ValueError("mode参数只能是'min'或'max'")

        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_value):
        if self.best_score is None:
            self.best_score = current_value
        elif self._is_improvement(current_value):
            if self.mode == 'min':
                print(f'Value decreased ({self.best_score:.6f} --> {current_value:.6f})')
            else:  # mode == 'max'
                print(f'Value increased ({self.best_score:.6f} --> {current_value:.6f})')
            self.best_score = current_value
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def _is_improvement(self, current_value):
        if self.mode == 'min':
            return current_value < self.best_score - self.min_delta
        else:  # mode == 'max'
            return current_value > self.best_score + self.min_delta


def split(*tuple, test_size=0.2, random_state, split_classes=1, dataset_sizes):
    '''
        主要计算，以多大的间隔分层dataset_sizes / value
        num_classes:[classes,[other]]
    '''

    value = reduce(lambda x, y: x * y, split_classes)

    array = np.arange(value)  # 生成从 0 到 10 的数组，每个元素都不同

    # 创建一个形状为  的数组，每一行都填充不同的值
    array = np.tile(array, (int(dataset_sizes / value), 1)).T
    array = np.concatenate(array)
    print(array)

    return train_test_split(*tuple, test_size=test_size, random_state=random_state, stratify=array)


if __name__ == '__main__':
    print(choose_cuda())
