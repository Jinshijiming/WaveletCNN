# Wavelet-Based Lightweight Convolutional Neural Network for Wireless Signal Recognition(Official Code)

## 数据准备

将RML2016.10a等数据集下载存储，并修改`dataset\conf.yaml`对应的数据集路径

## 训练-测试

```
python run.py suffix=flops dataset=RML128  model=WaveletCNN  dataset.epochs=100 dataset.norm=unit-power
```

更多训练命令，参考`scripts`目录

# 运行环境

`python>=3.10`,其他安装包版本请参考`requirements.txt`
