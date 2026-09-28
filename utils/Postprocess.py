#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2023/7/20 15:51

@author: AS ME
"""
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import functional as F_metric
from sklearn.manifold import TSNE
from os.path import join
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(data, labels, label_names=None, title='ConfusionMatrix', dir=None,cbar=True):
    """

    :param data: 最后一层的输出
    :param labels: 分类结果对应的正确标签
    :param title: 混淆矩阵标题
    """

    '''
    混淆矩阵
    '''
    data = data.numpy() if isinstance(data, Tensor) else data
    labels = labels.numpy() if isinstance(labels, Tensor) else labels

    if data.shape[1] > 1:
        data = np.argmax(data, axis=-1)

    matrix = confusion_matrix(labels, data, normalize='true')*100

    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues", xticklabels=label_names, yticklabels=label_names,cbar=cbar)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title, pad=20)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    if dir:
        plt.savefig(dir, transparent=False)
    plt.show()


def plot_series(series, x_bias=0, xlabel='X', ylabel='Y', title='FigName', dir='./', background=True, subFig=False):
    """
    输入一批一元序列，画出序列的波形
    :param background:
    :param series:【B,L,1】
    :param x_bias:
    :param xlabel:
    :param ylabel:
    :param title:
    :param dir:
    """
    x = np.linspace(x_bias, series.shape[1] + x_bias, series.shape[1])

    plt.figure(figsize=(10, 8))  # 设置大图的大小
    plt.suptitle(title, fontsize=16)
    for i in range(series.shape[0]):
        if subFig:
            plt.subplot(series.shape[0], 1, i + 1)  # 划分出三行一列（三个子图），现在编辑第j个
            plt.plot(x, series[i], label=f'Series {i}')  # 子图标题
        else:
            plt.plot(x, series[i])

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()

        if background:
            plt.grid(True)
            plt.axis('on')
        else:
            plt.grid(False)
            plt.axis('off')
    plt.subplots_adjust()
    # plt.savefig(dir + title, format='png', transparent=True)
    plt.show()  # 显示图形


def plot_tsne(data, labels, label_name=None, dir=None, title='T-SNE Visual'):
    tsne = TSNE(n_components=2, perplexity=40, n_iter=100,learning_rate=1000)
    embedded_data = tsne.fit_transform(data)

    # 设置不同类别的颜色
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    # 绘制 t-SNE 图
    for i, y in enumerate(unique_labels):
        mask = (labels == y)
        label = label_name[i] if label_name is not None else f'Label {y}'
        plt.scatter(embedded_data[mask, 0], embedded_data[mask, 1], c=[colors[i]], label=label,rasterized=True)

    plt.legend(loc='best', bbox_to_anchor=(1,1))
    plt.title(title)
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    plt.subplots_adjust(right=0.8)
    if dir:
        plt.savefig(dir,dpi=300)
    plt.show()
