#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2024-04-10 16:12

@author: AS ME
"""
import torch
import matplotlib.pyplot as plt
##幅值相位
def cal_amplitude_phase_fre(IQ):
    #[B,L,N]
    if IQ.dim()<3:
        IQ=torch.unsqueeze(IQ,dim=0)
    # 计算幅值
    amplitude = torch.sqrt(IQ[:,:,0] ** 2 + IQ[:,:,1] ** 2)

    # 计算相位
    phase = torch.atan2(IQ[:,:,1] , IQ[:,:,0])  # 使用atan2函数可以处理所有四个象限的角度
    fre=torch.diff(phase)
    return amplitude.unsqueeze(dim=-1), phase.unsqueeze(dim=-1), fre.unsqueeze(dim=-1)

# 星座图
def plot_constellation(stars):
    fig, ax = plt.subplots()

    for star in stars:
        x, y = star
        ax.scatter(x, y, color='blue')  # 绘制星点

    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_title('星座图')
    ax.grid(True)

    plt.show()

if __name__=='__main__':
    data=torch.randn([128,2])
    aml,theta,fre=cal_amplitude_phase_fre(data)
    print(aml.shape)
    print(theta.shape)
    print(fre.shape)