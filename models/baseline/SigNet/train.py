"""
SigNet
"""
import argparse
import os
import numpy as np
import time
import csv
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import signet as model_file
from torch.autograd import Variable
import torch.nn.functional as F
import pickle
# torch.backends.cudnn.enabled = False
num = 1  # /////////////////////////////////////////////
def train_model(args, batch_signal, dataset_sizes, model, criterion, optimizer, scheduler, num_epochs=50):
    since = time.time()  # 开始训练的时间
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                # scheduler.step()
                model.train()  # Set Decom to training mode
            else:
                model.eval()  # Set Decom to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            signal_num = 0
            # Iterate over data.
            pbar = tqdm(batch_signal[phase])
            for inputs, labels in pbar:
                inputs = inputs.cuda(num).float()  # (batch_size, 2, 128)
                labels = labels.cuda(num)  # (batch_size, )
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                signal_num += inputs.size(0)
                # 在进度条的右边实时显示数据集类型、loss值和精度
                epoch_loss = running_loss / signal_num
                epoch_acc = running_corrects.double() / signal_num
                pbar.set_postfix({'Set': '{}'.format(phase),
                                  'Loss': '{:.4f}'.format(epoch_loss),
                                  'Acc': '{:.4f}'.format(epoch_acc)})
                # print('\r{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), end=' ')
            if phase == 'train':
                scheduler.step()
            # 显示该轮的loss和精度
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the Decom
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            # 保存当前的训练集精度、测试集精度和最高测试集精度
            with open("./result_lr={}.csv".format(args.lr), 'a', newline='') as t1:
                writer_train1 = csv.writer(t1)
                writer_train1.writerow([epoch, phase, epoch_loss, epoch_acc, best_acc])
        # 保存测试精度最高时的模型参数
        torch.save(best_model_wts, "./best_model_parameters_lr={}.pth".format(args.lr))
        print('Best test Acc: {:4f}'.format(best_acc))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    # load best Decom weights
    model.load_state_dict(best_model_wts)
    return model


def prepare_data(args):
    # 导入数据集
    data_path_I = '/public/qkf/qiukunfeng/data/signal_npy/my_11_original_train_I.npy'  # I训练集
    data_path_Q = '/public/qkf/qiukunfeng/data/signal_npy/my_11_original_train_Q.npy'  # Q训练集
    train_label_path = '/public/qkf/qiukunfeng/data/signal_npy/radio11CNormTrainSnrY.npy'  # 训练集标签
    test_path_I = '/public/qkf/qiukunfeng/data/signal_npy/my_11_original_test_I.npy'  # I测试集
    test_path_Q = '/public/qkf/qiukunfeng/data/signal_npy/my_11_original_test_Q.npy'  # Q测试集
    test_label_path = '/public/qkf/qiukunfeng/data/signal_npy/radio11CNormTestSnrY.npy'  # 测试集标签
    train_I = np.load(data_path_I)  # [176000, 128]
    train_Q = np.load(data_path_Q)  # [176000, 128]
    train_2 = np.stack([train_I, train_Q], axis=-1)  # [176000, 128, 2]
    train_label = np.load(train_label_path)[1]  # 得到0到11的类标签数据
    test_I = np.load(test_path_I)  # [44000, 128]
    test_Q = np.load(test_path_Q)  # [44000, 128]
    test_2 = np.stack([test_I, test_Q], axis=-1)  # [44000, 128, 2]
    test_label = np.load(test_label_path)[1]  # 得到0到11的类标签数据
    # 训练集和测试集的数量
    data_sizes = {'train': len(train_label), 'test': len(test_label)}
    # 数组变张量
    train_2 = torch.from_numpy(train_2).permute(0, 2, 1)  # [312000, 2, 128]
    train_label = torch.from_numpy(train_label)
    test_2 = torch.from_numpy(test_2).permute(0, 2, 1)  # [156000, 2, 128]
    test_label = torch.from_numpy(test_label)
    print(train_2.shape, train_label.shape, test_2.shape, test_label.shape)
    # 把数据放在数据库中
    train_signal = torch.utils.data.TensorDataset(train_2, train_label)
    test_signal = torch.utils.data.TensorDataset(test_2, test_label)
    # 将训练集和测试集分批
    batch_signal = {'train': torch.utils.data.DataLoader(dataset=train_signal, batch_size=args.batch_size,
                                                         shuffle=True, num_workers=args.num_workers),
                    'test': torch.utils.data.DataLoader(dataset=test_signal, batch_size=args.batch_size,
                                                        shuffle=False, num_workers=args.num_workers)}
    return batch_signal, data_sizes


def arg_parse():
    parser = argparse.ArgumentParser(description='Signal_diagonal_matrix_CNN arguments.')
    parser.add_argument('--lr', dest='lr', type=float, help='Learning rate.')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int, help='Number of epochs to train.')
    parser.add_argument('--num_workers', dest='num_workers', type=int, help='Number of workers to load data.')

    parser.set_defaults(lr=0.001, batch_size=128, num_epochs=50, num_workers=8)
    return parser.parse_args()


def main():
    prog_args = arg_parse()
    with open("./result_lr={}.csv".format(prog_args.lr), 'a', newline='') as t:
        writer_train = csv.writer(t)
        writer_train.writerow(['epoch', 'phase', 'epoch_loss', 'epoch_acc', 'best_acc'])
    batch_signal, data_sizes = prepare_data(prog_args)  # 跳转到prepare_data函数，得到批训练集和批测试集
    # 模型放到GPU上
    model_ft = model_file.ResNet50(num_classes=11).cuda(num)
    print(model_ft)
    # 导入预训练模型
    # model_ft.load_state_dict(torch.load(r'D:\program_Q\new\torchvision_model\resnet50-19c8e357.pth'))
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    # optimizer_ft = optim.SGD([{"params": model_ft.parameters()}, {"params": filter_all}, {"params": bias_all}],
    #                          lr=prog_args.lr, momentum=0.9)
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=prog_args.lr)
    # 学习率衰减
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.8)
    # 训练模型
    train_model(prog_args, batch_signal, data_sizes, model_ft, criterion, optimizer_ft,
                exp_lr_scheduler, num_epochs=prog_args.num_epochs)


if __name__ == '__main__':
    main()
