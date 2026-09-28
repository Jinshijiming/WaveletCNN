import copy
import csv
import os
import time
from datetime import timedelta

import torch.nn as nn
import torchmetrics

from ptflops import get_model_complexity_info
from thop import clever_format
from torch.optim import lr_scheduler
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from utils.pytools import *

from provider import *


class Save_Data():
    def __init__(self, cfg, epoch_fn):
        super().__init__()
        self.version = cfg.version
        self.general = cfg.general
        self.path = cfg.path
        self.name = cfg.name
        self.dataset = cfg.dataset
        self.cmodel = cfg.model

        if os.path.exists(join(self.path['output_dir'], epoch_fn)) and self.general['mode'] == 'Train':
            os.remove(join(self.path['output_dir'], epoch_fn))

    def save_model(self, best_model_wts, name='bestmodel.pth'):
        torch.save(best_model_wts,
                   join(self.path['output_dir'], name))

    def save2csv(self, log: dict, root, filename):

        with open(join(root, filename), 'a', newline='') as t1:
            writer = csv.DictWriter(t1, fieldnames=log.keys())
            if t1.tell() == 0:
                writer.writeheader()
            writer.writerow(log)

    # Api保存C语言可调用模型用
    def save_model_api(self, traced_script_module):
        # 保存TorchScript模型

        if self.general['device'] == 'cpu':
            traced_script_module.save(
                join(self.path['output_dir'],
                     self.dataset['name'] + '_' + self.cmodel['name'] + '_cpu.pt'))
            print('保存CPU模型')
        else:
            traced_script_module.save(
                join(self.path['output_dir'],
                     self.dataset['name'] + '_' + self.cmodel['name'] + '_gpu.pt'))
            print('保存GPU模型')


class ClassfierTask(Save_Data):
    def __init__(self, cfg, epoch_fn='cls_epoch_log.csv', total_fn='cls_total_record.csv', model_fn='clsBestModel.pth',
                 objective_mode='max'):
        """

        :param cfg:
        :param epoch_fn:
        :param objective_mode: 指标优化方向 max/min
        """
        super().__init__(cfg, epoch_fn)
        # 加载模型，加载数据loader
        self.epoch_fn = epoch_fn
        self.total_fn = f"{self.name['file_name_compare']}_{total_fn}"
        self.model_fn = model_fn
        self.best_metric = float('inf') if objective_mode == 'min' else float('-inf')

        self.model = modelloader(cfg, task='cls').to(self.general['device'])
        self.dataloaders = self.dataloader(cfg)

        if self.general['optim'] == "adamw":
            self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                               lr=self.general['lr']
                                               )
        elif self.general['optim'] == "adam":
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                              lr=self.general['lr']
                                              )
        elif self.general['optim'] == "sgd":
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                             lr=self.general['lr'], momentum=0.9)
        else:
            raise ValueError

        # 学习率衰减
        if self.general['lr_scheduler'] == 'StepLR':
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.6)
        elif self.general['lr_scheduler'] == 'ReduceLROnPlateau':
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                            objective_mode,
                                                            factor=0.6,
                                                            patience=3,
                                                            # verbose=True,
                                                            min_lr=1e-7)
        elif self.general['lr_scheduler'] == 'cos_warm':
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=10,
                                                             num_training_steps=self.dataset['epochs'])

        self.early_stopping = EarlyStopping(patience=10, mode=objective_mode, min_delta=0.00001)

        self.train_time_elapsed = None

        self.loss_metric = torchmetrics.aggregation.MeanMetric().to(self.general['device'])
        self.metric = torchmetrics.MetricCollection({
            'acc': torchmetrics.Accuracy(average='macro', num_classes=self.dataset['num_classes'],
                                         task='multiclass'),
            'recall': torchmetrics.Recall(average='macro', num_classes=self.dataset['num_classes'],
                                          task='multiclass'),
            'precision': torchmetrics.Precision(average='macro', num_classes=self.dataset['num_classes'],
                                                task='multiclass'),
            'f1score': torchmetrics.F1Score(average='macro', num_classes=self.dataset['num_classes'],
                                            task='multiclass'),
        }).to(self.general['device'])

    def epoch_step(self, phase, epoch_metric, best_metric, best_model_wts):
        '''
        更新学习率、早停、最佳指标、模型
        :param phase:
        :param epoch_metric:
        :param best_metric:
        :param best_model_wts:
        :return:
        '''
        if phase != 'train':
            if self.general['lr_scheduler'] == 'StepLR':
                self.scheduler.step()
            elif self.general['lr_scheduler'] == 'ReduceLROnPlateau':
                self.scheduler.step(epoch_metric)
            elif self.general['lr_scheduler'] == 'cos_warm':
                self.scheduler.step()
            self.early_stopping(epoch_metric)

        if phase != 'train' and epoch_metric > best_metric:
            best_metric = epoch_metric
            best_model_wts = copy.deepcopy(self.model.state_dict())
            self.save_model(best_model_wts, name=self.model_fn)
        return best_metric, best_model_wts

    def train(self):
        since = time.time()  # 开始训练的时间
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_metric = self.best_metric

        for epoch in range(self.dataset['epochs']):
            print('Epoch {}/{}'.format(epoch, self.dataset['epochs'] - 1))
            print('-' * 10)
            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                self.loss_metric.reset()
                self.metric.reset()
                pbar = tqdm(self.dataloaders[phase])
                for batch in pbar:
                    # print(type(inputs))

                    with torch.set_grad_enabled(phase == 'train'):
                        output = self.model.train_step(batch)
                        loss, logits, labels = output['loss'], output['logits'], output['target']

                        # backward + optimize only if in training phase
                        # print("test")
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                            self.optimizer.zero_grad()

                    batch_loss = self.loss_metric(loss)
                    batch_acc = self.metric['acc'](logits, labels)
                    # 在进度条的右边实时显示数据集类型、loss值和精度
                    pbar.set_postfix({'Set': '{}'.format(phase),
                                      'Loss': '{:.4f}'.format(batch_loss),
                                      'Acc': '{:.4f}'.format(batch_acc)})

                ##### 一轮迭代，进行数据统计 #####
                # 显示该轮的loss和精度
                epoch_loss = self.loss_metric.compute().item()
                epoch_acc = self.metric['acc'].compute().item()

                best_metric, best_model_wts = self.epoch_step(phase, epoch_acc, best_metric, best_model_wts)

                # # 保存  当前一轮  的训练集精度、测试集精度和最高测试集精度
                epoch_log = {'epoch': epoch, 'phase': phase, 'epoch_loss': epoch_loss, 'epoch_acc': epoch_acc,
                             'best_metric': best_metric}
                self.save2csv(epoch_log, self.path['output_dir'], self.epoch_fn)

            print('Best test Acc: {:4f}'.format(best_metric))
            print()
            ### 收敛则中断 ####
            if self.early_stopping.early_stop and self.general['is_early_stop']:
                print('Early stopping')
                break
        self.train_time_elapsed = timedelta(seconds=int(time.time() - since))
        print(f'Training complete in {self.train_time_elapsed}')
        print('Best test Metric: {:4f}'.format(best_metric))

        del batch
        del labels
        self.test()
        return self.model

    def test(self, phase='test'):
        self.model.load_state_dict(
            torch.load(join(self.path['output_dir'], self.model_fn)),
            strict=False)

        # Each epoch has a training and validation phase
        self.model.eval()  # Set Decom to evaluate mode

        torch.no_grad()

        print('-' * 10)
        self.loss_metric.reset()
        self.metric.reset()
        pbar = tqdm(self.dataloaders[phase])  # dataloaders是字典
        for batch in pbar:

            output = self.model.train_step(batch)
            loss, logits, labels = output['loss'], output['logits'], output['target']

            batch_loss = self.loss_metric(loss)
            batch_acc = self.metric(logits, labels)['acc']
            # 在进度条的右边实时显示数据集类型、loss值和精度
            pbar.set_postfix({'Set': '{}'.format(phase),
                              'Loss': '{:.4f}'.format(batch_loss),
                              'Acc': '{:.4f}'.format(batch_acc)})
        ###################
        epoch_loss = self.loss_metric.compute().item()
        metric = self.metric.compute()
        # 显示该轮的loss和精度
        print('-' * 10)
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, metric['acc'].item()))
        print(batch[0].shape)
        flops, params = get_model_complexity_info(model=self.model,
                                                  input_res=tuple(batch[0][0].shape),
                                                  as_strings=False,
                                                  output_precision=4,
                                                  print_per_layer_stat=False,
                                                  verbose=False,
                                                #   input_constructor=lambda res: {'x': torch.randn(res).to(self.general['device'])}
                                                  
                                                  )
        print("Params: %.4f M ", params / 1e6)
        print("FLOPs: %.4f G ",flops / 1e9)
        flops, params = clever_format([flops, params], '%.2f')
        print('-' * 10)
        print("%s | %s | %s" % ("Model", "Params", "FLOPs"))
        print("---|---|---")
        print("%s | %s | %s" % (self.cmodel['name'], params, flops))

        #########  记录整体表现：训练时间、最好精度。 ######
        total_record = {'model': f"{self.cmodel['name']}_" + self.version,
                        'flops': flops,
                        'params': params,
                        'train_time': self.train_time_elapsed,
                        'loss': epoch_loss
                        }
        metric = {key: value.item() for key, value in metric.items()}
        total_record.update(metric)
        self.save2csv(total_record, './outputs', self.total_fn)
        return self.model

    def dataloader(self, cfg):
        transform = transform_provider(cfg.model['name'])
        _, train_dl, _, _ = SigData_provider('train', cfg, extra_T=transform)
        _, val_dl, _, _ = SigData_provider('val', cfg)
        _, test_dl, _, _ = SigData_provider('test', cfg)
        dataloader = {'train': train_dl, 'val': val_dl, 'test': test_dl}
        return dataloader

    def run(self):
        if self.general['mode'] == 'Train':
            # 训练模型
            self.train()
        elif self.general['mode'] == 'Test':
            # 导入预训练模型
            # self.Decom.load_state_dict(
            #     torch.load(self.output_path_model + self.file_name + 'bestmodel.pth'))
            self.test()
