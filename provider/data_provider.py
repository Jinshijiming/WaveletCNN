from torch.utils.data import DataLoader
from dataset import Data_Dict


# def SigData_provider(dataset, flag, snr='all', use_debug=0, use_scaler=True, batch_size=128, num_workers=8):
#     Data = Data_Dict[dataset]
#
#     data_set = Data(flag=flag, snr=snr, use_debug=use_debug, use_scaler=use_scaler)
#     # print(flag, len(data_set))
#
#     # data_loader = DataLoader(data_set, batch_size=2,shuffle=True, num_workers=num_workers), #多一个逗号，dataloader会变成元组 差错半天
#     data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
#     print(flag, len(data_loader))
#     return data_set, data_loader, len(data_set), data_set.classes

def SigData_provider(flag, cfg, extra_T=None):
    '''

    :param flag:
    :param cfg:
    :param extra_T: 可插拔的增强变换
    :return:
    '''
    config = {'flag': flag, 'use_debug': cfg.general['debug_size'], 'extra_T': extra_T, **cfg.dataset}
    data_set = Data_Dict[cfg.dataset['name']].from_config(config)

    data_loader = DataLoader(data_set,
                             batch_size=cfg.dataset['batch_size'],
                             shuffle=True,
                             num_workers=cfg.general['num_workers'],
                             persistent_workers=True,
                             pin_memory=True)
    print(flag, len(data_loader))
    return data_set, data_loader, len(data_set), data_set.classes


if __name__ == "__main__":
    data, _, datadl = SigData_provider('128', 'test', use_debug=300)
    x, y = data[:]
    print(type(x))
    # for batch_data, batch_labels in datadl:
    #     print("Batch data:", batch_data)
    #     print("Batch labels:", batch_labels)
    # print(x.shape)
