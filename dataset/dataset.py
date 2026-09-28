from pathlib import Path
import shutil

import h5py
import numpy as np
import pickle
import torch
import yaml
from datasets import Dataset as HFDataset
from datasets import Features, Value, Sequence, Array2D
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

try:
    from . import iqtransforms as transforms
except ImportError:
    import iqtransforms as transforms

with open(Path(__file__).parent / 'conf.yaml', 'r') as file:
    cfg = yaml.safe_load(file)


class BaseDataset(Dataset):
    def __init__(self, **kwargs):
        self.root_path = None
        self.data_name = None
        self.label_name = None

        self.flag = None
        self.snr = None
        self.use_debug = None
        self.norm = None

        self.data = None
        self.label = None
        self.T = None
        self.mean = None
        self.std = None

    def load_data(self):

        normParm = np.load(self.root_path / 'NormParam.npz')
        self.mean = normParm['mean']
        self.std = normParm['std']

        if self.flag == 'train':
            ds = HFDataset.load_from_disk(self.root_path / 'train')
        elif self.flag == 'val':
            ds = HFDataset.load_from_disk(self.root_path / 'val')
        elif self.flag == 'test':
            ds = HFDataset.load_from_disk(self.root_path / 'test')

        else:
            raise ValueError("Dataset flag should be 'train', 'val' or 'test'.")

        ds.set_format(type='numpy', columns=['data', 'label'])
        ds.map(
            lambda x: None,
            batched=True,
            batch_size=2048,
            num_proc=8,  # 根据你的 CPU 核心数调整
            desc="Parallel Warming up"
        )
        return ds

    def __read_data__(self):
        self.ds = self.load_data()

        if self.snr != 'all':
            label=self.ds['label'][:]
            idx= np.where(np.isin(label[:, 0], self.snr))[0]
            self.ds = self.ds.select(idx)  #select比filter快
        if self.cls != 'all':
            label=self.ds['label'][:]
            idx= np.where(np.isin(label[:, 1], self.classes.index(self.cls)))[0]
            self.ds = self.ds.select(idx)  #select比filter快
            
        if self.use_debug:
            label = self.ds['label'][:]
            idx = np.arange(len(self.ds))
            idx, _,  label, _ = train_test_split(
                idx,  label, train_size=self.use_debug, random_state=42, stratify=label)
            self.ds = self.ds.select(idx)

        self.T = transforms.Compose([
            transforms.ToTensor(),
            transforms.normlization(self.mean, self.std, norm=self.norm)
        ])
        
    def __append_transform__(self, extra_T,use_extra_T=True):
        if extra_T and use_extra_T:
            extra = extra_T.transforms if isinstance(extra_T, transforms.Compose) else [extra_T]  # 单变换还是组合变换
            self.T = transforms.Compose([*self.T.transforms, *extra])
            
    def __getitem__(self, idx):

        # data = self.data[idx]
        item = self.ds[idx]
        data = item['data']
        data = self.T(data)

        # label = self.label[idx]
        label = item['label']
        return data.T, label

    def __len__(self):
        return len(self.ds)
    
    def get_all_labels(self):
        return self.ds['label'][:]

    @classmethod
    def from_config(cls, config: dict):
        import inspect
        sig = inspect.signature(cls.__init__)
        params = sig.parameters

        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return cls(**config)  # 直接全量透传，包容错误参数

        filtered = {k: config[k] for k in params if k != "self" and k in config}  # 过滤掉不需要的参数，避免传入错误
        return cls(**filtered)


class RML_2016a(BaseDataset):
    def __init__(self, flag='train', use_debug=0, snr='all',cls='all', norm=None, extra_T=None, use_extra_T: bool = True):
        """

        :param flag:
        :param use_debug:
        :param snr:
        :param norm:
        :param extra_T: 传入变换
        :param use_extra_T: 传入变化是否使用
        """
        super().__init__()
        self.data_name = cfg['RML_2016a']['data']
        self.label_name = cfg['RML_2016a']['label']
        self.root_path = Path(cfg['RML_2016a']['path'])
        self.classes = cfg['RML_2016a']['classes']
        self.split=cfg['RML_2016a']['split']
        
        self.flag = flag
        self.snr ,self.cls= snr,cls
        self.use_debug = use_debug
        self.norm = norm

        self.__prepare_data__()
        self.__read_data__()

        # 拼接变换增强，实现可插拔的设计
        self.__append_transform__(extra_T, use_extra_T)
    
        
    def __prepare_data__(self):
        if not (self.root_path / 'train').exists():
            print('First prepare/split dataset...')
            with open(self.root_path / 'RML2016.10a_dict.pkl', 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                # data.keys() = ['8psk', -10]
                snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], data.keys())))), [1, 0])

                X_train = []
                X_test = []
                Y_train = []
                Y_test = []
                np.random.seed(2016)
                for mod in mods:
                    for snr in snrs:
                        n_sample = data[(mod, snr)].shape[0]

                        n_train = n_sample * self.split['train']
                        train_idx = np.random.choice(range(0, int(n_sample)), size=int(n_train), replace=False)
                        test_idx = list(set(range(0, n_sample)) - set(train_idx))

                        X_train.append(data[(mod, snr)][train_idx])
                        X_test.append(data[(mod, snr)][test_idx])

                        Y_train.extend([(snr, mods.index(mod))] * int(n_train))
                        Y_test.extend([(snr, mods.index(mod))] * int(n_sample - n_train))

                X_train = np.vstack(X_train)
                X_test = np.vstack(X_test)
                Y_train = np.array(Y_train)
                Y_test = np.array(Y_test)
                _, X_val, _, Y_val = train_test_split(X_train, Y_train, test_size=0.25)

                # 数据集标准化
                # 防止训练集信息泄露
                mean = np.mean(X_train, axis=(0, 2))
                std = np.std(X_train, axis=(0, 2))

                np.savez(self.root_path / 'NormParam.npz', mean=mean, std=std)
                # np.save(self.root_path / self.data_name['train'], X_train)
                # np.save(self.root_path / self.data_name['val'], X_val)
                # np.save(self.root_path / self.data_name['test'], X_test)
                # np.save(self.root_path / self.label_name['train'], Y_train)
                # np.save(self.root_path / self.label_name['val'], Y_val)
                # np.save(self.root_path / self.label_name['test'], Y_test)

                split_info = {
                    'train': (X_train, Y_train),
                    'val': (X_val, Y_val),
                    'test': (X_test, Y_test)
                }

                for split_name, (X, Y) in split_info.items():
                    # 这里使用 from_dict，因为规模较小，内存吃得消
                    ds = HFDataset.from_dict({'data': X, 'label': Y})
                    ds.set_format(type='numpy', columns=['data', 'label'])

                    ds.save_to_disk(self.root_path / split_name)

class RML_2016b(BaseDataset):
    def __init__(self, flag='train', snr='all',cls='all', use_debug=0, norm=None, extra_T=None, use_extra_T: bool = True):
        super().__init__()
        self.data_name = cfg['RML_2016b']['data']
        self.label_name = cfg['RML_2016b']['label']
        self.root_path = Path(cfg['RML_2016b']['path'])
        self.classes = cfg['RML_2016b']['classes']
        self.split=cfg['RML_2016b']['split']
        
        self.flag = flag
        self.snr ,self.cls= snr,cls
        self.use_debug = use_debug
        self.norm = norm

        self.__prepare_data__()
        self.__read_data__()

        # 拼接变换增强，实现可插拔的设计
        self.__append_transform__(extra_T, use_extra_T)

    def __prepare_data__(self):
        if not (self.root_path / 'train').exists():
            print('First prepare/split dataset...')
            with open(self.root_path / 'RML2016.10b.dat', 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                # data.keys() = ['8psk', -10]
                snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], data.keys())))), [1, 0])

                X_train = []
                X_test = []
                Y_train = []
                Y_test = []
                np.random.seed(2016)
                for mod in mods:
                    for snr in snrs:
                        n_sample = data[(mod, snr)].shape[0]

                        n_train = n_sample * self.split['train']
                        train_idx = np.random.choice(range(0, int(n_sample)), size=int(n_train), replace=False)
                        test_idx = list(set(range(0, n_sample)) - set(train_idx))

                        X_train.append(data[(mod, snr)][train_idx])
                        X_test.append(data[(mod, snr)][test_idx])

                        mods.index(mod)
                        Y_train.extend([(snr, mods.index(mod))] * int(n_train))
                        Y_test.extend([(snr, mods.index(mod))] * int(n_sample - n_train))

                X_train = np.vstack(X_train)
                X_test = np.vstack(X_test)
                Y_train = np.array(Y_train)
                Y_test = np.array(Y_test)
                _, X_val, _, Y_val = train_test_split(X_train, Y_train, test_size=0.25)

                # 数据集标准化
                # 防止训练集信息泄露
                mean = np.mean(X_train, axis=(0, 2))
                std = np.std(X_train, axis=(0, 2))

                np.savez(self.root_path / 'NormParam.npz', mean=mean, std=std)
                # np.save(self.root_path / self.data_name['train'], X_train)
                # np.save(self.root_path / self.data_name['val'], X_val)
                # np.save(self.root_path / self.data_name['test'], X_test)
                # np.save(self.root_path / self.label_name['train'], Y_train)
                # np.save(self.root_path / self.label_name['val'], Y_val)
                # np.save(self.root_path / self.label_name['test'], Y_test)

                split_info = {
                    'train': (X_train, Y_train),
                    'val': (X_val, Y_val),
                    'test': (X_test, Y_test)
                }

                for split_name, (X, Y) in split_info.items():
                    # 这里使用 from_dict，因为规模较小，内存吃得消
                    ds = HFDataset.from_dict({'data': X, 'label': Y})
                    ds.set_format(type='numpy', columns=['data', 'label'])

                    ds.save_to_disk(self.root_path / split_name)


class RML_2018a(BaseDataset):
    def __init__(self, flag='train', snr='all',cls='all', use_debug=0, norm=None, extra_T=None, use_extra_T: bool = True):
        super().__init__()
        self.data_name = cfg['RML_2018a']['data']
        self.label_name = cfg['RML_2018a']['label']
        self.root_path = Path(cfg['RML_2018a']['path'])
        self.classes = cfg['RML_2018a']['classes']
        self.split=cfg['RML_2018a']['split']

        self.flag = flag
        self.snr ,self.cls= snr,cls
        self.use_debug = use_debug
        self.norm = norm

        self.__prepare_data__()
        self.__read_data__()
        # 拼接变换增强，实现可插拔的设计
        self.__append_transform__(extra_T, use_extra_T)

    def __prepare_data__(self):
        if not (self.root_path / 'train').exists():
            print('First prepare/split dataset...')

            def gen():
                with h5py.File(self.root_path / 'GOLD_XYZ_OSC.0001_1024.hdf5', "r") as f:
                    # List all groups
                    print("Keys: %s" % f.keys())
                    keys = list(f.keys())
                    X_ptr = f.get(keys[0])
                    Y_ptr = f.get(keys[1])
                    Z_ptr = f.get(keys[2])

                    print('数据读取完毕')
                    # 每次只读一小块（比如 1000 条），转换并写入磁盘
                    step_size = 10000
                    for i in range(0, len(X_ptr), step_size):
                        batch_x = X_ptr[i: i+step_size].transpose(0, 2, 1)
                        batch_y = Y_ptr[i: i+step_size]
                        batch_z = Z_ptr[i: i+step_size]

                        batch_y = np.reshape(np.argmax(batch_y, axis=1), (-1, 1))
                        labels = np.hstack((batch_z, batch_y))
                        for j in range(len(batch_x)):
                            yield {'data': batch_x[j], 'label': labels[j]}

            features = Features({
                'data': Array2D(dtype='float32', shape=(2, 1024)),  # 根据实际数据形状调整
                'label': Sequence(Value('int64'), length=2)  # [snr, class_label]
            })
            full_ds = HFDataset.from_generator(gen, features=features)
            full_ds.set_format(type='numpy', columns=['data', 'label'])
            
            labels = full_ds['label'][:]
            unique_classes = sorted(np.unique(labels[:, 1]))
            unique_snrs = sorted(np.unique(labels[:, 0]))

            idx_map = {'train': [], 'val': [], 'test': []}
            for cls in unique_classes:
                for snr in unique_snrs:
                    pos = np.where((labels[:, 1] == cls) & (labels[:, 0] == snr))[0]
                    n_sample = len(pos)
                    print(len(pos))
                    n_train = n_sample * self.split['train']
                    n_val = n_sample * self.split['val']
                    train_idx = np.random.choice(range(0, int(n_sample)), size=int(n_train), replace=False)
                    val_idx = np.random.choice(list(set(range(0, n_sample)) - set(train_idx)), size=int(n_val),
                                               replace=False)
                    test_idx = list(set(range(0, n_sample)) - set(train_idx) - set(val_idx))

                    idx_map['train'].append(pos[train_idx])
                    idx_map['val'].append(pos[val_idx])
                    idx_map['test'].append(pos[test_idx])

            idx_map['train'] = np.hstack(idx_map['train'])
            idx_map['val'] = np.hstack(idx_map['val'])
            idx_map['test'] = np.hstack(idx_map['test'])

            # 数据集标准化
            # 防止训练集信息泄露
            # X_train = np.array(full_ds['data'][idx_map['train']])  list->numpy这一过程很占内存 所以set_format(type='numpy'
            X_train = full_ds['data'][idx_map['train']]
            mean = np.mean(X_train, axis=(0, 2))
            std = np.std(X_train, axis=(0, 2))
            np.savez(self.root_path / 'NormParam.npz', mean=mean, std=std)

            # np.save(self.root_path / self.data_name['train'], X_train)
            # np.save(self.root_path / self.data_name['val'], X_val)
            # np.save(self.root_path / self.data_name['test'], X_test)
            # np.save(self.root_path / self.label_name['train'], Y_train)
            # np.save(self.root_path / self.label_name['val'], Y_val)
            # np.save(self.root_path / self.label_name['test'], Y_test)
            
            for split_name, idx_list in idx_map.items():
                split_ds = full_ds.select(idx_list)
                split_ds.save_to_disk(self.root_path / split_name)
            shutil.rmtree(Path(full_ds.cache_files[0]['filename']).parent.parent)  #删除generatord的缓存目录，内存直读无缓存            


class Sig2019_12(BaseDataset):
    def __init__(self, flag='train', snr='all',cls='all', use_debug=0, norm=None, extra_T=None, use_extra_T: bool = True):
        super().__init__()
        self.data_name = cfg['Sig2019_12']['data']
        self.label_name = cfg['Sig2019_12']['label']
        self.root_path = Path(cfg['Sig2019_12']['path'])
        self.classes = cfg['Sig2019_12']['classes']

        self.flag = flag
        self.snr ,self.cls= snr,cls
        self.use_debug = use_debug
        self.norm = norm

        self.__prepare_data__()
        self.__read_data__()
        # 拼接变换增强，实现可插拔的设计
        self.__append_transform__(extra_T, use_extra_T)

    def __prepare_data__(self):
        if not (self.root_path / 'NormParam.npz').exists():
            print('First prepare/split dataset...')
            X_train = np.load(self.root_path / self.data_name['train']).transpose(0, 2, 1)
            X_val = np.load(self.root_path / self.data_name['val']).transpose(0, 2, 1)
            X_test = np.load(self.root_path / self.data_name['test']).transpose(0, 2, 1)

            Y_train = np.load(self.root_path / self.label_name['train']).T
            Y_val = np.load(self.root_path / self.label_name['val']).T
            Y_test = np.load(self.root_path / self.label_name['test']).T
            # 数据集标准化
            # 防止训练集信息泄露
            mean = np.mean(X_train, axis=(0, 2))
            std = np.std(X_train, axis=(0, 2))

            np.savez(self.root_path / 'NormParam.npz', mean=mean, std=std)

            split_info = {
                'train': (X_train, Y_train),
                'val': (X_val, Y_val),
                'test': (X_test, Y_test)
            }

            for split_name, (X, Y) in split_info.items():
                # 这里使用 from_dict，因为规模较小，内存吃得消
                ds = HFDataset.from_dict({'data': X, 'label': Y})
                ds.set_format(type='numpy', columns=['data', 'label'])
                ds.save_to_disk(self.root_path / split_name)


class HKDD36(BaseDataset):
    def __init__(self, flag='train', use_debug=0, snr='all',cls='all', norm=None, extra_T=None, use_extra_T: bool = True):
        """

        :param flag:
        :param use_debug:
        :param snr:
        :param norm:
        :param extra_T: 传入变换
        :param use_extra_T: 传入变化是否使用
        """
        super().__init__()
        self.data_name = cfg['HKDD36']['data']
        self.label_name = cfg['HKDD36']['label']
        self.root_path = Path(cfg['HKDD36']['path'])
        self.classes = cfg['HKDD36']['classes']

        self.flag = flag
        self.snr ,self.cls= snr,cls
        self.use_debug = use_debug
        self.norm = norm

        self.__prepare_data__()
        self.__read_data__()

        # 拼接变换增强，实现可插拔的设计
        self.__append_transform__(extra_T, use_extra_T)

    def __prepare_data__(self):
        if not (self.root_path / 'train').exists():
            print('First prepare/split dataset...')
            import h5py
            label_base = np.arange(0, 36)  # [1,2,3]  repeat [1,1,2,2,3,3]  tile [1,1,2,2,3,3, 1,1,2,2,3,3]
            snr_base = np.arange(-20, 31, 2)

            label_train = np.repeat(label_base, 1000)
            label_train = np.tile(label_train, 26)
            snr_train = np.repeat(snr_base, 36*1000)
            Y_train = np.stack((snr_train, label_train), 1)  # 36*26*1000
            
            label_test = np.repeat(label_base, 500)
            label_test = np.tile(label_test, 26)
            snr_test = np.repeat(snr_base, 36*500)
            Y_test = np.stack((snr_test, label_test), 1)

            def gen(file_path, Y, step_size=10000):
                with h5py.File(file_path, "r") as f:
                    X_ptr = f.get('XTrainIQ')

                    print('数据读取完毕')
                    # 每次只读一小块（比如 1000 条），转换并写入磁盘
                    for i in range(0, len(X_ptr), step_size):
                        batch_x = X_ptr[i: i+step_size].squeeze(1).transpose(0, 2, 1)
                        labels = Y[i: i+step_size]
                        for j in range(len(batch_x)):
                            yield {'data': batch_x[j], 'label': labels[j]}
            ds_train = HFDataset.from_generator(
                gen, gen_kwargs={'file_path': self.root_path / 'original/HKDD_AMC36_train.mat', 'Y': Y_train})
            ds_test = HFDataset.from_generator(
                gen, gen_kwargs={'file_path': self.root_path / 'original/HKDD_AMC36_test.mat', 'Y': Y_test})
            ds_train.set_format(type='numpy', columns=['data', 'label'])
            ds_test.set_format(type='numpy', columns=['data', 'label'])

            # 数据集标准化
            # 防止训练集信息泄露
            X_train = ds_train['data'][:]
            mean = np.mean(X_train, axis=(0, 2))
            std = np.std(X_train, axis=(0, 2))

            np.savez(self.root_path / 'NormParam.npz', mean=mean, std=std)
            # np.save(self.root_path / self.data_name['train'], X_train)
            # np.save(self.root_path / self.data_name['val'], X_val)
            # np.save(self.root_path / self.data_name['test'], X_test)
            # np.save(self.root_path / self.label_name['train'], Y_train)
            # np.save(self.root_path / self.label_name['val'], Y_val)
            # np.save(self.root_path / self.label_name['test'], Y_test)
            ds_train.save_to_disk(self.root_path / 'train')
            ds_test.save_to_disk(self.root_path / 'test')
            shutil.rmtree(Path(ds_train.cache_files[0]['filename']).parent.parent) #删除generatord的缓存目录    
            shutil.rmtree(Path(ds_test.cache_files[0]['filename']).parent.parent)

    def load_data(self):
        # TODO: 优化数据集加载，加速page cache加载

        normParm = np.load(self.root_path / 'NormParam.npz')
        self.mean = normParm['mean']
        self.std = normParm['std']

        if self.flag == 'train':
            ds = HFDataset.load_from_disk(self.root_path / 'train')
        elif self.flag == 'val':
            ds = HFDataset.load_from_disk(self.root_path / 'test')
        elif self.flag == 'test':
            ds = HFDataset.load_from_disk(self.root_path / 'test')

        else:
            raise ValueError("Dataset flag should be 'train', 'val' or 'test'.")

        ds.set_format(type='numpy', columns=['data', 'label'])
        ds.map(
            lambda x: None,
            batched=True,
            batch_size=2048,
            num_proc=8,  # 根据你的 CPU 核心数调整
            desc="Parallel Warming up"
        )
        return ds

class TechRec(BaseDataset):
    def __init__(self, flag='train', snr='all', cls='all',use_debug=0, norm=None, extra_T=None, use_extra_T: bool = True):
        super().__init__()

        self.root_path = Path(cfg['TechRec']['path'])
        self.raw_data_path = self.root_path/'orginal'

        self.classes = cfg['TechRec']['classes']
        self.split = cfg['TechRec']['split']
        self.SNR = cfg['TechRec']['SNR']

        self.flag = flag
        self.snr ,self.cls= snr,cls
        self.use_debug = use_debug
        self.norm = norm

        self.signal_length = 1024

        self.__prepare_data__()
        self.__read_data__()

        # 拼接变换增强，实现可插拔的设计
        self.__append_transform__(extra_T, use_extra_T)

    def __prepare_data__(self):
        if not (self.root_path / 'train').exists():
            print('First prepare/split dataset for Techrec...')
            x_list = []
            y_list = []

            # 1. 遍历并加载目标文件夹下的所有 .bin 信号数据
            target_dirs = {"igent", "merelbeke", "rabot", "reep", "uz", "gentbrugge"}
            for bin_file in self.raw_data_path.rglob("*.bin"):
                if bin_file.parent.name not in target_dirs:
                    continue

                # 提取信号类型前缀 (去除数字和 'Msps' 等后缀)
                signal_type = bin_file.name.split("_")[0]
                if "Msps" in signal_type:
                    signal_type = "".join(c for c in signal_type if not c.isdigit() and c not in "Msp")

                if signal_type in self.classes:
                    segments = self.load_bin_file(bin_file, self.signal_length)
                    if len(segments) > 0:
                        x_list.append(segments)
                        y_list.extend([self.classes.index(signal_type)] * len(segments))

            # 合并为完整数组
            IQ_data_all = np.concatenate(x_list, axis=0)
            labels_all = np.array(y_list)

            # 2. 随机打乱并通过 array_split 均分为多份，分别注入不同梯度的 SNR 噪声
            indices = np.random.permutation(len(IQ_data_all))
            shuffled_IQ = IQ_data_all[indices]
            shuffled_label = labels_all[indices]

            snr_range = list(range(self.SNR['min'], self.SNR['max'] + 1, self.SNR['step']))
            chunks_IQ = np.array_split(shuffled_IQ, len(snr_range))
            chunks_label = np.array_split(shuffled_label, len(snr_range))

            X_all = []
            Y_all = []
            for i, snr in enumerate(snr_range):
                X_all.append(self.add_noise_awgn(chunks_IQ[i], snr))
                Y_all.append(np.stack([np.full(len(chunks_label[i]), snr), chunks_label[i]], axis=1))

            X_all = np.concatenate(X_all, axis=0)
            Y_all = np.concatenate(Y_all, axis=0)

            # Split dataset
            X_train, X_temp, Y_train, Y_temp = train_test_split(
                X_all, Y_all,
                test_size=1 - self.split['train'],
                random_state=42,
                stratify=Y_all[:, 1]
            )

            test_val_ratio = self.split['test'] / (self.split['test'] + self.split['val'])
            X_val, X_test, Y_val, Y_test = train_test_split(
                X_temp, Y_temp,
                test_size=test_val_ratio,
                random_state=42,
                stratify=Y_temp[:, 1]
            )

            # Standardize dataset using Training set statistics
            mean = np.mean(X_train, axis=(0, 2))
            std = np.std(X_train, axis=(0, 2))
            np.savez(self.root_path / 'NormParam.npz', mean=mean, std=std)

            split_info = {
                'train': (X_train, Y_train),
                'val': (X_val, Y_val),
                'test': (X_test, Y_test)
            }

            for split_name, (X, Y) in split_info.items():
                ds = HFDataset.from_dict({'data': X, 'label': Y})
                ds.set_format(type='numpy', columns=['data', 'label'])
                ds.save_to_disk(self.root_path / split_name)

    def load_bin_file(self, file_path, segment_length=1024):
        """
        读取二进制文件，分离 I 和 Q 通道，并切割为指定长度的数据段。
        MATLAB 的 'short' 数据对应 np.int16，同时原始文件中为 I 和 Q 交错排列。
        """
        # 1. 按照 numpy int16 读取为一维数组，并转成模型训练需要使用的 float32
        with open(file_path, "rb") as f:
            data = np.fromfile(f, dtype=np.int16).astype(np.float32)
        iq_data = data.reshape(-1, 2).T
        num_segments = iq_data.shape[1] // segment_length
        if num_segments == 0:
            return np.array([])
        iq_data = iq_data[:, :num_segments * segment_length]
        iq_segments = iq_data.reshape(2, num_segments, segment_length).transpose(1, 0, 2)

        return iq_segments

    def add_noise_awgn(self, x, snr):
        """
        向信号中注入指定信噪比 (SNR) 的加性高斯白噪声 (AWGN)。
        支持对一个 Batch (形状如 [B, 2, Length]) 的张量进行快速向量化运算。
        """
        x = np.array(x)
        signal_power = np.mean(x ** 2, axis=(1, 2), keepdims=True)
        noise_power = signal_power / (10 ** (snr / 10.0))
        noise = np.random.normal(0, 1, x.shape).astype(x.dtype)
        noise *= np.sqrt(noise_power / 2)

        return x + noise


if __name__ == '__main__':
    dataset = TechRec(flag='train', use_debug=0, norm=None, transform=transforms.Compose([]), use_extra_T=False)
    sig = dataset[1][0]
    print(sig.min(), sig.max())
    # print(sig.pow(2).sum())
    print(len(dataset))
    # normparm = np.load("/public/6MountData/DataDir/bsh/36InstituteData/Sig2019-12/radio12CNormValX.npy", allow_pickle=True)
    # print(normparm.shape)
