from torch.utils.data import Dataset, DataLoader
from lib.utils import print_log, StandardScaler, vrange
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch.utils.data import TensorDataset, DataLoader

class Dataset_Custom(Dataset):
    def __init__(self, data_dir, cluster_id, flag='train', args=None, tod=False, dow=False, dom=False, log=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.args = args
        size = [args.seq_len, args.pred_len]
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.tod = tod
        self.dow = dow
        self.dom = dom
        self.log = log
        self.data_dir = data_dir
        self.cluster_id = cluster_id
        self.__read_data__()

    def __read_data__(self):
        file_path = os.path.join(self.data_dir, f'cluster_{self.cluster_id}.npz')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        data = np.load(file_path)
        data = data['data'].astype(np.float32)
        data = data.transpose(1, 0, 2)
        print(f"cluster_{self.cluster_id} data.shape:", data.shape)

        features = [0]  # [0, 1, 2]
        if self.tod:
            features.append(1)
        if self.dow:
            features.append(2)
        data = data[..., features]

        num_train = int(len(data) * 0.6)
        num_test = int(len(data) * 0.2)
        num_vali = len(data) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(data)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.scaler = StandardScaler(mean=data[border1s[0]:border2s[0], :, 0].mean(), std=data[border1s[0]:border2s[0], :, 0].std())
        self.data_x = data[border1:border2, :, :].copy()
        self.data_y = data[border1:border2, :, :1].copy()
        self.data_x[..., 0] = self.scaler.transform(self.data_x[..., 0])

        self.num_sites = data.shape[1]  # 获取站点数

    def get_scaler(self):
        return self.scaler

    def get_num_sites(self):
        return self.num_sites

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


def data_provider(args, data_dir, tod=False, dow=False, batch_size=64, log=None):
    # Dataset is the data source, responsible for loading and preprocessing individual data samples.
    # DataLoader is a wrapper around Dataset, responsible for dividing data into batches and providing them to the model.

    # For cluster_0
    train_set_0 = Dataset_Custom(data_dir=data_dir, cluster_id=0, flag='train', args=args, tod=tod, dow=dow, log=log)
    trainset_loader_0 = DataLoader(train_set_0, batch_size=batch_size, shuffle=True, drop_last=True)
    test_set_0 = Dataset_Custom(data_dir=data_dir, cluster_id=0, flag='test', args=args, tod=tod, dow=dow, log=log)
    testset_loader_0 = DataLoader(test_set_0, batch_size=batch_size, shuffle=False, drop_last=True)
    val_set_0 = Dataset_Custom(data_dir=data_dir, cluster_id=0, flag='val', args=args, tod=tod, dow=dow, log=log)
    valset_loader_0 = DataLoader(val_set_0, batch_size=batch_size, shuffle=False, drop_last=True)
    SCALER_0 = train_set_0.get_scaler()
    num_sites_0 = train_set_0.get_num_sites()

    # For cluster_1
    train_set_1 = Dataset_Custom(data_dir=data_dir, cluster_id=1, flag='train', args=args, tod=tod, dow=dow, log=log)
    trainset_loader_1 = DataLoader(train_set_1, batch_size=batch_size, shuffle=True, drop_last=True)
    test_set_1 = Dataset_Custom(data_dir=data_dir, cluster_id=1, flag='test', args=args, tod=tod, dow=dow, log=log)
    testset_loader_1 = DataLoader(test_set_1, batch_size=batch_size, shuffle=False, drop_last=True)
    val_set_1 = Dataset_Custom(data_dir=data_dir, cluster_id=1, flag='val', args=args, tod=tod, dow=dow, log=log)
    valset_loader_1 = DataLoader(val_set_1, batch_size=batch_size, shuffle=False, drop_last=True)
    SCALER_1 = train_set_1.get_scaler()
    num_sites_1 = train_set_1.get_num_sites()

    # For cluster_2
    train_set_2 = Dataset_Custom(data_dir=data_dir, cluster_id=2, flag='train', args=args, tod=tod, dow=dow, log=log)
    trainset_loader_2 = DataLoader(train_set_2, batch_size=batch_size, shuffle=True, drop_last=True)
    test_set_2 = Dataset_Custom(data_dir=data_dir, cluster_id=2, flag='test', args=args, tod=tod, dow=dow, log=log)
    testset_loader_2 = DataLoader(test_set_2, batch_size=batch_size, shuffle=False, drop_last=True)
    val_set_2 = Dataset_Custom(data_dir=data_dir, cluster_id=2, flag='val', args=args, tod=tod, dow=dow, log=log)
    valset_loader_2 = DataLoader(val_set_2, batch_size=batch_size, shuffle=False, drop_last=True)
    SCALER_2 = train_set_2.get_scaler()
    num_sites_2 = train_set_2.get_num_sites()

    return (trainset_loader_0, valset_loader_0, testset_loader_0, SCALER_0, num_sites_0), \
           (trainset_loader_1, valset_loader_1, testset_loader_1, SCALER_1, num_sites_1), \
           (trainset_loader_2, valset_loader_2, testset_loader_2, SCALER_2, num_sites_2)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    data_dir = "../data/PEMS08/clusters"
    (trainset_loader_0, valset_loader_0, testset_loader_0, SCALER_0, num_sites_0), \
    (trainset_loader_1, valset_loader_1, testset_loader_1, SCALER_1, num_sites_1), \
    (trainset_loader_2, valset_loader_2, testset_loader_2, SCALER_2, num_sites_2) = data_provider(args, data_dir, tod=True, dow=True, batch_size=args.batch_size)

    print(f'Cluster 0: Train/Val/Test loaders: {len(trainset_loader_0)}, {len(valset_loader_0)}, {len(testset_loader_0)}')
    print(f'Cluster 0: Number of sites: {num_sites_0}')
    print(f'Cluster 1: Train/Val/Test loaders: {len(trainset_loader_1)}, {len(valset_loader_1)}, {len(testset_loader_1)}')
    print(f'Cluster 1: Number of sites: {num_sites_1}')
    print(f'Cluster 2: Train/Val/Test loaders: {len(trainset_loader_2)}, {len(valset_loader_2)}, {len(testset_loader_2)}')
    print(f'Cluster 2: Number of sites: {num_sites_2}')