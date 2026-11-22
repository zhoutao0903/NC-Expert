# import numpy as np
# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# from torch.utils.data import TensorDataset, DataLoader
#
# # 设置文件路径
# data_dir = f"../data/PEMS08"
# file_path = os.path.join(data_dir, 'data.npz')
#
# # 检查文件是否存在
# if not os.path.exists(file_path):
#     raise FileNotFoundError(f"File not found: {file_path}")
#
# # 加载数据
# data = np.load(file_path)
# X1 = data['data']  # 假设数据存储在'data'键中
# X1 = X1.transpose(1, 0, 2)  # 转置以匹配时间序列维度
# print("X1.shape:", X1.shape)
# X = X1[:, :, 0]  # 取第一个特征
# print("X.shape:", X.shape)
# X = X[:, 1728:]  # 截取部分数据
# print("X.shape:", X.shape)
#
# # 每8条数据取均值合并
# X_mean = np.mean(X.reshape(-1, 8, X.shape[1] // 8), axis=1)
# print("X_mean.shape:", X_mean.shape)
#
# # 转换为PyTorch张量
# X_mean_tensor = torch.tensor(X_mean, dtype=torch.float32)
# print("X_mean_tensor.shape:", X_mean_tensor.shape)
#
# # 定义自编码器
# class Autoencoder(nn.Module):
#     def __init__(self, input_dim, encoding_dim):
#         super(Autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, encoding_dim),
#             nn.ReLU()
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(encoding_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 512),
#             nn.ReLU(),
#             nn.Linear(512, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, input_dim)
#         )
#
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return encoded, decoded
#
# # 设置超参数
# encoding_dim = 128  # 编码后的维度
# learning_rate = 0.0001
# num_epochs = 50
# batch_size = 64
# n_clusters = 2
#
# # 损失函数
# criterion = nn.L1Loss()
#
# # 数据加载器
# dataset = TensorDataset(X_mean_tensor)  # 使用均值数据
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
# # 训练自编码器
# def train_autoencoder(autoencoder, dataloader, optimizer, criterion, num_epochs):
#     autoencoder.train()
#     for epoch in range(num_epochs):
#         for batch in dataloader:
#             inputs = batch[0]
#             optimizer.zero_grad()
#             encoded, decoded = autoencoder(inputs)
#             loss = criterion(decoded, inputs)
#             loss.backward()
#             optimizer.step()
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
#
# # 训练自编码器
# print("Training Autoencoder...")
# autoencoder = Autoencoder(X_mean.shape[1], encoding_dim)  # 输入维度为 X_mean 的特征维度
# optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
# train_autoencoder(autoencoder, dataloader, optimizer, criterion, num_epochs)
#
# # 使用训练好的自编码器提取特征
# autoencoder.eval()
# with torch.no_grad():
#     X_encoded = autoencoder.encoder(X_mean_tensor)
#
# print("X_encoded.shape:", X_encoded.shape)  # 输出编码后的特征形状
#
# # 使用 K-means 进行聚类
# kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# kmeans.fit(X_encoded)
# labels = kmeans.labels_
#
# # 计算轮廓系数
# silhouette_avg = silhouette_score(X_encoded, labels)
# print(f'For n_clusters = {n_clusters}, the average silhouette score is: {silhouette_avg:.4f}')
#
# # 11. 打印每个类别的数据形状并保存为 .npz 文件
# clustered_data = {}
# output_dir = os.path.join(data_dir, "clusters")
# os.makedirs(output_dir, exist_ok=True)  # 创建保存目录
#
# for i in range(n_clusters):  # 假设标签从0开始
#     clustered_data[i] = X1[labels == i]
#     cluster_file_path = os.path.join(output_dir, f"cluster_{i}.npz")
#     np.savez(cluster_file_path, data=clustered_data[i])
#     print(f"Cluster {i} data saved to {cluster_file_path}")
#
# for cluster_id, cluster_data in clustered_data.items():
#     print(f'Cluster {cluster_id}: {cluster_data.shape}')
#
# print("类别 0 的数据集合形状:", clustered_data[0].shape)


import os
import torch
import numpy as np
import os
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

    return (trainset_loader_0, valset_loader_0, testset_loader_0, SCALER_0, num_sites_0), \
           (trainset_loader_1, valset_loader_1, testset_loader_1, SCALER_1, num_sites_1)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    data_dir = "../data/PEMS04/clusters"
    (trainset_loader_0, valset_loader_0, testset_loader_0, SCALER_0, num_sites_0), \
    (trainset_loader_1, valset_loader_1, testset_loader_1, SCALER_1, num_sites_1) = data_provider(args, data_dir, tod=True, dow=True, batch_size=args.batch_size)

    print(f'Cluster 0: Train/Val/Test loaders: {len(trainset_loader_0)}, {len(valset_loader_0)}, {len(testset_loader_0)}')
    print(f'Cluster 0: Number of sites: {num_sites_0}')
    print(f'Cluster 1: Train/Val/Test loaders: {len(trainset_loader_1)}, {len(valset_loader_1)}, {len(testset_loader_1)}')
    print(f'Cluster 1: Number of sites: {num_sites_1}')

# import os
# import torch
# import numpy as np
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# from torch.utils.data import TensorDataset, Dataset, DataLoader
#
# # 模拟lib.utils中的StandardScaler（如果你本地已有可删除这段）
# class StandardScaler:
#     def __init__(self, mean=0., std=1.):
#         self.mean = mean
#         self.std = std
#
#     def transform(self, x):
#         return (x - self.mean) / self.std
#
#     def inverse_transform(self, x):
#         return x * self.std + self.mean
#
#
# # ===================== 统一噪声接口：随机缺失 / 全局均值高斯噪声 / 脉冲噪声 / 连续块缺失 =====================
# def add_random_missing_noise(data_flow, missing_prob=0.0):
#     """
#     零散随机缺失：单一参数直接控制全局缺失点位占比
#     :param missing_prob: 全局任意数据点缺失概率 0~1
#     """
#     if missing_prob <= 0:
#         return data_flow.copy()
#     T, N = data_flow.shape
#     data_noise = data_flow.copy()
#     # 单层掩码，每个点直接以 missing_prob 概率置0缺失
#     final_mask = np.random.random(size=data_flow.shape) < missing_prob
#     data_noise[final_mask] = 0.0
#     return data_noise
#
#
# def add_gaussian_noise(data_flow, noise_std_scale=0.0, low_flow_thresh=5, noise_ratio=1.0, gauss_abs_std=None):
#     """
#     全局均值高斯观测噪声，支持两种浮动幅度控制：
#     1. 相对缩放：gauss_abs_std=None，std = 全局流量均值 * noise_std_scale
#     2. 绝对浮动规模：指定 gauss_abs_std，噪声标准差固定为此值，上下浮动范围可控
#     :param data_flow: [T,N] 原始流量
#     :param noise_std_scale: 相对模式缩放系数，gauss_abs_std 有效时失效
#     :param low_flow_thresh: 低于阈值不添加噪声，防止MAPE爆炸
#     :param noise_ratio: 施加噪声的数据点比例 0~1
#     :param gauss_abs_std: 绝对浮动标准差，如设为10代表噪声±10区间浮动
#     """
#     if (noise_std_scale <= 0 and gauss_abs_std is None) or noise_ratio <= 0:
#         return data_flow.copy()
#     data_noise = data_flow.copy()
#
#     # 二选一确定噪声标准差
#     if gauss_abs_std is not None and gauss_abs_std > 0:
#         std = gauss_abs_std
#     else:
#         global_flow_mean = np.mean(data_flow)
#         std = global_flow_mean * noise_std_scale
#
#     gauss = np.random.normal(loc=0.0, scale=std, size=data_flow.shape)
#
#     # 仅指定比例点位施加噪声
#     apply_mask = np.random.random(size=data_flow.shape) < noise_ratio
#     valid_mask = data_flow > low_flow_thresh
#     final_mask = apply_mask & valid_mask
#     data_noise[final_mask] += gauss[final_mask]
#
#     # 车流非负 + 整数
#     data_noise = np.clip(data_noise, 0.0, None)
#     data_noise = np.round(data_noise).astype(np.float32)
#     return data_noise
#
#
# def add_pulse_noise(data_flow, pulse_rate=0.0, pulse_amp=3.0, flow_th=1.0, noise_ratio=1.0):
#     """脉冲噪声，控制仅noise_ratio比例点位产生故障尖峰"""
#     if pulse_rate <= 0 or noise_ratio <= 0:
#         return data_flow.copy()
#     T, N = data_flow.shape
#     data_noise = data_flow.copy()
#
#     apply_mask = np.random.random(size=data_flow.shape) < noise_ratio
#     valid_mask = data_flow >= flow_th
#     pulse_mask = apply_mask & valid_mask & (np.random.random(size=data_flow.shape) < pulse_rate)
#
#     pos_pulse = pulse_mask & (np.random.random(size=data_flow.shape) > 0.5)
#     neg_pulse = pulse_mask & (np.random.random(size=data_flow.shape) <= 0.5)
#
#     data_noise[pos_pulse] = data_noise[pos_pulse] * pulse_amp
#     data_noise[neg_pulse] = 0.0
#
#     data_noise = np.clip(data_noise, a_min=0.0, a_max=None)
#     data_noise = np.round(data_noise).astype(np.float32)
#     return data_noise
#
#
# def add_block_missing_noise(data_flow, missing_rate=0.0, block_len=6, noise_ratio=1.0):
#     """连续块缺失：只在noise_ratio比例的站点上生成缺失块"""
#     if missing_rate <= 0 or noise_ratio <= 0:
#         return data_flow.copy()
#     T, N = data_flow.shape
#     data_noise = data_flow.copy()
#     total_elements = T * N
#     target_missing_num = int(total_elements * missing_rate)
#     missing_cnt = 0
#
#     # 先筛选允许加噪的站点
#     node_apply = np.random.random(N) < noise_ratio
#
#     while missing_cnt < target_missing_num:
#         # 只选允许加噪的节点
#         valid_nodes = np.where(node_apply)[0]
#         if len(valid_nodes) == 0:
#             break
#         n = np.random.choice(valid_nodes)
#         t_start = np.random.randint(0, T - block_len)
#         t_end = t_start + block_len
#         data_noise[t_start:t_end, n] = 0.0
#         missing_cnt += block_len
#     return data_noise
#
#
# def add_noise(data_flow, noise_type="none", noise_param=0.0, block_len=6, noise_ratio=1.0, gauss_abs_std=None):
#     """
#     统一噪声入口
#     gauss_abs_std：高斯噪声绝对浮动标准差
#     """
#     if noise_type == "none" or noise_param <= 0:
#         return data_flow.copy()
#     if noise_type == "random_missing":
#         # 仅传单一缺失概率，不再传入 noise_ratio
#         return add_random_missing_noise(data_flow, missing_prob=noise_param)
#     elif noise_type == "gaussian":
#         return add_gaussian_noise(
#             data_flow,
#             noise_std_scale=noise_param,
#             noise_ratio=noise_ratio,
#             gauss_abs_std=gauss_abs_std
#         )
#     elif noise_type == "pulse":
#         return add_pulse_noise(data_flow, pulse_rate=noise_param, pulse_amp=3.0, flow_th=1.0, noise_ratio=noise_ratio)
#     elif noise_type == "block_missing":
#         return add_block_missing_noise(data_flow, missing_rate=noise_param, block_len=block_len, noise_ratio=noise_ratio)
#     else:
#         raise ValueError("noise_type must be in ['none', 'random_missing', 'gaussian', 'pulse', 'block_missing']")
#
#
#
# # ===================== 数据集类（新增噪声比例、高斯绝对幅度参数） =====================
# class Dataset_Custom(Dataset):
#     def __init__(self, data_dir, cluster_id, flag='train', args=None, tod=False, dow=False, dom=False, log=None,
#                  noise_type="none", noise_param=0.0, block_len=6, noise_ratio=1.0, gauss_abs_std=None):
#         # size [seq_len, pred_len]
#         self.args = args
#         self.noise_type = noise_type
#         self.noise_param = noise_param
#         self.block_len = block_len
#         self.noise_ratio = noise_ratio
#         self.gauss_abs_std = gauss_abs_std
#         size = [args.seq_len, args.pred_len]
#         if size is None:
#             self.seq_len = 24 * 4 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.pred_len = size[1]
#         assert flag in ['train', 'val', 'test']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]
#
#         self.tod = tod
#         self.dow = dow
#         self.dom = dom
#         self.log = log
#         self.data_dir = data_dir
#         self.cluster_id = cluster_id
#         self.__read_data__()
#
#     def __read_data__(self):
#         file_path = os.path.join(self.data_dir, f'cluster_{self.cluster_id}.npz')
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"File not found: {file_path}")
#
#         data = np.load(file_path)
#         data = data['data'].astype(np.float32)
#         data = data.transpose(1, 0, 2)  # [T, N, F] T时间 N站点 F特征
#         print(f"cluster_{self.cluster_id} data.shape:", data.shape)
#
#         # 分离流量与时序特征
#         flow_raw = data[..., 0].copy()
#         tod_raw = data[..., 1:2].copy()
#         dow_raw = data[..., 2:3].copy()
#
#         # 通过统一接口施加噪声
#         flow_noisy = add_noise(
#             flow_raw,
#             noise_type=self.noise_type,
#             noise_param=self.noise_param,
#             block_len=self.block_len,
#             noise_ratio=self.noise_ratio,
#             gauss_abs_std=self.gauss_abs_std
#         )
#         data[..., 0] = flow_noisy
#
#         features = [0]
#         if self.tod:
#             features.append(1)
#         if self.dow:
#             features.append(2)
#         data = data[..., features]
#
#         num_train = int(len(data) * 0.6)
#         num_test = int(len(data) * 0.2)
#         num_vali = len(data) - num_train - num_test
#         border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
#         border2s = [num_train, num_train + num_vali, len(data)]
#
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]
#
#         # scaler 使用干净训练集均值方差，不被噪声干扰
#         train_flow_clean = data[border1s[0]:border2s[0], :, 0].copy()
#         self.scaler = StandardScaler(mean=train_flow_clean.mean(), std=train_flow_clean.std())
#         self.data_x = data[border1:border2, :, :].copy()
#         self.data_y = data[border1:border2, :, :1].copy()
#         self.data_x[..., 0] = self.scaler.transform(self.data_x[..., 0])
#
#         self.num_sites = data.shape[1]
#
#     def get_scaler(self):
#         return self.scaler
#
#     def get_num_sites(self):
#         return self.num_sites
#
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end
#         r_end = r_begin + self.pred_len
#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         return seq_x, seq_y
#
#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1
#
#
# # ===================== 数据集加载函数（全部噪声参数透传） =====================
# def data_provider(args, data_dir, tod=False, dow=False, batch_size=64, log=None,
#                   noise_type="none", noise_param=0.0, block_len=6, noise_ratio=1.0, gauss_abs_std=None):
#     # Cluster 0
#     train_set_0 = Dataset_Custom(
#         data_dir=data_dir, cluster_id=0, flag='train', args=args,
#         tod=tod, dow=dow, log=log,
#         noise_type=noise_type, noise_param=noise_param, block_len=block_len,
#         noise_ratio=noise_ratio, gauss_abs_std=gauss_abs_std
#     )
#     trainset_loader_0 = DataLoader(train_set_0, batch_size=batch_size, shuffle=True, drop_last=True)
#     test_set_0 = Dataset_Custom(
#         data_dir=data_dir, cluster_id=0, flag='test', args=args,
#         tod=tod, dow=dow, log=log,
#         noise_type=noise_type, noise_param=noise_param, block_len=block_len,
#         noise_ratio=noise_ratio, gauss_abs_std=gauss_abs_std
#     )
#     testset_loader_0 = DataLoader(test_set_0, batch_size=batch_size, shuffle=False, drop_last=True)
#     val_set_0 = Dataset_Custom(
#         data_dir=data_dir, cluster_id=0, flag='val', args=args,
#         tod=tod, dow=dow, log=log,
#         noise_type=noise_type, noise_param=noise_param, block_len=block_len,
#         noise_ratio=noise_ratio, gauss_abs_std=gauss_abs_std
#     )
#     valset_loader_0 = DataLoader(val_set_0, batch_size=batch_size, shuffle=False, drop_last=True)
#     SCALER_0 = train_set_0.get_scaler()
#     num_sites_0 = train_set_0.get_num_sites()
#
#     # Cluster 1
#     train_set_1 = Dataset_Custom(
#         data_dir=data_dir, cluster_id=1, flag='train', args=args,
#         tod=tod, dow=dow, log=log,
#         noise_type=noise_type, noise_param=noise_param, block_len=block_len,
#         noise_ratio=noise_ratio, gauss_abs_std=gauss_abs_std
#     )
#     trainset_loader_1 = DataLoader(train_set_1, batch_size=batch_size, shuffle=True, drop_last=True)
#     test_set_1 = Dataset_Custom(
#         data_dir=data_dir, cluster_id=1, flag='test', args=args,
#         tod=tod, dow=dow, log=log,
#         noise_type=noise_type, noise_param=noise_param, block_len=block_len,
#         noise_ratio=noise_ratio, gauss_abs_std=gauss_abs_std
#     )
#     testset_loader_1 = DataLoader(test_set_1, batch_size=batch_size, shuffle=False, drop_last=True)
#     val_set_1 = Dataset_Custom(
#         data_dir=data_dir, cluster_id=1, flag='val', args=args,
#         tod=tod, dow=dow, log=log,
#         noise_type=noise_type, noise_param=noise_param, block_len=block_len,
#         noise_ratio=noise_ratio, gauss_abs_std=gauss_abs_std
#     )
#     valset_loader_1 = DataLoader(val_set_1, batch_size=batch_size, shuffle=False, drop_last=True)
#     SCALER_1 = train_set_1.get_scaler()
#     num_sites_1 = train_set_1.get_num_sites()
#
#     return (trainset_loader_0, valset_loader_0, testset_loader_0, SCALER_0, num_sites_0), \
#            (trainset_loader_1, valset_loader_1, testset_loader_1, SCALER_1, num_sites_1)
#
#
# # ===================== 测试入口（完整噪声命令行参数） =====================
# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--seq_len', type=int, default=12)
#     parser.add_argument('--pred_len', type=int, default=12)
#     parser.add_argument('--batch_size', type=int, default=64)
#     parser.add_argument('--noise_type', type=str, default="none",
#                         choices=["none", "random_missing", "gaussian", "pulse", "block_missing"],
#                         help="噪声类型：随机缺失/高斯观测噪声/脉冲瞬时故障噪声/连续块缺失")
#     parser.add_argument('--noise_param', type=float, default=0.0,
#                         help="高斯相对缩放系数；随机/块缺失填缺失率；pulse填脉冲故障概率")
#     parser.add_argument('--block_len', type=int, default=6, help="连续缺失块的时间步长度")
#     parser.add_argument('--noise_ratio', type=float, default=1.0, help="全局施加噪声的数据点比例，0~1，1=全部加噪")
#     parser.add_argument('--gauss_abs_std', type=float, default=None, help="高斯噪声绝对浮动标准差，设置后覆盖noise_param相对模式")
#     args = parser.parse_args()
#
#     data_dir = "../data/PEMS04/clusters"
#     (trainset_loader_0, valset_loader_0, testset_loader_0, SCALER_0, num_sites_0), \
#     (trainset_loader_1, valset_loader_1, testset_loader_1, SCALER_1, num_sites_1) = data_provider(
#         args, data_dir, tod=True, dow=True, batch_size=args.batch_size,
#         noise_type=args.noise_type, noise_param=args.noise_param, block_len=args.block_len,
#         noise_ratio=args.noise_ratio, gauss_abs_std=args.gauss_abs_std
#     )
#
#     print(f"当前噪声配置: type={args.noise_type}, param={args.noise_param}, block_len={args.block_len}, "
#           f"noise_ratio={args.noise_ratio}, gauss_abs_std={args.gauss_abs_std}")
#     print(f'Cluster 0: Train/Val/Test loader num: {len(trainset_loader_0)}, {len(valset_loader_0)}, {len(testset_loader_0)}')
#     print(f'Cluster 0: Number of sites: {num_sites_0}')
#     print(f'Cluster 1: Train/Val/Test loader num: {len(trainset_loader_1)}, {len(valset_loader_1)}, {len(testset_loader_1)}')
#     print(f'Cluster 1: Number of sites: {num_sites_1}')

