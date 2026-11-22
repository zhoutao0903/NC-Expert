import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt

# ===================== 全局固定随机种子函数 =====================
def set_all_seeds(seed=60):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

SEED = 60
set_all_seeds(SEED)

# --------------------------------------------------
# 1. 掩码工具
# --------------------------------------------------
def rand_mask(x, mask_ratio=0.5):
    mask = torch.rand_like(x) > mask_ratio
    masked_x = x.clone()
    masked_x[~mask] = float('nan')
    return masked_x, mask

# --------------------------------------------------
# 2. 1-D CNN 编码器
# --------------------------------------------------
class CNNEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, encoding_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        c = self.cnn(x)
        c = c.transpose(1, 2)
        _, h_n = self.gru(c)
        z = self.fc(h_n.squeeze(0))
        return z

# --------------------------------------------------
# 3. 掩码自编码器
# --------------------------------------------------
class MaskAutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=128, mask_ratio=0.5):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        self.encoder = CNNEncoder(input_dim, encoding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256), nn.ReLU(),
            nn.Linear(256, 1024), nn.ReLU(),
            nn.Linear(1024, 4096), nn.ReLU(),
            nn.Linear(4096, input_dim)
        )

    def forward(self, x, mask=None):
        if mask is not None:
            mask_tokens = self.mask_token.squeeze(0).expand_as(x)
            x = torch.where(mask, x, mask_tokens)
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec

# --------------------------------------------------
# 4. 聚类管道
# --------------------------------------------------
class DataClusteringPipeline:
    def __init__(self,
                 data_dir,
                 encoding_dim=128,
                 learning_rate=1e-4,
                 num_epochs=30,
                 batch_size=16,
                 n_clusters=2,
                 mask_ratio=0.5,
                 train_ratio=0.6):          # === 修改：新增训练段比例
        self.data_dir    = data_dir
        self.encoding_dim = encoding_dim
        self.lr          = learning_rate
        self.num_epochs  = num_epochs
        self.batch_size  = batch_size
        self.n_clusters  = n_clusters
        self.mask_ratio  = mask_ratio
        self.train_ratio = train_ratio
        self.model       = None
        self.kmeans      = None

    # === 修改：只取前 train_ratio 的时间步，按整"天"对齐 ===
    def _load_data(self, file_name='data.npz', steps_per_day=288):
        file_path = os.path.join(self.data_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)
        data = np.load(file_path)
        X1_full = data['data'].transpose(1, 0, 2)            # (N,T,F) 全量
        T_full  = X1_full.shape[1]                            # 如 17856
        days    = T_full // steps_per_day                     # 62
        train_days = int(round(days * self.train_ratio))       # 37
        T_train = train_days * steps_per_day                   # 10656
        X1_train = X1_full[:, :T_train, :]                     # 仅前 60%
        print(f'全量时间步 {T_full}（{days}天）→ 训练段 {T_train}（{train_days}天）')
        X_train = X1_train[:, :, 0]                            # (N,T_train) 流量
        return X_train, X1_train, X1_full

    def _train_model(self, X_tensor):
        dataset = TensorDataset(X_tensor)
        loader  = DataLoader(dataset, batch_size=self.batch_size,
                             shuffle=True, drop_last=False, num_workers=0)
        self.model = MaskAutoEncoder(X_tensor.shape[1],
                                     self.encoding_dim,
                                     self.mask_ratio)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.L1Loss(reduction='none')

        for epoch in range(self.num_epochs):
            self.model.train()
            for batch in loader:
                x = batch[0]
                masked_x, mask = rand_mask(x, self.mask_ratio)
                optimizer.zero_grad()
                z, x_rec = self.model(masked_x, mask)
                loss = (criterion(x_rec, x) * ~mask).sum() / (~mask).sum().clamp(min=1)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}/{self.num_epochs}, Mask-L1: {loss.item():.4f}')

    def _extract_features(self, X_tensor):
        self.model.eval()
        with torch.no_grad():
            z, _ = self.model(X_tensor)
        return z.cpu().numpy()

    def _perform_clustering(self, Z, X_mean):
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        raw_labels = self.kmeans.fit_predict(Z)
        df = pd.DataFrame({'label': raw_labels,
                           'volume': X_mean.mean(axis=1)})
        vol_per_cluster = df.groupby('label')['volume'].mean()
        new_order = vol_per_cluster.sort_values().index.tolist()
        remap = {old: new for new, old in enumerate(new_order)}
        labels = np.vectorize(remap.get)(raw_labels)
        print(f'Silhouette: {silhouette_score(Z, labels):.4f}')
        return labels

    # === 修改：训练/聚类只用前 60%，返回全量 X1 供热力图与统计 ===
    def cluster_data(self, file_name='data.npz'):
        X_train, X1_train, X1_full = self._load_data(file_name)
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        self._train_model(X_tensor)
        Z       = self._extract_features(X_tensor)
        labels  = self._perform_clustering(Z, X_train)
        indices = np.arange(X1_full.shape[0])          # 站点序号 0…N-1
        return X1_full, labels, indices

# --------------------------------------------------
# 5. 工具函数
# --------------------------------------------------
def cluster_data(data_dir,
                 encoding_dim=128,
                 learning_rate=1e-4,
                 num_epochs=30,
                 batch_size=16,
                 n_clusters=2,
                 file_name='data.npz',
                 mask_ratio=0.6,
                 train_ratio=0.6):            # === 修改：透传比例
    pipe = DataClusteringPipeline(data_dir,
                                  encoding_dim=encoding_dim,
                                  learning_rate=learning_rate,
                                  num_epochs=num_epochs,
                                  batch_size=batch_size,
                                  n_clusters=n_clusters,
                                  mask_ratio=mask_ratio,
                                  train_ratio=train_ratio)
    X1, labels, indices = pipe.cluster_data(file_name)
    return X1, labels, indices

# --------------------------------------------------
# 7. 可视化：聚类结果折线图
# --------------------------------------------------
def plot_cluster_curves(X, labels, day_idx=0, samples_per_class=20, save_path=None):
    N, total_steps, F = X.shape
    steps_per_day = 288
    days = total_steps // steps_per_day
    X_day = X.reshape(N, days, steps_per_day, F)
    daily_flow = X_day[:, day_idx, :, 0]
    time_axis = np.arange(steps_per_day)

    plt.figure(figsize=(12, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    for c in range(labels.max() + 1):
        idx_all = np.where(labels == c)[0]
        idx = np.random.choice(idx_all,
                               size=min(samples_per_class, len(idx_all)),
                               replace=False)
        for i in idx:
            plt.plot(time_axis, daily_flow[i],
                     color=colors[c % len(colors)], alpha=0.6, linewidth=1)

    handles = [plt.Line2D([0], [0], color=colors[c], lw=2)
               for c in range(labels.max() + 1)]
    labels_legend = [f'Cluster {c}' for c in range(labels.max() + 1)]
    plt.legend(handles, labels_legend)

    plt.title(f'Daily Traffic Flow – Day {day_idx}')
    plt.xlabel('Time Step (5-min)')
    plt.ylabel('Flow')
    plt.xlim(0, 287)

    if save_path is None:
        save_path = f'cluster_day{day_idx}.png'
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Figure saved to {save_path}')
    plt.show()

# --------------------------------------------------
# 8. 主入口（带可视化 + 输出均值、变异系数CV、峰谷差）
# --------------------------------------------------
if __name__ == '__main__':
    X1, labels, indices = cluster_data(
        data_dir=r'D:/code/STAEformer-main/data/PEMS03',
        n_clusters=2,
        mask_ratio=0.6,
        train_ratio=0.6                # === 修改：只用前 60% 训练聚类
    )
    print('数据形状:', X1.shape)
    print('标签分布:', np.bincount(labels))

    flow_all = X1[:, :, 0]

    print("\n========== 全部站点整体统计 ==========")
    global_mean = np.mean(flow_all)
    global_std  = np.std(flow_all)
    global_cv   = global_std / global_mean if abs(global_mean) > 1e-6 else 0.0
    global_peak_valley = np.mean(np.max(flow_all, axis=1) - np.min(flow_all, axis=1))
    print(f"全局流量均值: {global_mean:.4f}")
    print(f"全局标准差: {global_std:.4f}")
    print(f"全局变异系数CV: {global_cv:.4f}")
    print(f"全局平均峰谷差: {global_peak_valley:.4f}\n")

    for c in range(labels.max() + 1):
        station_mask = (labels == c)
        station_idx = indices[station_mask].tolist()
        cluster_flow = flow_all[station_mask]

        each_mean = np.mean(cluster_flow, axis=1)
        each_std  = np.std(cluster_flow, axis=1)
        each_pv   = np.max(cluster_flow, axis=1) - np.min(cluster_flow, axis=1)

        cls_mean = np.mean(each_mean)
        cls_std  = np.mean(each_std)
        cls_cv   = cls_std / cls_mean if abs(cls_mean) > 1e-6 else 0.0
        cls_pv_avg = np.mean(each_pv)

        print(f'========== 类别 {c} 统计（{len(station_idx)}个站点） ==========')
        print(f"类别内站点时序均值平均：{cls_mean:.4f}")
        print(f"类别内站点时序标准差平均：{cls_std:.4f}")
        print(f"类别变异系数CV(std/mean)：{cls_cv:.4f}")
        print(f"类别平均峰谷差(max-min)：{cls_pv_avg:.4f}")
        print(f'类别 {c} 包含站点序号:')
        print(station_idx)
        print()

    plot_cluster_curves(X1, labels,
                        day_idx=0,
                        samples_per_class=10,
                        save_path='outputs/PEMS03_10_4.png')


