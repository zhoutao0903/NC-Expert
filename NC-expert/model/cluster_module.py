import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== 全局随机种子固定 =====================
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# 随机掩码生成
# --------------------------------------------------
def rand_mask(x, mask_ratio=0.5):
    mask = torch.rand_like(x) > mask_ratio
    masked_x = x.clone()
    masked_x[~mask] = float('nan')
    return masked_x, mask

# --------------------------------------------------
# CNN+GRU 时序共享编码器（嵌入预测主干）
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
        # x: [B, T]
        x = x.unsqueeze(1)
        feat = self.cnn(x)
        feat = feat.transpose(1, 2)
        _, hn = self.gru(feat)
        z = self.fc(hn.squeeze(0))
        return z

# --------------------------------------------------
# 聚类计算头：DEC软分配、目标分布、重构解码器
# --------------------------------------------------
class ClusterHead(nn.Module):
    def __init__(self, enc_dim, input_dim, n_clusters=2, alpha=1.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        # 二维mask_token 无维度冲突
        self.mask_token = nn.Parameter(torch.zeros(1, input_dim))
        # 时序重构解码器
        self.decoder = nn.Sequential(
            nn.Linear(enc_dim, 256), nn.ReLU(),
            nn.Linear(256, 1024), nn.ReLU(),
            nn.Linear(1024, 4096), nn.ReLU(),
            nn.Linear(4096, input_dim)
        )
        # 可学习聚类中心
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, enc_dim) * 0.1)

    def forward_mask_fill(self, x, mask):
        if mask is not None:
            mask_tok = self.mask_token.expand(x.size(0), -1)
            x = torch.where(mask, x, mask_tok)
        return x

    def reconstruct(self, z):
        return self.decoder(z)

    def soft_assignment(self, z):
        dist = torch.sum((z.unsqueeze(1) - self.cluster_centers.unsqueeze(0)) ** 2, dim=-1)
        q = (1.0 + dist / self.alpha) ** (-(self.alpha + 1) / 2)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

    def target_distribution(self, q):
        weight = q ** 2 / torch.sum(q, dim=0)
        p = weight / torch.sum(weight, dim=1, keepdim=True)
        return p

