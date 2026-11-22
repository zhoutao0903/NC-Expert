import torch
import numpy as np
import os
from lib.utils import print_log, StandardScaler, vrange
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch.utils.data import TensorDataset, DataLoader


# ! X shape: (B, T, N, C)


class Dataset_Custom(Dataset):
    def __init__(self, data_dir, flag='train', args=None, tod=False, dow=False, dom=False, log=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.args = args
        size = [args.seq_len, args.pred_len]
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            # self.label_len = size[1]
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
        self.__read_data__()


    def __read_data__(self):
        # data = np.load(os.path.join(self.data_dir, "data.npz"))["data"].astype(np.float32)#流量、占有率、速度
        data_dir = f"../data/PEMS08"
        file_path = os.path.join(data_dir, 'data.npz')
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # 加载数据
        data = np.load(file_path)
        data = data['data']  # 假设数据存储在'data'键中
        data = data.astype(np.float32)
        # data = data.transpose(1,0,2)
        print("data.shape:",data.shape)

        print(data[:,:,0].shape) #(17856, 170)
        data_reshaped = data[:, :, 0].copy()
        features = [0]  #[0,1,2]
        if self.tod:             #"time_of_day"
            features.append(1)
        if self.dow:             #"day_of_wee                  k"
            features.append(2)
        # if dom:
        #     features.append(3)
        data = data[..., features]#对最后一维选择features列，即选择完整3维

        # print(f"之后的", data.shape)  模型的输入的维度torch.Size([64, 12, 170, 3])

        num_train = int(len(data) * 0.6)  # 10713       #(10700, 3) (3567, 3) (3566, 3)
        num_test = int(len(data) * 0.2)  # 3571
        num_vali = len(data) - num_train - num_test  # 3572
        border1s = [0, num_train - self.seq_len,len(data) - num_test - self.seq_len]  # seq_len =12序列长度 定义训练集、验证集和测试集的起始位置。
        border2s = [num_train, num_train + num_vali, len(data)]  # 定义训练集、验证集和测试集的结束位置

        # print(border1s, border2s) #[0, 10701, 14273] [10713, 14285, 17856]
        border1 = border1s[self.set_type]  # 根据当前数据集类型（训练、验证或测试）选择起始位置。
        border2 = border2s[self.set_type]

        self.scaler = StandardScaler(mean=data[border1s[0]:border2s[0],:,0].mean(),std=data[border1s[0]:border2s[0],:,0].std())
        self.data_x = data[border1:border2, :, :].copy()
        self.data_y = data[border1:border2, :, :1].copy()
        self.data_x[..., 0] = self.scaler.transform(self.data_x[..., 0])


    def get_scaler(self):
        return self.scaler

    def __getitem__(self, index):
        s_begin = index #过去的序列起始位置
        s_end = s_begin + self.seq_len #过去的序列结束位置
        r_begin = s_end
        r_end = r_begin + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

def data_provider(args, data_dir, tod=False, dow=False, batch_size=64, log=None):
    # Dataset是数据源，负责加载和预处理单个数据样本。
    train_set = Dataset_Custom(data_dir=data_dir,flag='train',args =args,tod=tod,dow =dow,log =log,)#DataLoader是对 Dataset 的封装，负责将数据分成批次并提供给模型。
    trainset_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,drop_last=True)# num_workers=args.num_workers,  # 使用 4 个子进程加载数据# drop_last=drop_last)# 丢弃最后一个不完整的批次
    test_set = Dataset_Custom(data_dir=data_dir,flag='test',args =args,tod=tod,dow=dow,log=log,)
    testset_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,drop_last=True)
    val_set = Dataset_Custom(data_dir=data_dir, flag='val', args =args, tod=tod, dow=dow,log=log, )
    valset_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,drop_last=True)
    SCALER = train_set.get_scaler()
    return trainset_loader,valset_loader,testset_loader,SCALER

'加入噪声实验'
# import torch
# import numpy as np
# import os
# import random
# from torch.utils.data import Dataset, DataLoader
# # 沿用你项目内部工具
# from lib.utils import print_log, StandardScaler
#
# # ===================== 统一噪声函数模块（完全复用你第一段噪声逻辑） =====================
# def add_random_missing_noise(data_flow, missing_prob=0.0):
#     """零散随机缺失：全局点位缺失概率"""
#     if missing_prob <= 0:
#         return data_flow.copy()
#     T, N = data_flow.shape
#     data_noise = data_flow.copy()
#     final_mask = np.random.random(size=data_flow.shape) < missing_prob
#     data_noise[final_mask] = 0.0
#     return data_noise
#
#
# def add_gaussian_noise(data_flow, noise_std_scale=0.0, low_flow_thresh=5, noise_ratio=1.0, gauss_abs_std=None):
#     """高斯观测噪声：相对缩放 / 绝对标准差两种模式，低流量不加噪防止MAPE爆炸"""
#     if (noise_std_scale <= 0 and gauss_abs_std is None) or noise_ratio <= 0:
#         return data_flow.copy()
#     data_noise = data_flow.copy()
#
#     if gauss_abs_std is not None and gauss_abs_std > 0:
#         std = gauss_abs_std
#     else:
#         global_flow_mean = np.mean(data_flow)
#         std = global_flow_mean * noise_std_scale
#
#     gauss = np.random.normal(loc=0.0, scale=std, size=data_flow.shape)
#     apply_mask = np.random.random(size=data_flow.shape) < noise_ratio
#     valid_mask = data_flow > low_flow_thresh
#     final_mask = apply_mask & valid_mask
#
#     data_noise[final_mask] += gauss[final_mask]
#     data_noise = np.clip(data_noise, 0.0, None)
#     data_noise = np.round(data_noise).astype(np.float32)
#     return data_noise
#
#
# def add_pulse_noise(data_flow, pulse_rate=0.0, pulse_amp=3.0, flow_th=1.0, noise_ratio=1.0):
#     """脉冲故障噪声：流量激增 / 直接归零断流"""
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
#     """连续时间块缺失"""
#     if missing_rate <= 0 or noise_ratio <= 0:
#         return data_flow.copy()
#     T, N = data_flow.shape
#     data_noise = data_flow.copy()
#     total_elements = T * N
#     target_missing_num = int(total_elements * missing_rate)
#     missing_cnt = 0
#
#     node_apply = np.random.random(N) < noise_ratio
#     valid_nodes = np.where(node_apply)[0]
#     if len(valid_nodes) == 0:
#         return data_noise
#
#     while missing_cnt < target_missing_num:
#         n = np.random.choice(valid_nodes)
#         t_start = np.random.randint(0, T - block_len)
#         t_end = t_start + block_len
#         data_noise[t_start:t_end, n] = 0.0
#         missing_cnt += block_len
#     return data_noise
#
#
# def add_noise(data_flow, noise_type="none", noise_param=0.0, block_len=6, noise_ratio=1.0, gauss_abs_std=None):
#     """统一噪声入口"""
#     if noise_type == "none" or noise_param <= 0:
#         return data_flow.copy()
#     if noise_type == "random_missing":
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
#         raise ValueError("noise_type可选: none, random_missing, gaussian, pulse, block_missing")
#
#
# # ===================== 改造后 Dataset_Custom 类（新增全部噪声入参） =====================
# class Dataset_Custom(Dataset):
#     def __init__(self, data_dir, flag='train', args=None, tod=False, dow=False, dom=False, log=None,
#                  noise_type="none", noise_param=0.0, block_len=6, noise_ratio=1.0, gauss_abs_std=None):
#         # 时序长度参数
#         self.args = args
#         size = [args.seq_len, args.pred_len]
#         if size is None:
#             self.seq_len = 24 * 4 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.pred_len = size[1]
#
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]
#
#         self.tod = tod
#         self.dow = dow
#         self.dom = dom
#         self.log = log
#         self.data_dir = data_dir
#
#         # 噪声参数存储
#         self.noise_type = noise_type
#         self.noise_param = noise_param
#         self.block_len = block_len
#         self.noise_ratio = noise_ratio
#         self.gauss_abs_std = gauss_abs_std
#
#         self.__read_data__()
#
#     def __read_data__(self):
#         # 固定读取PEMS07数据
#         data_dir = f"../data/PEMS04"
#         file_path = os.path.join(data_dir, 'data.npz')
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"File not found: {file_path}")
#
#         data = np.load(file_path)
#         data = data['data'].astype(np.float32)
#         print("原始data.shape:", data.shape)  # [T, N, C]
#
#         # 选取特征通道：流量+tod+dow
#         features = [0]
#         if self.tod:
#             features.append(1)
#         if self.dow:
#             features.append(2)
#         data = data[..., features]
#
#         # 数据集划分 6:2:2
#         num_train = int(len(data) * 0.6)
#         num_test = int(len(data) * 0.2)
#         num_vali = len(data) - num_train - num_test
#
#         border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
#         border2s = [num_train, num_train + num_vali, len(data)]
#
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]
#
#         # 【核心】scaler永远基于干净训练集流量计算，不受噪声影响
#         train_clean_flow = data[border1s[0]:border2s[0], :, 0].copy()
#         self.scaler = StandardScaler(mean=train_clean_flow.mean(), std=train_clean_flow.std())
#
#         # 截取当前数据集片段
#         data_slice = data[border1:border2, :, :].copy()
#         flow_raw = data_slice[..., 0]
#
#         # 对流量通道施加噪声，时间特征不动
#         flow_noisy = add_noise(
#             flow_raw,
#             noise_type=self.noise_type,
#             noise_param=self.noise_param,
#             block_len=self.block_len,
#             noise_ratio=self.noise_ratio,
#             gauss_abs_std=self.gauss_abs_std
#         )
#         data_slice[..., 0] = flow_noisy
#
#         # 输入标准化，标签保持原始数值
#         self.data_x = data_slice.copy()
#         self.data_x[..., 0] = self.scaler.transform(self.data_x[..., 0])
#         self.data_y = data_slice[..., :1].copy()
#
#     def get_scaler(self):
#         return self.scaler
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
# # ===================== 改造data_provider：噪声参数全部透传 =====================
# def data_provider(args, data_dir, tod=False, dow=False, batch_size=64, log=None,
#                   noise_type="none", noise_param=0.0, block_len=6, noise_ratio=1.0, gauss_abs_std=None):
#     # 训练集
#     train_set = Dataset_Custom(
#         data_dir=data_dir, flag='train', args=args, tod=tod, dow=dow, log=log,
#         noise_type=noise_type, noise_param=noise_param, block_len=block_len,
#         noise_ratio=noise_ratio, gauss_abs_std=gauss_abs_std
#     )
#     trainset_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
#
#     # 验证集
#     val_set = Dataset_Custom(
#         data_dir=data_dir, flag='val', args=args, tod=tod, dow=dow, log=log,
#         noise_type=noise_type, noise_param=noise_param, block_len=block_len,
#         noise_ratio=noise_ratio, gauss_abs_std=gauss_abs_std
#     )
#     valset_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True)
#
#     # 测试集
#     test_set = Dataset_Custom(
#         data_dir=data_dir, flag='test', args=args, tod=tod, dow=dow, log=log,
#         noise_type=noise_type, noise_param=noise_param, block_len=block_len,
#         noise_ratio=noise_ratio, gauss_abs_std=gauss_abs_std
#     )
#     testset_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)
#
#     SCALER = train_set.get_scaler()
#     return trainset_loader, valset_loader, testset_loader, SCALER

# '连续运行多个数据集实验'
# import torch
# import numpy as np
# import os
# from lib.utils import print_log, StandardScaler
# from torch.utils.data import Dataset, DataLoader
#
#
# # ! X shape: (B, T, N, C)
# class Dataset_Custom(Dataset):
#     def __init__(self, data_dir, flag='train', args=None, tod=False, dow=False, dom=False, log=None):
#         # size [seq_len, pred_len]
#         self.args = args
#         size = [args.seq_len, args.pred_len]
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.pred_len = size[1]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]
#
#         self.tod = tod
#         self.dow = dow
#         self.dom = dom
#         self.log = log
#         self.data_dir = data_dir
#         self.__read_data__()
#
#
#     def __read_data__(self):
#         file_path = os.path.join(self.data_dir, 'data.npz')
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"File not found: {file_path}")
#
#         data = np.load(file_path)
#         data = data['data'].astype(np.float32)
#         print("data.shape:", data.shape)
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
#         self.scaler = StandardScaler(
#             mean=data[border1s[0]:border2s[0],:,0].mean(),
#             std=data[border1s[0]:border2s[0],:,0].std()
#         )
#         self.data_x = data[border1:border2, :, :].copy()
#         self.data_y = data[border1:border2, :, :1].copy()
#         self.data_x[..., 0] = self.scaler.transform(self.data_x[..., 0])
#
#
#     def get_scaler(self):
#         return self.scaler
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
# def data_provider(args, data_dir, tod=False, dow=False, batch_size=64, log=None):
#     train_set = Dataset_Custom(data_dir=data_dir, flag='train', args=args, tod=tod, dow=dow, log=log)
#     trainset_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
#     test_set = Dataset_Custom(data_dir=data_dir, flag='test', args=args, tod=tod, dow=dow, log=log)
#     testset_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)
#     val_set = Dataset_Custom(data_dir=data_dir, flag='val', args=args, tod=tod, dow=dow, log=log)
#     valset_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True)
#     SCALER = train_set.get_scaler()
#     return trainset_loader, valset_loader, testset_loader, SCALER



