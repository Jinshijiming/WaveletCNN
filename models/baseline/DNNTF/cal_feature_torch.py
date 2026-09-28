import torch
import torch.fft
from itertools import combinations_with_replacement
import math


def mixed_moment(signal, k, q):
    """
    计算混合矩
    :param signal: 复数信号张量
    :param k: 阶数
    :param q: 共轭阶数
    :return: 混合矩值
    """
    p = k - q
    # mixed_moments为complex数据
    mixed_moments = torch.mean((signal**p) * (torch.conj(signal)**q))
    return mixed_moments


def calculate_z(r_n, D):
    """
    计算z(n,D)信号
    :param r_n: 输入复数信号
    :param D: 延迟量
    :return: z(n,D)信号
    """
    r_n_shifted_conj = torch.conj(torch.roll(r_n, D))  # 向右移 D 个位置
    r_n_shifted_conj[:D] = 0  # 在头部添加 0，处理边界
    z_n_D = r_n * r_n_shifted_conj
    return z_n_D


def find_local_maxima(arr, num_maxima=3):
    """
    找到数组中的局部最大值
    :param arr: 输入张量
    :param num_maxima: 最大值的数量
    :return: 排序后的最大值
    """
    if len(arr) < num_maxima:
        raise ValueError("Array is too short to find the required number of maxima")

    values, indices = torch.topk(arr, num_maxima)
    return torch.flip(torch.sort(values)[0], [0])


def find_top3_max_spectral_amplitude(data):
    """
    找到频谱振幅的前3个最大值
    :param data: 输入复数信号
    :return: 包含12个特征的字典
    """
    features = {}

    dft_r = torch.fft.fft(data)
    dft_r2 = torch.fft.fft(data**2)
    dft_r4 = torch.fft.fft(data**4)
    dft_r8 = torch.fft.fft(data**8)

    # 结果与论文中有出入，论文中是10*log10，实际计算应该没有*10
    log_dft_r = torch.log10(torch.abs(dft_r))
    log_dft_r2 = torch.log10(torch.abs(dft_r2))
    log_dft_r4 = torch.log10(torch.abs(dft_r4))
    log_dft_r8 = torch.log10(torch.abs(dft_r8))

    features['F1_1'], features['F2_1'], features['F3_1'] = find_local_maxima(log_dft_r)
    features['F1_2'], features['F2_2'], features['F3_2'] = find_local_maxima(log_dft_r2)
    features['F1_4'], features['F2_4'], features['F3_4'] = find_local_maxima(log_dft_r4)
    features['F1_8'], features['F2_8'], features['F3_8'] = find_local_maxima(log_dft_r8)

    return features


def create_feature_names():
    """创建特征名称字典"""
    feature_names = {
        0: "num_inst_Amp", 1: "symmetry_of_spectrum", 2: "psd_max", 3: "inst_abs_amp_std", 4: "inst_abs_freq_std",
        5: "inst_amp_std ", 6: "inst_amp_kurtosis", 7: "inst_freq_kurtosis",
        8: "1st_spect_amp_r", 9: "2st_spect_amp_r", 10: "3st_spect_amp_r",
        11: "1st_spect_amp_r2", 12: "2st_spect_amp_r2", 13: "3st_spect_amp_r2",
        14: "1st_spect_amp_r4", 15: "2st_spect_amp_r4", 16: "3st_spect_amp_r4",
        17: "1st_spect_amp_r8", 18: "2st_spect_amp_r8", 19: "3st_spect_amp_r8"
    }

    # 添加r(n)的特征
    base_idx = 20
    for k, q in [(2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1), (4, 2),
                 (6, 0), (6, 1), (6, 2), (6, 3), (8, 0), (8, 1), (8, 2), (8, 3), (8, 4),
                 (12, 0), (12, 1), (12, 2), (12, 3), (12, 4), (12, 5), (12, 6),
                 (16, 0), (16, 1), (16, 2), (16, 3), (16, 4), (16, 5), (16, 6), (16, 7), (16, 8)]:
        feature_names[base_idx] = f"r(n)@M_{k},{q}"
        base_idx += 1

    # 添加r(n)的累积量特征
    cumulant_orders = [(4, 0), (4, 1), (4, 2), (6, 0), (6, 1), (6, 2), (6, 3),
                       (8, 0), (8, 1), (8, 2), (8, 3), (8, 4)]
    for k, q in cumulant_orders:
        feature_names[base_idx] = f"r(n)@C_{k},{q}"
        base_idx += 1

    # 添加r(n)的标准化累积量特征
    hat_cumulant_orders = [(6, 0), (6, 1), (6, 2), (8, 0), (8, 1), (8, 2), (8, 3), (4, 2)]
    for k, q in hat_cumulant_orders:
        feature_names[base_idx] = f"r(n)@hat_C_{k},{q}"
        base_idx += 1

    # 对z(n,2), z(n,4), z(n,8)重复相同的模式
    for D in [2, 4, 8]:
        # 添加混合矩
        for k, q in [(2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1), (4, 2),
                     (6, 0), (6, 1), (6, 2), (6, 3), (8, 0), (8, 1), (8, 2), (8, 3), (8, 4),
                     (12, 0), (12, 1), (12, 2), (12, 3), (12, 4), (12, 5), (12, 6),
                     (16, 0), (16, 1), (16, 2), (16, 3), (16, 4), (16, 5), (16, 6), (16, 7), (16, 8)]:
            feature_names[base_idx] = f"z(n,{D})@M_{k},{q}"
            base_idx += 1

        # 添加累积量
        for k, q in cumulant_orders:
            feature_names[base_idx] = f"z(n,{D})@C_{k},{q}"
            base_idx += 1

        # 添加标准化累积量
        for k, q in hat_cumulant_orders:
            feature_names[base_idx] = f"z(n,{D})@hat_C_{k},{q}"
            base_idx += 1

    return feature_names


def mixed_moment_batch(signal, k, q):
    """
    批量计算混合矩
    :param signal: 复数信号张量，形状为 (N, L)
    :param k: 阶数
    :param q: 共轭阶数
    :return: 混合矩值，形状为 (N,)
    """
    p = k - q
    mixed_moments = torch.mean((signal**p) * (torch.conj(signal)**q), dim=1)
    return mixed_moments


def calculate_z_batch(r_n, D):
    """
    批量计算z(n,D)信号
    :param r_n: 输入复数信号，形状为 (N, L)
    :param D: 延迟量
    :return: z(n,D)信号，形状为 (N, L)
    """
    r_n_shifted_conj = torch.conj(torch.roll(r_n, D, dims=1))
    r_n_shifted_conj[:, :D] = 0
    z_n_D = r_n * r_n_shifted_conj
    return z_n_D


def find_top3_max_spectral_amplitude_batch(data):
    """
    批量找到频谱振幅的前3个最大值
    :param data: 输入复数信号，形状为 (N, L)
    :return: 特征张量，形状为 (N, 12)
    """
    N = data.shape[0]
    features = torch.zeros((N, 12), dtype=data.dtype, device=data.device)

    dft_r = torch.fft.fft(data, dim=1)
    dft_r2 = torch.fft.fft(data**2, dim=1)
    dft_r4 = torch.fft.fft(data**4, dim=1)
    dft_r8 = torch.fft.fft(data**8, dim=1)

    log_dft_r = torch.log10(torch.abs(dft_r))
    log_dft_r2 = torch.log10(torch.abs(dft_r2))
    log_dft_r4 = torch.log10(torch.abs(dft_r4))
    log_dft_r8 = torch.log10(torch.abs(dft_r8))

    # 对每个样本找top3
    for i, (arr, name_base) in enumerate([(log_dft_r, 0), (log_dft_r2, 3),
                                          (log_dft_r4, 6), (log_dft_r8, 9)]):
        top3 = torch.topk(arr, 3, dim=1)[0]
        top3_sorted = torch.flip(torch.sort(top3, dim=1)[0], dims=[1])
        features[:, name_base:name_base+3] = top3_sorted

    return features


def calculate_traditional_features_batch(iq_data):
    """
    批量计算传统特征（向量化实现）
    :param iq_data: 输入IQ数据，形状为(N, L, 2)，其中N是批大小，L是信号长度
    :param device: 计算设备 ('cpu' 或 'cuda')
    :return: 特征张量，形状为(N, 228)
    """
    if not torch.is_tensor(iq_data):
        iq_data = torch.tensor(iq_data, dtype=torch.float32)

    N, L, _ = iq_data.shape
    features = torch.zeros((N, 228), dtype=torch.float32, device=iq_data.device)

    # 批量构建复数信号 (N, L)
    I = iq_data[:, :, 0]
    Q = iq_data[:, :, 1]
    r_n = torch.complex(I, Q)

    # 批量归一化
    r_n_mean = torch.mean(r_n, dim=1, keepdim=True)
    r_n_std = torch.std(r_n, dim=1, keepdim=True)
    r_n_norm = (r_n - r_n_mean) / r_n_std

    # 批量计算 K1

    K = torch.sum((0.4 < torch.abs(r_n_norm)) & (torch.abs(r_n_norm) < 0.8), dim=1).float()
    K1 = K / L

    # 批量计算瞬时幅度特征
    a_n = torch.abs(r_n)
    a_n_mean = torch.mean(a_n, dim=1, keepdim=True)
    A_n = a_n / a_n_mean - 1

    # 批量计算瞬时频率特征
    f_n = torch.angle(r_n[:, 1:] * torch.conj(r_n[:, :-1]))
    f_n_mean = torch.mean(f_n, dim=1, keepdim=True)
    F_n = f_n - f_n_mean

    # 批量计算频谱特征
    R_k = torch.fft.fft(r_n, dim=1)
    A_k = torch.fft.fft(A_n, dim=1)

    gamma_max = torch.max(torch.abs(A_k) ** 2, dim=1)[0] / L

    N_half = L // 2
    P_L = torch.sum(torch.abs(R_k[:, :N_half]) ** 2, dim=1)
    P_U = torch.sum(torch.abs(R_k[:, N_half:]) ** 2, dim=1)
    P = (P_L - P_U) / (P_L + P_U)

    # 批量计算频谱振幅特征
    spectral_features = find_top3_max_spectral_amplitude_batch(r_n)

    # 批量计算统计特征
    sigma_a = torch.sqrt(torch.mean(A_n ** 2, dim=1) - torch.mean(A_n, dim=1) ** 2)
    sigma_aa = torch.sqrt(torch.mean(A_n ** 2, dim=1) - torch.mean(a_n / a_n_mean - 1, dim=1) ** 2)
    sigma_af = torch.sqrt(torch.mean(F_n ** 2, dim=1) - torch.mean(torch.abs(F_n), dim=1) ** 2)
    mu_a = torch.mean(A_n ** 4, dim=1) / (torch.mean(A_n ** 2, dim=1) ** 2)
    mu_f = torch.mean(F_n ** 4, dim=1) / (torch.mean(F_n ** 2, dim=1) ** 2)

    # 填充基础特征
    features[:, 0] = K1
    features[:, 1] = P
    features[:, 2] = torch.log10(gamma_max)
    features[:, 3] = torch.log10(sigma_aa)
    features[:, 4] = torch.log10(sigma_af)
    features[:, 5] = torch.log10(sigma_a)
    features[:, 6] = torch.log10(mu_a)
    features[:, 7] = torch.log10(mu_f)
    features[:, 8:20] = spectral_features

    # 批量计算高阶统计特征
    signals = [r_n]  # (N, L)
    for D in [2, 4, 8]:
        z = calculate_z_batch(r_n, D)
        signals.append(z)

    stat_idx = 20
    moment_orders = [(2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1), (4, 2), 
                     (6, 0), (6, 1), (6, 2), (6, 3), (8, 0), (8, 1), (8, 2), (8, 3), (8, 4),
                     (12, 0), (12, 1), (12, 2), (12, 3), (12, 4), (12, 5), (12, 6),
                     (16, 0), (16, 1), (16, 2), (16, 3), (16, 4), (16, 5), (16, 6), (16, 7), (16, 8)]

    for sig in signals:
        # 批量计算混合矩
        mom = {}
        for k, q in moment_orders:
            mom[(k, q)] = mixed_moment_batch(sig, k, q)

        # 混合矩特征
        for k, q in moment_orders:
            features[:, stat_idx] = torch.log10(torch.abs(mom[(k, q)]))
            stat_idx += 1

        # 计算累积量所需的混合矩
        M20 = mom[(2, 0)]
        M21 = mom[(2, 1)]
        M22 = mixed_moment_batch(sig, 2, 2)
        M40 = mom[(4, 0)]
        M41 = mom[(4, 1)]
        M42 = mom[(4, 2)]
        M43 = mixed_moment_batch(sig,4, 3)
        M60 = mom[(6, 0)]
        M61 = mom[(6, 1)]
        M62 = mom[(6, 2)]
        M63 = mom[(6, 3)]
        M80 = mom[(8, 0)]
        M81 = mom[(8, 1)]
        M82 = mom[(8, 2)]
        M83 = mom[(8, 3)]
        M84 = mom[(8, 4)]

        # 批量计算累积量
        C21 = M21
        C40 = M40 - 3*M20**2
        C41 = M41 - 3*M20*M21
        C42 = M42 - M20**2 - 2*M21**2
        C60 = M60 - 15*M20*M40 + 30*M20**3
        C61 = M61 - 5*M21*M40 - 10*M20*M41 + 30*M20**2*M21
        C62 = M62 - 6*M20*M42 - 8*M21*M41 - M22*M40 + 6*M20**2*M22 + 24*M21**2*M20
        C63 = M63 - 3*M20*M43 - 9*M21*M42 - 3*M22*M41 + 18*M20*M21*M22 + 12*M21**3
        C80 = M80 - 28*M60*M20 - 35*M40**2 + 420*M20**2*M40 - 630*M20**4
        C81 = M81 - 21*M20*M61 - 7*M21*M60 - 35*M40*M41 + 210*M20**2*M41 + 210*M20*M21*M40 - 360*M20**3*M21
        C82 = M82 - 15*M40*M42 - 20*M41**2 + 30*M40*M20*M22 + 60*M40*M21**2 + 240*M41*M21*M20 + \
            90*M42*M20**2 - 90*M20**4 - 540*M20**2*M21**2 - 270*M22*M21*M20**2 - 360*M21**3*M20
        C83 = M83 - M63*M20 - 15*M62*M21 - 3*M61*M22 - 5*M43*M40 - 30*M42*M41 + 30*M43 * \
            M20**2 + 90*M42*M21*M20 + 60*M41*M22*M20 + 120*M41*M21**2 + 30*M40*M22*M21
        C84 = M84 - M40**2 - 18*M42**2 - 16*M41**2 - 54*M20**4 - 144*M21**4 - 432*M20**2*M21**2 + \
            12*M40*M20**2 + 96*M41*M21*M20 + 144*M42*M21**2 + 72*M42*M20**2 + 96*M41*M20*M21

        # 累积量特征
        for C in [C40, C41, C42, C60, C61, C62, C63, C80, C81, C82, C83, C84]:
            features[:, stat_idx] = torch.log10(torch.abs(C))
            stat_idx += 1

        # 批量计算标准化累积量
        C_hat_42 = C42 / (C21 ** 2)
        C_hat_60 = C60 / (C42 ** 3)
        C_hat_61 = C61 / (C42 ** 3)
        C_hat_62 = C62 / (C42 ** 3)
        C_hat_80 = C80 / (C42 ** 2)
        C_hat_81 = C81 / (C42 ** 2)
        C_hat_82 = C82 / (C42 ** 2)
        C_hat_83 = C83 / (C42 ** 2)

        # 标准化累积量特征
        for C_hat in [C_hat_60, C_hat_61, C_hat_62, C_hat_80, C_hat_81, C_hat_82, C_hat_83, C_hat_42]:
            features[:, stat_idx] = torch.log10(torch.abs(C_hat))
            stat_idx += 1

    return features


def calculate_traditional_features(iq_data, device='cpu'):
    """
    计算传统特征（单样本）
    :param iq_data: 输入IQ数据，形状为(L, 2)
    :param device: 计算设备 ('cpu' 或 'cuda')
    :return: 特征张量
    """
    if not torch.is_tensor(iq_data):
        iq_data = torch.tensor(iq_data, dtype=torch.float32)

    iq_data = iq_data.to(device)
    features = torch.zeros(228, dtype=torch.float32, device=device)

    I = iq_data[:, 0]
    Q = iq_data[:, 1]
    r_n = torch.complex(I, Q)
    L = len(r_n)
    print(L)

    r_n = (r_n - torch.mean(r_n)) / torch.std(r_n)
    K = torch.sum((0.4 < torch.abs(r_n)) & (torch.abs(r_n) < 0.8))
    K1 = K / L

    a_n = torch.abs(r_n)
    A_n = a_n / torch.mean(a_n) - 1

    f_n = torch.angle(r_n[1:] * torch.conj(r_n[:-1]))
    F_n = f_n - torch.mean(f_n)

    sigma_a = torch.sqrt(torch.mean(A_n ** 2) - torch.mean(A_n) ** 2)
    sigma_aa = torch.sqrt(torch.mean(A_n ** 2) - torch.mean(torch.abs(A_n)) ** 2)
    sigma_af = torch.sqrt(torch.mean(F_n ** 2) - torch.mean(torch.abs(F_n)) ** 2)
    mu_a = torch.mean(A_n ** 4) / (torch.mean(A_n ** 2) ** 2)
    mu_f = torch.mean(F_n ** 4) / (torch.mean(F_n ** 2) ** 2)

    print("number of inst_features", 6)

    R_k = torch.fft.fft(r_n)
    A_k = torch.fft.fft(A_n)

    gamma_max = torch.max(torch.abs(A_k) ** 2) / L

    N_half = L // 2
    P_L = torch.sum(torch.abs(R_k[:N_half]) ** 2)
    P_U = torch.sum(torch.abs(R_k[N_half:]) ** 2)
    P = (P_L - P_U) / (P_L + P_U)

    spectral_features = find_top3_max_spectral_amplitude(r_n)

    print("number of spec_features", len(spectral_features))

    features[0] = K1
    features[1] = P
    features[2] = torch.log10(gamma_max)
    features[3] = torch.log10(sigma_aa)
    features[4] = torch.log10(sigma_af)
    features[5] = torch.log10(sigma_a)
    features[6] = torch.log10(mu_a)
    features[7] = torch.log10(mu_f)

    spec_values = torch.stack(list(spectral_features.values()))
    features[8:20] = spec_values

    stat_features = []
    signals = []

    # 计算高阶累积量 (可选，原代码中被注释掉)
    # signals.append(r_n)
    # for D in [2, 4, 8]:
    #     z = calculate_z(r_n, D)
    #     signals.append(z)

    for sig in signals:
        combos = [(2, 0), (2, 1), (2, 2),
                  (3, 0), (3, 1),
                  (4, 0), (4, 1), (4, 2), (4, 3),
                  (6, 0), (6, 1), (6, 2), (6, 3),
                  (8, 0), (8, 1), (8, 2), (8, 3), (8, 4),
                  (12, 0), (12, 1), (12, 2), (12, 3), (12, 4), (12, 5), (12, 6),
                  (16, 0), (16, 1), (16, 2), (16, 3), (16, 4), (16, 5), (16, 6), (16, 7), (16, 8)]

        # 只计算一次混合矩并保存
        mom = {}
        for k, q in combos:
            mom[(k, q)] = mixed_moment(sig, k, q)

        mixed_moments = [torch.log10(torch.abs(mom[(k, q)])) for (k, q) in combos]

        # 分配后续用到的混合矩
        M20 = mom.get((2, 0))
        M21 = mom.get((2, 1))
        M22 = mom.get((2, 2))
        M40 = mom.get((4, 0))
        M41 = mom.get((4, 1))
        M42 = mom.get((4, 2))
        M43 = mom.get((4, 3))
        M60 = mom.get((6, 0))
        M61 = mom.get((6, 1))
        M62 = mom.get((6, 2))
        M63 = mom.get((6, 3))
        M80 = mom.get((8, 0))
        M81 = mom.get((8, 1))
        M82 = mom.get((8, 2))
        M83 = mom.get((8, 3))
        M84 = mom.get((8, 4))

        # 使用混合矩计算累积量
        C21 = M21
        C40 = M40 - 3*M20**2
        C41 = M41 - 3*M20*M21
        C42 = M42 - M20**2 - 2*M21**2
        C60 = M60 - 15*M20*M40 + 30*M20**3
        C61 = M61 - 5*M21*M40 - 10*M20*M41 + 30*M20**2*M21
        C62 = M62 - 6*M20*M42 - 8*M21*M41 - M22*M40 + 6*M20**2*M22 + 24*M21**2*M20
        C63 = M63 - 3*M20*M43 - 9*M21*M42 - 3*M22*M41 + 18*M20*M21*M22 + 12*M21**3
        C80 = M80 - 28*M60*M20 - 35*M40**2 + 420*M20**2*M40 - 630*M20**4
        C81 = M81 - 21*M20*M61 - 7*M21*M60 - 35*M40*M41 + 210*M20**2*M41 + 210*M20*M21*M40 - 360*M20**3*M21
        C82 = M82 - 15*M40*M42 - 20*M41**2 + 30*M40*M20*M22 + 60*M40*M21**2 + 240*M41*M21*M20 + \
            90*M42*M20**2 - 90*M20**4 - 540*M20**2*M21**2 - 270*M22*M21*M20**2 - 360*M21**3*M20
        C83 = M83 - M63*M20 - 15*M62*M21 - 3*M61*M22 - 5*M43*M40 - 30*M42*M41 + 30*M43 * \
            M20**2 + 90*M42*M21*M20 + 60*M41*M22*M20 + 120*M41*M21**2 + 30*M40*M22*M21
        C84 = M84 - M40**2 - 18*M42**2 - 16*M41**2 - 54*M20**4 - 144*M21**4 - 432*M20**2*M21**2 + \
            12*M40*M20**2 + 96*M41*M21*M20 + 144*M42*M21**2 + 72*M42*M20**2 + 96*M41*M20*M21

        # 计算标准化累积量
        C_hat_42 = C42 / (C21 ** 2)
        C_hat_60 = C60 / (C42 ** 3)
        C_hat_61 = C61 / (C42 ** 3)
        C_hat_62 = C62 / (C42 ** 3)
        C_hat_80 = C80 / (C42 ** 2)
        C_hat_81 = C81 / (C42 ** 2)
        C_hat_82 = C82 / (C42 ** 2)
        C_hat_83 = C83 / (C42 ** 2)

        # 对所有累积量取log10(abs())
        cumulants = [torch.log10(torch.abs(x)) for x in [C40, C41, C42, C60, C61, C62, C63, C80, C81, C82, C83, C84]]
        norm_cumulants = [torch.log10(torch.abs(x)) for x in [C_hat_60, C_hat_61, C_hat_62,
                                                              C_hat_80, C_hat_81, C_hat_82, C_hat_83, C_hat_42]]

        stat_features.extend(mixed_moments + cumulants + norm_cumulants)

    print("number of stat_features", len(stat_features))
    if len(stat_features) > 0:
        stat_tensor = torch.stack(stat_features[:52])
        features[20:(20+52*len(signals))] = stat_tensor

    return features[0:(20+52*len(signals))]


def get_feature_dict(iq_data, device='cpu'):
    """
    计算传统特征并返回特征字典
    :param iq_data: 输入的IQ数据，形状为(N, 2)或(N,)的复数数组/张量
    :param device: 计算设备 ('cpu' 或 'cuda')
    :return: 特征字典
    """
    # 如果输入是numpy数组，先转换为torch张量
    if not torch.is_tensor(iq_data):
        import numpy as np
        if np.issubdtype(iq_data.dtype, np.complexfloating):
            iq_data = np.stack((iq_data.real, iq_data.imag), axis=-1)
        iq_data = torch.tensor(iq_data, dtype=torch.float32)

    # 如果是复数张量，转换为实部虚部分离的形式
    if torch.is_complex(iq_data):
        iq_data = torch.stack((iq_data.real, iq_data.imag), dim=-1)

    print(iq_data.shape)
    features = calculate_traditional_features(iq_data, device=device)
    feature_names = create_feature_names()

    # 将features转回CPU并转为numpy
    features_cpu = features.cpu().numpy()

    idxs = list(feature_names.keys())[:len(features_cpu)]
    names = list(feature_names.values())[:len(features_cpu)]
    values = features_cpu[idxs].tolist()  # 转为python的原生类型
    feat = dict(zip(names, values))
    return feat


if __name__ == "__main__":
    # 检查CUDA是否可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    iq, _, _ = get_iq_from_wav(
        r"C:\Users\ASmys\Desktop\22所项目\IQ分析客户端\IQData\Task=IQANA;Freq=88.1MHz;Span=500kHz;ANBW=500kHz;Rate=800kHz;101501.591.wav")

    feat = get_feature_dict(iq, device=device)
    print(len(feat))
    print(feat)
