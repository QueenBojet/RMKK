import numpy as np
from scipy import signal, interpolate

def angular_resampling(sig_t, speed, sample_freq, max_order=50, order_resolution=0.01, window_type='none'):
    """
    角域重采样 - 将信号从时域转换到角域并计算阶次谱

    Parameters:
    -----------
    sig_t : array_like
        时域信号
    speed : array_like
        转速向量 (转/秒)
    sample_freq : float
        原始采样频率 (Hz)
    max_order : float, optional
        最大分析阶次 (默认: 50)
    order_resolution : float, optional
        阶次分辨率 (默认: 0.01)
    window_type : str, optional
        FFT窗函数 ('hann', 'hamming', 'none')

    Returns:
    --------
    sig_cyc : ndarray
        角域信号
    cyc_fs : float
        角域采样率 (每转采样点数)
    sig_order1 : ndarray
        阶次幅值谱 (单边谱)
    order1 : ndarray
        阶次向量
    dorder : float
        实际阶次分辨率
    """

    # 确保输入为numpy数组
    sig_t = np.asarray(sig_t).flatten()
    speed = np.asarray(speed).flatten()

    # 验证窗函数类型
    if window_type not in ['hann', 'hamming', 'none']:
        raise ValueError(f"Invalid window_type: {window_type}. Must be 'hann', 'hamming', or 'none'")

    # ----------------------------------------------------------------------- #
    # Step 1: 计算累积相位（角位置）
    # ----------------------------------------------------------------------- #
    dt = 1 / sample_freq
    t = np.arange(len(sig_t)) * dt
    cumulative_phase = np.cumsum(speed * dt)  # 累积相位（转数）
    cumulative_phase = cumulative_phase - cumulative_phase[0]  # 从0开始
    total_revolutions = np.max(cumulative_phase)

    # ----------------------------------------------------------------------- #
    # Step 2: 根据阶次分辨率确定分析参数
    # ----------------------------------------------------------------------- #
    # 阶次分辨率 = 1 / 分析长度（转数）
    required_analysis_revs = 1 / order_resolution

    # 检查数据是否足够
    if required_analysis_revs > total_revolutions:
        # warnings.warn(f'数据长度不足以达到目标阶次分辨率 {order_resolution:.4f}，'
        #               f'实际分辨率将为 {1 / total_revolutions:.4f}')
        analysis_revs = total_revolutions
        actual_order_res = 1 / analysis_revs
    else:
        analysis_revs = required_analysis_revs
        actual_order_res = order_resolution

    # 每转采样点数（需满足奈奎斯特采样定理）
    cyc_fs = 2 ** int(np.ceil(np.log2(2 * max_order / actual_order_res)))  # 使用2的幂次方提高FFT效率

    # 确保不低于基本采样要求
    min_speed = np.min(speed)
    cyc_fs_min = int(np.ceil(sample_freq / min_speed))
    cyc_fs = max(cyc_fs, cyc_fs_min)

    # 计算分析样本数
    n_analysis_samples = int(np.round(cyc_fs * analysis_revs))

    # ----------------------------------------------------------------------- #
    # Step 3: 角域重采样 - 等角度间隔重采样
    # ----------------------------------------------------------------------- #
    # 选择数据段（从中间部分提取，避免端点效应）
    if analysis_revs < total_revolutions:
        start_rev = (total_revolutions - analysis_revs) / 2
        end_rev = start_rev + analysis_revs
    else:
        start_rev = 0
        end_rev = analysis_revs

    # 生成等角度间隔的相位点
    constant_phase_intervals = np.linspace(start_rev, end_rev, n_analysis_samples)

    # 通过插值找到对应的时间点
    f_interp = interpolate.interp1d(cumulative_phase, t,
                                    kind='linear',
                                    fill_value='extrapolate')
    times_of_constant_phase_intervals = f_interp(constant_phase_intervals)

    # 对信号进行插值重采样
    f_sig = interpolate.interp1d(t, sig_t, kind='cubic', fill_value=np.nan, bounds_error=False)
    sig_cyc = f_sig(times_of_constant_phase_intervals)

    # 去除NaN值
    valid_idx = ~np.isnan(sig_cyc)
    sig_cyc = sig_cyc[valid_idx]
    n_analysis_samples = len(sig_cyc)

    # ----------------------------------------------------------------------- #
    # Step 4: 计算阶次谱
    # ----------------------------------------------------------------------- #
    # 应用窗函数
    if window_type == 'hann':
        window = signal.windows.hann(n_analysis_samples)
        sig_cyc_windowed = sig_cyc * window
        window_correction = np.mean(window)
    elif window_type == 'hamming':
        window = signal.windows.hamming(n_analysis_samples)
        sig_cyc_windowed = sig_cyc * window
        window_correction = np.mean(window)
    else:  # 'none'
        sig_cyc_windowed = sig_cyc
        window_correction = 1

    # FFT计算阶次谱
    sig_order_complex = np.fft.fft(sig_cyc_windowed)
    sig_order = np.abs(sig_order_complex) * 2 / (n_analysis_samples * window_correction)
    sig_order[0] = sig_order[0] / 2  # DC分量修正

    # ----------------------------------------------------------------------- #
    # Step 5: 生成阶次向量并提取指定范围的单边谱
    # ----------------------------------------------------------------------- #
    # 实际阶次分辨率
    dorder = 1 / analysis_revs

    # 生成完整阶次向量
    full_order = np.arange(n_analysis_samples) * dorder

    # 提取指定最大阶次范围内的单边谱
    max_order_idx_array = np.where(full_order <= max_order)[0]
    if len(max_order_idx_array) == 0:
        max_order_idx = min(int(n_analysis_samples / 2), len(full_order))
    else:
        max_order_idx = max_order_idx_array[-1] + 1  # +1 因为Python切片不包含结束索引

    # 输出结果
    order1 = full_order[:max_order_idx]
    sig_order1 = sig_order[:max_order_idx]

    # 确保输出为列向量（在Python中为1D数组）
    order1 = order1.flatten()
    sig_order1 = sig_order1.flatten()

    return sig_cyc, cyc_fs, sig_order1, order1, dorder

# 使用示例
if __name__ == "__main__":
    # 生成测试信号
    fs = 10000  # 采样频率 Hz
    t = np.arange(0, 10, 1 / fs)  # 10秒信号

    # 变转速信号
    speed = 30 + 5 * np.sin(2 * np.pi * 0.1 * t)  # 基础转速30转/秒，有0.1Hz的速度波动

    # 生成包含多个阶次成分的信号
    sig = (np.sin(2 * np.pi * 1 * np.cumsum(speed) / fs) +  # 1阶成分
           0.5 * np.sin(2 * np.pi * 2 * np.cumsum(speed) / fs) +  # 2阶成分
           0.3 * np.sin(2 * np.pi * 3.5 * np.cumsum(speed) / fs))  # 3.5阶成分

    # 角域重采样
    sig_cyc, cyc_fs, sig_order1, order1, dorder = angular_resampling(
        sig, speed, fs,
        max_order=20,
        order_resolution=0.01,
        window_type='hann'
    )

    print(f"角域采样率: {cyc_fs} 样本/转")
    print(f"阶次分辨率: {dorder:.4f}")
    print(f"分析阶次范围: 0 - {order1[-1]:.2f}")