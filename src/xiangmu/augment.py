"""
水声数据增强模块 (Enhanced Version with Visualization)
======================================================
【模块功能概述】
本模块是一个面向水声/水下声学信号的音频数据增强工具，主要用于深度学习中的数据扩增。
支持单文件处理和批量目录处理，并生成增强样本溯源记录。

支持的增强方法（共9种）：
  1. 高斯噪声注入 (Gaussian Noise)       - 按指定信噪比添加高斯白噪声
  2. 自定义噪声混合 (Custom Noise Mixing) - 将指定噪声文件按 SNR 混合到音频中
  3. 时间拉伸 (Time Stretching)           - 改变音频速度但不改变音高
  4. 音高偏移 (Pitch Shifting)            - 改变音高但不改变时长
  5. 音量扰动 (Volume Perturbation)       - 随机缩放音量幅度
  6. 混响模拟 (Reverberation)             - 基于梳状滤波器的简单混响效果
  7. SpecAugment 频率掩码 (Freq Mask)     - 在 Mel 谱图上随机遮挡频率带
  8. SpecAugment 时间掩码 (Time Mask)     - 在 Mel 谱图上随机遮挡时间段
  9. 随机裁剪/填充 (Random Crop/Pad)      - 将音频裁剪或填充到目标长度

可视化功能：
  支持生成原始信号和增强信号的波形图、频谱图、功率谱图、梅尔谱图，
  以及原始 vs 增强的 2×5 对比网格图。

核心流程：
  加载音频 → 重采样 → 统一时长 → 随机应用增强流水线 → 保存音频 + 可视化 + 溯源记录

命令行使用示例：
  # 单文件增强
  python augment.py --audio_path /path/to/audio.wav --snr 15 --output_dir ./output
  
  # 批量增强目录下所有文件
  python augment.py --input_dir /path/to/dataset --output_dir ./augmented --num_augmented 5
  
  # 使用背景噪声库
  python augment.py --input_dir ./data --noise_dir ./noise_lib --bg_noise_prob 0.4 \\
      --output_dir ./augmented
"""

import json
import os
import sys
import argparse
import logging
import random
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field, asdict

import librosa
import numpy as np
import soundfile as sf

# ============================================================
# [1] 可视化相关导入
# ============================================================
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适用于服务器环境
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# ============================================================
# [2] 数据结构定义
# ============================================================

@dataclass
class AugmentationRecord:
    """
    增强样本溯源记录 (Provenance Record)
    
    【功能】
    记录每个增强样本的来源信息，包括原始文件路径、增强后文件路径，
    以及所有应用的增强操作及其参数。用于实验可追溯性和结果复现。
    
    【字段说明】
    original_file:  原始音频文件的路径
    augmented_file: 增强后生成的音频文件路径
    augmentations:  应用的增强操作列表，每个元素为 {"type": str, "params": dict}
    
    【方法】
    to_dict(): 将记录转换为字典格式，用于 JSON 序列化保存
    """
    original_file: str
    augmented_file: str
    augmentations: List[Dict[str, object]] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """将溯源记录转换为字典格式，用于 JSON 序列化保存"""
        return {
            'original_file': self.original_file,
            'augmented_file': self.augmented_file,
            'augmentations': self.augmentations
        }


@dataclass
class AugmentationConfig:
    """
    增强配置 (Augmentation Configuration)
    
    【功能】
    集中管理所有数据增强方法的参数配置。每个增强方法都有对应的参数和独立的应用概率，
    通过概率控制来决定是否在每次增强中应用该方法，实现随机组合增强。
    
    【字段说明 - 按功能分组】
    
    --- 噪声增强 ---
    snr_db:              高斯噪声信噪比(dB)，None 表示不启用该增强
    snr_prob:            高斯噪声应用概率 (0~1)
    noise_file:          自定义噪声文件路径，None 表示不启用
    noise_snr:           自定义噪声混合信噪比(dB)
    noise_prob:          自定义噪声应用概率 (0~1)
    noise_dir:           背景噪声库目录路径，None 表示不启用
    bg_noise_snr_range:  背景噪声信噪比范围 (min, max)，在此范围内随机选取
    bg_noise_prob:       背景噪声应用概率 (0~1)
    
    --- 时间/频率增强 ---
    time_stretch_factors: 时间拉伸系数列表，<1 加速，>1 减速
    time_stretch_prob:    时间拉伸应用概率 (0~1)
    pitch_shift_steps:    音高偏移半音数列表，负数降低，正数升高
    pitch_shift_prob:     音高偏移应用概率 (0~1)
    
    --- 音量/混响增强 ---
    volume_range:         音量缩放范围 (min, max)，在此范围内随机选取
    volume_prob:          音量扰动应用概率 (0~1)
    reverb_decay:         混响衰减系数 (0~1)，越大混响越强
    reverb_delay:         混响延迟时间(秒)
    reverb_prob:          混响应用概率 (0~1)
    
    --- SpecAugment (谱图掩码) ---
    freq_mask_max:        最大频率掩码宽度 (mel bin 数)
    freq_mask_count:      频率掩码数量
    freq_mask_prob:       频率掩码应用概率 (0~1)
    time_mask_max:        最大时间掩码宽度 (帧数)
    time_mask_count:      时间掩码数量
    time_mask_prob:       时间掩码应用概率 (0~1)
    
    --- 裁剪/填充 ---
    random_crop_prob:     随机裁剪应用概率，0 表示不启用
    
    --- 通用参数 ---
    target_sr:            目标采样率 (Hz)，所有音频会被重采样到此采样率
    duration:             音频标准长度(秒)，所有音频会被统一到此时长
    output_dir:           增强后音频的输出目录
    provenance_file:      溯源记录 JSON 文件名
    num_augmented_per_file: 每个原始文件生成的增强样本数量
    
    --- 可视化参数 ---
    enable_visualization: 是否启用可视化输出
    vis_dir:              可视化图像输出目录
    vis_dpi:              可视化图像 DPI (分辨率)
    vis_format:           可视化图像格式 (png/jpg/pdf/svg)
    """
    # 高斯噪声
    snr_db: Optional[float] = None          # 信噪比(dB)，None 表示不启用
    snr_prob: float = 0.5                   # 应用概率
    
    # 自定义噪声文件混合
    noise_file: Optional[str] = None        # 噪声文件路径，None 表示不启用
    noise_snr: float = 15.0                 # 混合信噪比(dB)
    noise_prob: float = 0.3                 # 应用概率
    
    # 背景噪声库混合 (从目录中随机选取噪声文件)
    noise_dir: Optional[str] = None         # 噪声文件目录
    bg_noise_snr_range: Tuple[float, float] = (5.0, 20.0)  # 信噪比范围
    bg_noise_prob: float = 0.3              # 应用概率
    
    # 时间拉伸
    time_stretch_factors: List[float] = field(default_factory=lambda: [0.85, 0.9, 1.1, 1.15])
    time_stretch_prob: float = 0.3          # 应用概率
    
    # 音高偏移 (半音)
    pitch_shift_steps: List[int] = field(default_factory=lambda: [-3, -2, -1, 1, 2, 3])
    pitch_shift_prob: float = 0.3           # 应用概率
    
    # 音量扰动
    volume_range: Tuple[float, float] = (0.5, 1.5)  # 音量缩放范围
    volume_prob: float = 0.5                # 应用概率
    
    # 混响模拟
    reverb_decay: float = 0.3               # 混响衰减系数
    reverb_delay: float = 0.05              # 混响延迟(秒)
    reverb_prob: float = 0.2                # 应用概率
    
    # SpecAugment (频率掩码)
    freq_mask_max: int = 20                 # 最大频率掩码宽度 (mel bin 数)
    freq_mask_count: int = 2                # 频率掩码数量
    freq_mask_prob: float = 0.3             # 应用概率
    
    # SpecAugment (时间掩码)
    time_mask_max: int = 50                 # 最大时间掩码宽度 (帧数)
    time_mask_count: int = 2                # 时间掩码数量
    time_mask_prob: float = 0.3             # 应用概率
    
    # 随机裁剪/填充
    random_crop_prob: float = 0.0           # 应用概率 (0 表示不启用)
    
    # 通用参数
    target_sr: int = 16000                  # 目标采样率
    duration: float = 3.0                   # 音频标准长度(秒)
    output_dir: str = "./augmented_output"  # 增强输出目录
    provenance_file: str = "provenance.json" # 溯源记录文件
    num_augmented_per_file: int = 3         # 每个文件生成的增强样本数
    
    # 可视化参数
    enable_visualization: bool = True       # 是否启用可视化
    vis_dir: str = "./visualizations"       # 可视化输出目录
    vis_dpi: int = 150                      # 可视化图像 DPI
    vis_format: str = "png"                 # 可视化图像格式 (png, jpg, pdf, svg)


# ============================================================
# [3] 可视化函数 - 波形图 / 频谱图 / 功率谱图 / 梅尔谱图 / 对比图
# ============================================================

# [3.1] 波形图绘制
def plot_waveform(
    waveform: np.ndarray,
    sr: int,
    title: str = "Waveform",
    figsize: Tuple[int, int] = (10, 3),
    color: str = 'steelblue'
) -> Figure:
    """
    绘制波形图 (Waveform Plot)
    
    【功能】
    绘制音频信号的时域波形图，横轴为时间(秒)，纵轴为幅度。
    用于直观观察音频信号的幅度变化、能量分布和时序结构。
    
    Args:
        waveform: 音频波形 [n_samples]
        sr: 采样率
        title: 图表标题
        figsize: 图像尺寸 (宽, 高) 英寸
        color: 波形线条颜色
    
    Returns:
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    time_axis = np.linspace(0, len(waveform) / sr, len(waveform))
    ax.plot(time_axis, waveform, color=color, linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# [3.2] 频谱图绘制 (STFT 线性频率)
def plot_spectrogram(
    waveform: np.ndarray,
    sr: int,
    title: str = "Spectrogram",
    figsize: Tuple[int, int] = (10, 4),
    n_fft: int = 2048,
    hop_length: int = 512,
    cmap: str = 'viridis'
) -> Figure:
    """
    绘制频谱图 (Spectrogram Plot, 线性频率)
    
    【功能】
    通过短时傅里叶变换(STFT)计算并绘制音频信号的频谱图，
    横轴为时间，纵轴为线性频率，颜色表示能量强度(dB)。
    用于观察音频信号的时频分布特性。
    
    Args:
        waveform: 音频波形
        sr: 采样率
        title: 图表标题
        figsize: 图像尺寸
        n_fft: FFT 窗口大小 (默认 2048)
        hop_length: 帧移 (默认 512)
        cmap: 颜色映射方案
    
    Returns:
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)),
        ref=np.max
    )
    img = librosa.display.specshow(
        D, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='linear',
        cmap=cmap, ax=ax
    )
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title(title)
    plt.tight_layout()
    return fig


# [3.3] 功率谱图绘制 (Welch 方法)
def plot_power_spectrum(
    waveform: np.ndarray,
    sr: int,
    title: str = "Power Spectrum",
    figsize: Tuple[int, int] = (10, 3),
    n_fft: int = 2048,
    hop_length: int = 512,
    color: str = 'crimson'
) -> Figure:
    """
    绘制功率谱图 (Power Spectrum Plot)
    
    【功能】
    使用 Welch 方法计算音频信号的功率谱密度(PSD)，
    横轴为频率(Hz)，纵轴为功率谱密度(V²/Hz)，采用对数坐标。
    用于分析音频信号在各频率上的能量分布。
    
    Args:
        waveform: 音频波形
        sr: 采样率
        title: 图表标题
        figsize: 图像尺寸
        n_fft: FFT 窗口大小
        hop_length: 帧移
        color: 线条颜色
    
    Returns:
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    # 使用 Welch 方法计算功率谱密度
    from scipy import signal as scipy_signal
    f, Pxx = scipy_signal.welch(waveform, fs=sr, nperseg=n_fft, noverlap=hop_length)
    ax.semilogy(f, Pxx, color=color, linewidth=0.8)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density (V²/Hz)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# [3.4] 梅尔谱图绘制 (Mel 频率刻度)
def plot_mel_spectrogram(
    waveform: np.ndarray,
    sr: int,
    title: str = "Mel Spectrogram",
    figsize: Tuple[int, int] = (10, 4),
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    cmap: str = 'magma'
) -> Figure:
    """
    绘制梅尔谱图 (Mel Spectrogram Plot)
    
    【功能】
    计算并绘制音频信号的梅尔频谱图，将线性频率映射到梅尔刻度，
    更符合人耳听觉特性。横轴为时间，纵轴为梅尔频率。
    常用于音频分类、语音识别等深度学习任务的输入特征可视化。
    
    Args:
        waveform: 音频波形
        sr: 采样率
        title: 图表标题
        figsize: 图像尺寸
        n_fft: FFT 窗口大小
        hop_length: 帧移
        n_mels: Mel 滤波器组数量 (默认 128)
        cmap: 颜色映射方案
    
    Returns:
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    mel_spec = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(
        mel_spec_db, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='mel',
        cmap=cmap, ax=ax
    )
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title(title)
    plt.tight_layout()
    return fig


# [3.5] 原始 vs 增强 2×5 对比网格图
def plot_comparison_grid(
    original_waveform: np.ndarray,
    augmented_waveform: np.ndarray,
    sr: int,
    original_title: str = "Original",
    augmented_title: str = "Augmented",
    save_path: Optional[str] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    dpi: int = 150
) -> Figure:
    """
    绘制原始信号与增强信号的对比网格图 (2行5列)
    
    【功能】
    生成一个 2×5 的对比网格图，直观展示原始信号与增强信号在各维度的差异。
    第1行: 原始信号的波形、频谱、功率谱、梅尔谱(带颜色条)
    第2行: 增强信号的对应图
    
    用于评估增强效果，对比增强前后音频在时域、频域和感知域的变化。
    
    Args:
        original_waveform: 原始波形
        augmented_waveform: 增强后波形
        sr: 采样率
        original_title: 原始信号标题前缀
        augmented_title: 增强信号标题前缀
        save_path: 保存路径 (可选，提供则自动保存并关闭图像)
        n_fft: FFT 窗口大小
        hop_length: 帧移
        n_mels: Mel 滤波器组数量
        dpi: 图像 DPI
    
    Returns:
        matplotlib Figure 对象
    """
    fig, axes = plt.subplots(2, 5, figsize=(24, 8))
    
    # ---- 计算公共数据 ----
    # 原始信号
    orig_stft = np.abs(librosa.stft(original_waveform, n_fft=n_fft, hop_length=hop_length))
    orig_spec_db = librosa.amplitude_to_db(orig_stft, ref=np.max)
    orig_mel = librosa.feature.melspectrogram(
        y=original_waveform, sr=sr, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels
    )
    orig_mel_db = librosa.power_to_db(orig_mel, ref=np.max)
    
    # 增强信号
    aug_stft = np.abs(librosa.stft(augmented_waveform, n_fft=n_fft, hop_length=hop_length))
    aug_spec_db = librosa.amplitude_to_db(aug_stft, ref=np.max)
    aug_mel = librosa.feature.melspectrogram(
        y=augmented_waveform, sr=sr, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels
    )
    aug_mel_db = librosa.power_to_db(aug_mel, ref=np.max)
    
    # 功率谱 (Welch)
    from scipy import signal as scipy_signal
    f_orig, Pxx_orig = scipy_signal.welch(original_waveform, fs=sr, nperseg=n_fft, noverlap=hop_length)
    f_aug, Pxx_aug = scipy_signal.welch(augmented_waveform, fs=sr, nperseg=n_fft, noverlap=hop_length)
    
    # ---- 第1行: 原始信号 ----
    time_orig = np.linspace(0, len(original_waveform) / sr, len(original_waveform))
    
    # (0,0) 波形图
    axes[0, 0].plot(time_orig, original_waveform, color='steelblue', linewidth=0.5)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title(f'{original_title} - Waveform')
    axes[0, 0].grid(True, alpha=0.3)
    
    # (0,1) 频谱图
    librosa.display.specshow(
        orig_spec_db, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='linear',
        cmap='viridis', ax=axes[0, 1]
    )
    axes[0, 1].set_title(f'{original_title} - Spectrogram')
    
    # (0,2) 功率谱图
    axes[0, 2].semilogy(f_orig, Pxx_orig, color='crimson', linewidth=0.8)
    axes[0, 2].set_xlabel('Frequency (Hz)')
    axes[0, 2].set_ylabel('PSD (V²/Hz)')
    axes[0, 2].set_title(f'{original_title} - Power Spectrum')
    axes[0, 2].grid(True, alpha=0.3)
    
    # (0,3) 梅尔谱图 + 颜色条
    img0 = librosa.display.specshow(
        orig_mel_db, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='mel',
        cmap='magma', ax=axes[0, 3]
    )
    fig.colorbar(img0, ax=axes[0, 3], format='%+2.0f dB')
    axes[0, 3].set_title(f'{original_title} - Mel Spectrogram')
    
    # (0,4) 显示原始信号统计信息（替代原重复的梅尔谱图）
    axes[0, 4].axis('off')
    axes[0, 4].text(0.1, 0.7, 'Original Stats:', fontsize=10, fontweight='bold',
                    transform=axes[0, 4].transAxes)
    axes[0, 4].text(0.1, 0.5, f'  Length: {len(original_waveform)/sr:.2f}s', fontsize=9,
                    transform=axes[0, 4].transAxes)
    axes[0, 4].text(0.1, 0.35, f'  Max: {np.max(np.abs(original_waveform)):.4f}', fontsize=9,
                    transform=axes[0, 4].transAxes)
    axes[0, 4].text(0.1, 0.2, f'  RMS: {np.sqrt(np.mean(original_waveform**2)):.4f}', fontsize=9,
                    transform=axes[0, 4].transAxes)
    
    # ---- 第2行: 增强信号 ----
    time_aug = np.linspace(0, len(augmented_waveform) / sr, len(augmented_waveform))
    
    # (1,0) 波形图
    axes[1, 0].plot(time_aug, augmented_waveform, color='darkorange', linewidth=0.5)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].set_title(f'{augmented_title} - Waveform')
    axes[1, 0].grid(True, alpha=0.3)
    
    # (1,1) 频谱图
    librosa.display.specshow(
        aug_spec_db, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='linear',
        cmap='viridis', ax=axes[1, 1]
    )
    axes[1, 1].set_title(f'{augmented_title} - Spectrogram')
    
    # (1,2) 功率谱图
    axes[1, 2].semilogy(f_aug, Pxx_aug, color='darkorange', linewidth=0.8)
    axes[1, 2].set_xlabel('Frequency (Hz)')
    axes[1, 2].set_ylabel('PSD (V²/Hz)')
    axes[1, 2].set_title(f'{augmented_title} - Power Spectrum')
    axes[1, 2].grid(True, alpha=0.3)
    
    # (1,3) 梅尔谱图 + 颜色条
    img1 = librosa.display.specshow(
        aug_mel_db, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='mel',
        cmap='magma', ax=axes[1, 3]
    )
    fig.colorbar(img1, ax=axes[1, 3], format='%+2.0f dB')
    axes[1, 3].set_title(f'{augmented_title} - Mel Spectrogram')
    
    # (1,4) 显示增强信号统计信息
    axes[1, 4].axis('off')
    axes[1, 4].text(0.1, 0.7, 'Augmented Stats:', fontsize=10, fontweight='bold',
                    transform=axes[1, 4].transAxes)
    axes[1, 4].text(0.1, 0.5, f'  Length: {len(augmented_waveform)/sr:.2f}s', fontsize=9,
                    transform=axes[1, 4].transAxes)
    axes[1, 4].text(0.1, 0.35, f'  Max: {np.max(np.abs(augmented_waveform)):.4f}', fontsize=9,
                    transform=axes[1, 4].transAxes)
    axes[1, 4].text(0.1, 0.2, f'  RMS: {np.sqrt(np.mean(augmented_waveform**2)):.4f}', fontsize=9,
                    transform=axes[1, 4].transAxes)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        logger = logging.getLogger(__name__)
        logger.info(f"  [可视化-对比图] → {save_path}")
    
    return fig


# [3.6] 批量保存单个音频的4种可视化图像
def save_individual_visualizations(
    waveform: np.ndarray,
    sr: int,
    output_dir: str,
    base_name: str,
    prefix: str = "",
    dpi: int = 150,
    fmt: str = "png"
) -> Dict[str, str]:
    """
    保存单个音频的各类可视化图像到指定目录
    
    【功能】
    对一个音频信号生成并保存4种可视化图像到指定目录：
      1. 波形图 (waveform)
      2. 频谱图 (spectrogram)
      3. 功率谱图 (power_spectrum)
      4. 梅尔谱图 (mel_spectrogram)
    
    用于批量生成可视化文件，供后续分析或报告使用。
    
    Args:
        waveform: 音频波形
        sr: 采样率
        output_dir: 输出目录
        base_name: 文件名基础名
        prefix: 文件名前缀 (如 "original", "aug_1")
        dpi: 图像 DPI
        fmt: 图像格式
    
    Returns:
        保存的文件路径字典，键为 'waveform'/'spectrogram'/'power_spectrum'/'mel_spectrogram'
    """
    logger = logging.getLogger(__name__)
    os.makedirs(output_dir, exist_ok=True)
    
    prefix_str = f"{prefix}_" if prefix else ""
    saved_paths = {}
    
    # 1. 波形图
    fig_wave = plot_waveform(waveform, sr, title=f"{prefix} Waveform")
    wave_path = os.path.join(output_dir, f"{prefix_str}{base_name}_waveform.{fmt}")
    fig_wave.savefig(wave_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig_wave)
    saved_paths['waveform'] = wave_path
    logger.info(f"  [可视化-波形图] → {wave_path}")
    
    # 2. 频谱图
    fig_spec = plot_spectrogram(waveform, sr, title=f"{prefix} Spectrogram")
    spec_path = os.path.join(output_dir, f"{prefix_str}{base_name}_spectrogram.{fmt}")
    fig_spec.savefig(spec_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig_spec)
    saved_paths['spectrogram'] = spec_path
    logger.info(f"  [可视化-频谱图] → {spec_path}")
    
    # 3. 功率谱图
    fig_power = plot_power_spectrum(waveform, sr, title=f"{prefix} Power Spectrum")
    power_path = os.path.join(output_dir, f"{prefix_str}{base_name}_power_spectrum.{fmt}")
    fig_power.savefig(power_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig_power)
    saved_paths['power_spectrum'] = power_path
    logger.info(f"  [可视化-功率谱图] → {power_path}")
    
    # 4. 梅尔谱图
    fig_mel = plot_mel_spectrogram(waveform, sr, title=f"{prefix} Mel Spectrogram")
    mel_path = os.path.join(output_dir, f"{prefix_str}{base_name}_mel_spectrogram.{fmt}")
    fig_mel.savefig(mel_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig_mel)
    saved_paths['mel_spectrogram'] = mel_path
    logger.info(f"  [可视化-梅尔谱图] → {mel_path}")
    
    return saved_paths


# ============================================================
# [4] 增强函数 - 9 种数据增强方法
# ============================================================

# [4.1] 高斯噪声注入
def add_gaussian_noise(waveform: np.ndarray, snr_db: float) -> np.ndarray:
    """
    添加高斯白噪声 (Gaussian Noise Injection)
    
    【功能】
    向音频信号添加高斯白噪声，噪声功率由目标信噪比(SNR)控制。
    SNR 值越小，噪声越强，增强难度越大。
    用于模拟水下环境中的环境噪声，提高模型鲁棒性。
    
    算法：
      1. 计算信号功率 P_signal = mean(waveform²)
      2. 计算目标噪声功率 P_noise = P_signal / 10^(SNR/10)
      3. 生成 N(0, sqrt(P_noise)) 的高斯噪声
      4. 信号 + 噪声
    
    Args:
        waveform: 输入波形 [n_samples]
        snr_db: 信噪比 (dB)，值越小噪声越强
    
    Returns:
        加噪后的波形
    """
    signal_power = np.mean(waveform ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(noise_power), waveform.shape)
    return waveform + noise


# [4.2] 自定义噪声混合
def mix_noise_file(waveform: np.ndarray, noise_waveform: np.ndarray, snr_db: float) -> np.ndarray:
    """
    将自定义噪声文件混合到音频中 (Custom Noise Mixing)
    
    【功能】
    将指定的噪声文件（如发动机噪声、水流声等）按目标 SNR 混合到音频中。
    如果噪声文件比音频短，会自动循环填充至等长。
    用于模拟特定类型的水下噪声环境。
    
    算法：
      1. 确保噪声与信号长度一致（循环填充或截断）
      2. 按 SNR 调整噪声能量：scale = sqrt(P_target / P_noise)
      3. 信号 + 缩放后的噪声
    
    Args:
        waveform: 输入波形
        noise_waveform: 噪声波形
        snr_db: 混合信噪比 (dB)
    
    Returns:
        混合后的波形
    """
    # 确保噪声与信号长度一致
    if len(noise_waveform) < len(waveform):
        # 循环填充
        repeats = int(np.ceil(len(waveform) / len(noise_waveform)))
        noise_waveform = np.tile(noise_waveform, repeats)
    noise_waveform = noise_waveform[:len(waveform)]
    
    # 按 SNR 调整噪声能量
    signal_power = np.mean(waveform ** 2)
    noise_power = np.mean(noise_waveform ** 2)
    if noise_power > 0:
        target_noise_power = signal_power / (10 ** (snr_db / 10.0))
        scale = np.sqrt(target_noise_power / noise_power)
        noise_waveform = noise_waveform * scale
    
    return waveform + noise_waveform


# [4.3] 时间拉伸 (变速度不变音高)
def time_stretch(waveform: np.ndarray, rate: float, sr: int) -> np.ndarray:
    """
    时间拉伸 (Time Stretching)
    
    【功能】
    改变音频的播放速度但不改变音高。
    rate < 1 时加速（时长缩短），rate > 1 时减速（时长延长）。
    用于模拟不同速度的水声信号，增加模型对时间变化的鲁棒性。
    
    实现：基于 librosa.effects.time_stretch，使用相位声码器(Phase Vocoder)技术。
    
    Args:
        waveform: 输入波形
        rate: 拉伸系数 (<1 加速, >1 减速)
        sr: 采样率
    
    Returns:
        拉伸后的波形
    """
    return librosa.effects.time_stretch(y=waveform, rate=rate)


# [4.4] 音高偏移 (变音高不变时长)
def pitch_shift(waveform: np.ndarray, sr: int, n_steps: int) -> np.ndarray:
    """
    音高偏移 (Pitch Shifting)
    
    【功能】
    改变音频的音高但不改变时长（播放速度）。
    n_steps > 0 时音高升高，n_steps < 0 时音高降低。
    用于模拟不同频率特性的水声信号，增加模型对频率变化的鲁棒性。
    
    实现：基于 librosa.effects.pitch_shift，结合时间拉伸+重采样技术。
    
    Args:
        waveform: 输入波形
        sr: 采样率
        n_steps: 半音偏移量 (正数升高, 负数降低)
    
    Returns:
        偏移后的波形
    """
    return librosa.effects.pitch_shift(y=waveform, sr=sr, n_steps=n_steps)


# [4.5] 音量扰动
def volume_perturbation(waveform: np.ndarray, factor: float) -> np.ndarray:
    """
    音量扰动 (Volume Perturbation)
    
    【功能】
    随机缩放音频信号的幅度，改变音量大小。
    factor > 1 时放大音量，factor < 1 时减小音量。
    用于模拟不同距离、不同深度下的水声信号幅度变化。
    
    Args:
        waveform: 输入波形
        factor: 音量缩放因子
    
    Returns:
        调整后的波形
    """
    return waveform * factor


# [4.6] 混响模拟 (梳状滤波器)
def add_reverberation(waveform: np.ndarray, sr: int, decay: float = 0.3, delay: float = 0.05) -> np.ndarray:
    """
    添加简单混响效果 (Reverberation Simulation)
    
    【功能】
    基于梳状滤波器(Comb Filter)的简单混响模拟。
    通过将延迟后的信号以一定衰减系数叠加回原信号，模拟声音在水下
    多路径传播产生的混响效果。最后进行归一化防止削波。
    
    用于模拟水下声场中的混响环境，提高模型在混响条件下的泛化能力。
    
    Args:
        waveform: 输入波形
        sr: 采样率
        decay: 衰减系数 (0~1), 控制混响强度, 越大混响越强
        delay: 延迟时间 (秒), 控制混响延迟
    
    Returns:
        添加混响后的波形
    """
    delay_samples = int(sr * delay)
    output = np.copy(waveform)
    for i in range(delay_samples, len(waveform)):
        output[i] += decay * output[i - delay_samples]
    # 归一化防止削波
    max_val = np.max(np.abs(output))
    if max_val > 0:
        output = output / max_val
    return output


# [4.7] SpecAugment 频率掩码
def spec_augment_freq_mask(mel_spec: np.ndarray, max_mask_width: int = 20, mask_count: int = 2) -> np.ndarray:
    """
    SpecAugment 频率掩码 (Frequency Masking)
    
    【功能】
    在梅尔频谱图上随机遮挡若干频率带，将选定区域的值置为零。
    用于模拟频率选择性衰减或频带缺失的水声传播条件，
    提高模型在频率信息不完整时的泛化能力。
    
    Args:
        mel_spec: 梅尔频谱图 [n_mels, n_frames]
        max_mask_width: 最大掩码宽度 (mel bin 数)
        mask_count: 掩码数量
    
    Returns:
        掩码后的梅尔频谱图
    """
    mel_spec = mel_spec.copy()
    n_mels = mel_spec.shape[0]
    for _ in range(mask_count):
        mask_width = random.randint(1, max_mask_width)
        mask_start = random.randint(0, n_mels - mask_width)
        mel_spec[mask_start:mask_start + mask_width, :] = 0
    return mel_spec


# [4.8] SpecAugment 时间掩码
def spec_augment_time_mask(mel_spec: np.ndarray, max_mask_width: int = 50, mask_count: int = 2) -> np.ndarray:
    """
    SpecAugment 时间掩码 (Time Masking)
    
    【功能】
    在梅尔频谱图上随机遮挡若干时间段，将选定区域的值置为零。
    用于模拟信号中断或时间片段丢失的水下声学场景，
    提高模型在时域不连续条件下的鲁棒性。
    
    Args:
        mel_spec: 梅尔频谱图 [n_mels, n_frames]
        max_mask_width: 最大掩码宽度 (帧数)
        mask_count: 掩码数量
    
    Returns:
        掩码后的梅尔频谱图
    """
    mel_spec = mel_spec.copy()
    n_frames = mel_spec.shape[1]
    for _ in range(mask_count):
        mask_width = random.randint(1, max_mask_width)
        mask_start = random.randint(0, n_frames - mask_width)
        mel_spec[:, mask_start:mask_start + mask_width] = 0
    return mel_spec


# [4.9] 随机裁剪/填充
def random_crop_pad(waveform: np.ndarray, target_length: int) -> np.ndarray:
    """
    随机裁剪/填充 (Random Crop/Pad)
    
    【功能】
    将音频调整为指定长度：若音频长于目标长度则随机裁剪一段；
    若短于目标长度则在末尾补零。
    用于统一数据集中所有样本的时长，便于模型批处理。
    
    Args:
        waveform: 输入波形
        target_length: 目标长度（采样点数）
    
    Returns:
        调整后的波形
    """
    current_length = len(waveform)
    if current_length > target_length:
        # 随机裁剪
        start = random.randint(0, current_length - target_length)
        return waveform[start:start + target_length]
    elif current_length < target_length:
        # 末尾补零
        padding = np.zeros(target_length - current_length, dtype=waveform.dtype)
        return np.concatenate([waveform, padding])
    else:
        return waveform


# ============================================================
# 音频加载与预处理
# ============================================================

def load_audio(
    file_path: str,
    target_sr: int = 16000,
    duration: Optional[float] = None
) -> Tuple[np.ndarray, int]:
    """
    加载音频文件并重采样到目标采样率
    
    Args:
        file_path: 音频文件路径
        target_sr: 目标采样率
        duration: 目标时长（秒），若指定则统一长度
    
    Returns:
        (waveform, sr) 元组
    """
    logger = logging.getLogger(__name__)
    waveform, sr = librosa.load(file_path, sr=target_sr, mono=True)
    original_len = len(waveform)
    logger.info(f"  [加载] {file_path} → 采样率={sr}Hz, 原始长度={original_len}点({original_len/sr:.2f}s)")
    if duration is not None:
        target_length = int(target_sr * duration)
        if original_len != target_length:
            logger.info(f"  [裁剪/填充] {original_len}点 → {target_length}点 (目标时长={duration}s)")
        waveform = random_crop_pad(waveform, target_length)
    return waveform, sr


# ============================================================
# 增强管线
# ============================================================

def apply_augmentation_pipeline(
    waveform: np.ndarray,
    sr: int,
    config: AugmentationConfig
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """
    随机应用增强管线 (Augmentation Pipeline)
    
    【功能】
    根据 AugmentationConfig 中的配置和概率，随机选择并应用多种增强方法。
    每个增强方法独立以配置的概率决定是否应用，支持任意组合。
    记录所有已应用的增强操作及其参数，用于溯源。
    
    Args:
        waveform: 输入波形
        sr: 采样率
        config: 增强配置
    
    Returns:
        (增强后的波形, 溯源记录列表)
    """
    logger = logging.getLogger(__name__)
    augmented = np.copy(waveform)
    provenance: List[Dict[str, object]] = []
    applied_count = 0

    # --- 1. 高斯噪声 ---
    if config.snr_db is not None and random.random() < config.snr_prob:
        augmented = add_gaussian_noise(augmented, config.snr_db)
        provenance.append({"type": "gaussian_noise", "params": {"snr_db": config.snr_db}})
        logger.debug(f"    [增强] 高斯噪声 SNR={config.snr_db}dB")
        applied_count += 1

    # --- 2. 自定义噪声混合 ---
    if config.noise_file is not None and random.random() < config.noise_prob:
        if os.path.exists(config.noise_file):
            noise_wav, _ = librosa.load(config.noise_file, sr=sr, mono=True)
            augmented = mix_noise_file(augmented, noise_wav, config.noise_snr)
            provenance.append({"type": "custom_noise", "params": {"file": config.noise_file, "snr": config.noise_snr}})
            logger.debug(f"    [增强] 自定义噪声 file={config.noise_file} SNR={config.noise_snr}dB")
            applied_count += 1

    # --- 3. 背景噪声库混合 ---
    if config.noise_dir is not None and random.random() < config.bg_noise_prob:
        noise_files = list(Path(config.noise_dir).glob("*.wav")) + list(Path(config.noise_dir).glob("*.flac"))
        if noise_files:
            chosen_noise = str(random.choice(noise_files))
            noise_wav, _ = librosa.load(chosen_noise, sr=sr, mono=True)
            snr = random.uniform(*config.bg_noise_snr_range)
            augmented = mix_noise_file(augmented, noise_wav, snr)
            provenance.append({"type": "bg_noise", "params": {"file": chosen_noise, "snr": snr}})
            logger.debug(f"    [增强] 背景噪声 file={chosen_noise} SNR={snr:.1f}dB")
            applied_count += 1

    # --- 4. 时间拉伸 ---
    if config.time_stretch_factors and random.random() < config.time_stretch_prob:
        rate = random.choice(config.time_stretch_factors)
        augmented = time_stretch(augmented, rate, sr)
        provenance.append({"type": "time_stretch", "params": {"rate": rate}})
        logger.debug(f"    [增强] 时间拉伸 rate={rate}")
        applied_count += 1

    # --- 5. 音高偏移 ---
    if config.pitch_shift_steps and random.random() < config.pitch_shift_prob:
        n_steps = random.choice(config.pitch_shift_steps)
        augmented = pitch_shift(augmented, sr, n_steps)
        provenance.append({"type": "pitch_shift", "params": {"n_steps": n_steps}})
        logger.debug(f"    [增强] 音高偏移 n_steps={n_steps}")
        applied_count += 1

    # --- 6. 音量扰动 ---
    if random.random() < config.volume_prob:
        factor = random.uniform(*config.volume_range)
        augmented = volume_perturbation(augmented, factor)
        provenance.append({"type": "volume_perturbation", "params": {"factor": factor}})
        logger.debug(f"    [增强] 音量扰动 factor={factor:.3f}")
        applied_count += 1

    # --- 7. 混响 ---
    if random.random() < config.reverb_prob:
        augmented = add_reverberation(augmented, sr, config.reverb_decay, config.reverb_delay)
        provenance.append({"type": "reverberation", "params": {"decay": config.reverb_decay, "delay": config.reverb_delay}})
        logger.debug(f"    [增强] 混响 decay={config.reverb_decay} delay={config.reverb_delay}s")
        applied_count += 1

    # --- 8. SpecAugment 频率/时间掩码 (在 Mel 谱图上操作) ---
    # 注意: freq_mask / time_mask 需要在 Mel 谱图上操作，
    # 这里返回原始波形，掩码在可视化或特征提取阶段进行

    # --- 9. 随机裁剪/填充 ---
    if random.random() < config.random_crop_prob:
        target_length = int(sr * config.duration)
        augmented = random_crop_pad(augmented, target_length)
        provenance.append({"type": "random_crop", "params": {"target_length": target_length}})
        logger.debug(f"    [增强] 随机裁剪/填充 target_length={target_length}")
        applied_count += 1

    logger.info(f"  [管线] 应用了 {applied_count} 种增强方法")
    return augmented, provenance


# ============================================================
# 主函数
# ============================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="水声数据增强工具 - 支持9种增强方法与可视化输出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 单文件增强
  python augment.py --audio_path /path/to/audio.wav --snr 15 --output_dir ./output
  
  # 批量增强目录下所有文件
  python augment.py --input_dir /path/to/dataset --output_dir ./augmented --num_augmented 5
  
  # 使用背景噪声库
  python augment.py --input_dir ./data --noise_dir ./noise_lib --bg_noise_prob 0.4 --output_dir ./augmented
        """
    )
    
    # --- 输入/输出 ---
    io_group = parser.add_argument_group("输入/输出")
    io_group.add_argument("--audio_path", type=str, default=None,
                          help="单音频文件路径（与 --input_dir 二选一）")
    io_group.add_argument("--input_dir", type=str, default=None,
                          help="输入音频目录（与 --audio_path 二选一）")
    io_group.add_argument("--output_dir", type=str, default="./augmented_output",
                          help="增强音频输出目录 (默认: ./augmented_output)")
    io_group.add_argument("--num_augmented", type=int, default=3,
                          help="每个原始文件生成的增强样本数 (默认: 3)")
    
    # --- 基础参数 ---
    base_group = parser.add_argument_group("基础参数")
    base_group.add_argument("--target_sr", type=int, default=16000,
                            help="目标采样率 (Hz, 默认: 16000)")
    base_group.add_argument("--duration", type=float, default=3.0,
                            help="音频标准长度 (秒, 默认: 3.0)")
    
    # --- 噪声增强 ---
    noise_group = parser.add_argument_group("噪声增强")
    noise_group.add_argument("--snr", type=float, default=None,
                             help="高斯噪声信噪比 dB (默认: 不启用)")
    noise_group.add_argument("--snr_prob", type=float, default=0.5,
                             help="高斯噪声应用概率 (默认: 0.5)")
    noise_group.add_argument("--noise_file", type=str, default=None,
                             help="自定义噪声文件路径 (默认: 不启用)")
    noise_group.add_argument("--noise_snr", type=float, default=15.0,
                             help="自定义噪声混合信噪比 dB (默认: 15.0)")
    noise_group.add_argument("--noise_prob", type=float, default=0.3,
                             help="自定义噪声应用概率 (默认: 0.3)")
    noise_group.add_argument("--noise_dir", type=str, default=None,
                             help="背景噪声库目录 (默认: 不启用)")
    noise_group.add_argument("--bg_noise_snr_min", type=float, default=5.0,
                             help="背景噪声信噪比下限 (默认: 5.0)")
    noise_group.add_argument("--bg_noise_snr_max", type=float, default=20.0,
                             help="背景噪声信噪比上限 (默认: 20.0)")
    noise_group.add_argument("--bg_noise_prob", type=float, default=0.3,
                             help="背景噪声应用概率 (默认: 0.3)")
    
    # --- 时间/频率增强 ---
    tf_group = parser.add_argument_group("时间/频率增强")
    tf_group.add_argument("--time_stretch_factors", type=float, nargs="+",
                          default=[0.85, 0.9, 1.1, 1.15],
                          help="时间拉伸系数列表 (默认: 0.85 0.9 1.1 1.15)")
    tf_group.add_argument("--time_stretch_prob", type=float, default=0.3,
                          help="时间拉伸应用概率 (默认: 0.3)")
    tf_group.add_argument("--pitch_shift_steps", type=int, nargs="+",
                          default=[-3, -2, -1, 1, 2, 3],
                          help="音高偏移半音数列表 (默认: -3 -2 -1 1 2 3)")
    tf_group.add_argument("--pitch_shift_prob", type=float, default=0.3,
                          help="音高偏移应用概率 (默认: 0.3)")
    
    # --- 音量/混响 ---
    vol_group = parser.add_argument_group("音量/混响增强")
    vol_group.add_argument("--volume_min", type=float, default=0.5,
                           help="音量缩放最小值 (默认: 0.5)")
    vol_group.add_argument("--volume_max", type=float, default=1.5,
                           help="音量缩放最大值 (默认: 1.5)")
    vol_group.add_argument("--volume_prob", type=float, default=0.5,
                           help="音量扰动应用概率 (默认: 0.5)")
    vol_group.add_argument("--reverb_decay", type=float, default=0.3,
                           help="混响衰减系数 (默认: 0.3)")
    vol_group.add_argument("--reverb_delay", type=float, default=0.05,
                           help="混响延迟时间 秒 (默认: 0.05)")
    vol_group.add_argument("--reverb_prob", type=float, default=0.2,
                           help="混响应用概率 (默认: 0.2)")
    
    # --- SpecAugment ---
    spec_group = parser.add_argument_group("SpecAugment")
    spec_group.add_argument("--freq_mask_max", type=int, default=20,
                            help="最大频率掩码宽度 (默认: 20)")
    spec_group.add_argument("--freq_mask_count", type=int, default=2,
                            help="频率掩码数量 (默认: 2)")
    spec_group.add_argument("--freq_mask_prob", type=float, default=0.3,
                            help="频率掩码应用概率 (默认: 0.3)")
    spec_group.add_argument("--time_mask_max", type=int, default=50,
                            help="最大时间掩码宽度 (默认: 50)")
    spec_group.add_argument("--time_mask_count", type=int, default=2,
                            help="时间掩码数量 (默认: 2)")
    spec_group.add_argument("--time_mask_prob", type=float, default=0.3,
                            help="时间掩码应用概率 (默认: 0.3)")
    
    # --- 裁剪/填充 ---
    crop_group = parser.add_argument_group("裁剪/填充")
    crop_group.add_argument("--random_crop_prob", type=float, default=0.0,
                            help="随机裁剪应用概率 0=不启用 (默认: 0.0)")
    
    # --- 日志 ---
    log_group = parser.add_argument_group("日志")
    log_group.add_argument("--log_file", type=str, default=None,
                           help="日志文件保存路径 (默认: 不保存，仅输出到控制台)")
    
    # --- 可视化 ---
    vis_group = parser.add_argument_group("可视化")
    vis_group.add_argument("--disable_visualization", action="store_true",
                           help="禁用可视化输出")
    vis_group.add_argument("--vis_dir", type=str, default="./visualizations",
                           help="可视化输出目录 (默认: ./visualizations)")
    vis_group.add_argument("--vis_dpi", type=int, default=150,
                           help="可视化图像 DPI (默认: 150)")
    vis_group.add_argument("--vis_format", type=str, default="png",
                           choices=["png", "jpg", "pdf", "svg"],
                           help="可视化图像格式 (默认: png)")
    
    return parser.parse_args()


def args_to_config(args: argparse.Namespace) -> AugmentationConfig:
    """将命令行参数转换为 AugmentationConfig"""
    return AugmentationConfig(
        snr_db=args.snr,
        snr_prob=args.snr_prob,
        noise_file=args.noise_file,
        noise_snr=args.noise_snr,
        noise_prob=args.noise_prob,
        noise_dir=args.noise_dir,
        bg_noise_snr_range=(args.bg_noise_snr_min, args.bg_noise_snr_max),
        bg_noise_prob=args.bg_noise_prob,
        time_stretch_factors=args.time_stretch_factors,
        time_stretch_prob=args.time_stretch_prob,
        pitch_shift_steps=args.pitch_shift_steps,
        pitch_shift_prob=args.pitch_shift_prob,
        volume_range=(args.volume_min, args.volume_max),
        volume_prob=args.volume_prob,
        reverb_decay=args.reverb_decay,
        reverb_delay=args.reverb_delay,
        reverb_prob=args.reverb_prob,
        freq_mask_max=args.freq_mask_max,
        freq_mask_count=args.freq_mask_count,
        freq_mask_prob=args.freq_mask_prob,
        time_mask_max=args.time_mask_max,
        time_mask_count=args.time_mask_count,
        time_mask_prob=args.time_mask_prob,
        random_crop_prob=args.random_crop_prob,
        target_sr=args.target_sr,
        duration=args.duration,
        output_dir=args.output_dir,
        num_augmented_per_file=args.num_augmented,
        enable_visualization=not args.disable_visualization,
        vis_dir=args.vis_dir,
        vis_dpi=args.vis_dpi,
        vis_format=args.vis_format,
    )


def process_single_file(
    audio_path: str,
    config: AugmentationConfig,
    provenance_records: List[AugmentationRecord]
) -> None:
    """
    处理单个音频文件，生成多个增强版本
    
    Args:
        audio_path: 音频文件路径
        config: 增强配置
        provenance_records: 溯源记录列表（结果追加到此列表）
    """
    base_name = Path(audio_path).stem
    logger = logging.getLogger(__name__)
    
    # 加载原始音频
    original_waveform, sr = load_audio(audio_path, config.target_sr, config.duration)
    logger.info(f"已加载: {audio_path} (时长: {len(original_waveform)/sr:.2f}s)")
    
    # 保存原始音频的可视化
    if config.enable_visualization:
        save_individual_visualizations(
            original_waveform, sr,
            config.vis_dir, base_name,
            prefix="original",
            dpi=config.vis_dpi, fmt=config.vis_format
        )
    
    # 生成多个增强样本
    for aug_idx in range(config.num_augmented_per_file):
        logger.info(f"  --- 生成增强样本 #{aug_idx + 1} ---")
        
        # 应用增强管线
        augmented_waveform, aug_provenance = apply_augmentation_pipeline(
            original_waveform, sr, config
        )
        
        # 打印已应用的增强方法摘要
        aug_types = [a["type"] for a in aug_provenance]
        logger.info(f"  [增强摘要] 样本 #{aug_idx + 1}: {', '.join(aug_types) if aug_types else '无增强（原始）'}")
        
        # 生成输出文件名
        aug_filename = f"{base_name}_aug_{aug_idx + 1}.wav"
        aug_path = os.path.join(config.output_dir, aug_filename)
        
        # 保存增强音频
        os.makedirs(config.output_dir, exist_ok=True)
        sf.write(aug_path, augmented_waveform, sr)
        logger.info(f"  [保存音频] → {aug_path} ({len(augmented_waveform)/sr:.2f}s)")
        
        # 保存增强音频的可视化
        if config.enable_visualization:
            # 独立可视化
            save_individual_visualizations(
                augmented_waveform, sr,
                config.vis_dir, base_name,
                prefix=f"aug_{aug_idx + 1}",
                dpi=config.vis_dpi, fmt=config.vis_format
            )
            
            # 对比网格图
            comp_path = os.path.join(
                config.vis_dir,
                f"{base_name}_comparison_aug_{aug_idx + 1}.{config.vis_format}"
            )
            plot_comparison_grid(
                original_waveform, augmented_waveform, sr,
                save_path=comp_path,
                dpi=config.vis_dpi
            )
        
        # 记录溯源
        record = AugmentationRecord(
            original_file=audio_path,
            augmented_file=aug_path,
            augmentations=aug_provenance
        )
        provenance_records.append(record)


def main() -> None:
    """主入口函数"""
    # 解析参数
    args = parse_args()
    config = args_to_config(args)
    
    # 配置日志: 控制台输出 + 可选文件输出
    log_handlers: List[logging.Handler] = []
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    log_handlers.append(console_handler)

    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) if os.path.dirname(args.log_file) else ".", exist_ok=True)
        file_handler = logging.FileHandler(args.log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        ))
        log_handlers.append(file_handler)

    logging.basicConfig(
        level=logging.INFO,
        handlers=log_handlers,
        force=True
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("水声数据增强工具")
    logger.info("=" * 60)
    logger.info(f"配置: output_dir={config.output_dir}, target_sr={config.target_sr}Hz, "
                f"duration={config.duration}s, num_augmented={config.num_augmented_per_file}")
    logger.info(f"配置: snr_db={config.snr_db}, noise_dir={config.noise_dir}, "
                f"可视化={'启用' if config.enable_visualization else '禁用'}")
    if args.log_file:
        logger.info(f"配置: 日志文件 → {args.log_file}")
    
    # 校验输入
    if args.audio_path is None and args.input_dir is None:
        logger.error("请指定 --audio_path (单文件) 或 --input_dir (批量模式)")
        sys.exit(1)
    
    # 收集待处理文件
    audio_files: List[str] = []
    if args.audio_path:
        if os.path.isfile(args.audio_path):
            audio_files.append(args.audio_path)
        else:
            logger.error(f"文件不存在: {args.audio_path}")
            sys.exit(1)
    elif args.input_dir:
        input_path = Path(args.input_dir)
        if not input_path.is_dir():
            logger.error(f"目录不存在: {args.input_dir}")
            sys.exit(1)
        audio_files = sorted(
            [str(p) for p in input_path.glob("*") if p.suffix.lower() in (".wav", ".flac", ".mp3", ".m4a")]
        )
        if not audio_files:
            logger.error(f"目录中未找到音频文件: {args.input_dir}")
            sys.exit(1)
        logger.info(f"发现 {len(audio_files)} 个音频文件")
    
    # 处理每个文件
    provenance_records: List[AugmentationRecord] = []
    for file_path in audio_files:
        try:
            process_single_file(file_path, config, provenance_records)
        except Exception as e:
            logger.error(f"处理文件失败 {file_path}: {e}", exc_info=True)
    
    # 保存溯源记录
    if provenance_records:
        provenance_path = os.path.join(config.output_dir, config.provenance_file)
        records_dict = [r.to_dict() for r in provenance_records]
        with open(provenance_path, "w", encoding="utf-8") as f:
            json.dump(records_dict, f, ensure_ascii=False, indent=2)
        logger.info(f"溯源记录已保存: {provenance_path} ({len(provenance_records)} 条)")
    
    logger.info("=" * 60)
    total_augmented = len(provenance_records)
    logger.info(f"处理完成! 共处理 {len(audio_files)} 个原始文件，生成 {total_augmented} 个增强样本")
    logger.info(f"输出目录: {os.path.abspath(config.output_dir)}")
    if config.enable_visualization:
        logger.info(f"可视化目录: {os.path.abspath(config.vis_dir)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()