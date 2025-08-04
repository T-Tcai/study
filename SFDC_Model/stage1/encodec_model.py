import numpy as np
import torch
from torch import nn
import os
import sys
# 获取当前文件的绝对路径（例如：/home/xinyue/code/ControlTCEmodel/stage2_decouping/train_decouping.py）
current_file_path = os.path.abspath(__file__)
# 计算项目根目录（/home/xinyue/code/ControlTCEmodel）
project_root = os.path.dirname(os.path.dirname(current_file_path))
# 将根目录添加到Python路径的最前面（确保优先搜索）
sys.path.insert(0, project_root)
import stage1.modules as m
import math
import torch.nn.functional as F

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
     
class Encodec_model(nn.Module):
    def __init__(self,
                 target_bandwidths: int = 24,
                 sample_rate: int = 16_000,
                 channels: int = 1,
                 causal: bool = True,
                 model_norm: str = 'weight_norm',
                 dimension: int= 256,
                 ratios=[8, 5, 4, 2],
                 ):
        
        super().__init__()
        self.target_bandwidths = target_bandwidths
        self.ratios = ratios
        self.dimension = dimension
        self.encoder = m.SEANetEncoder(channels=channels,norm=model_norm,causal=causal, ratios=ratios)
        self.frame_rate = math.ceil(sample_rate / np.prod(self.ratios))
        self.sample_rate = sample_rate

        # 解码器主要是为了解耦
        self.decoder = m.SEANetDecoder(channels=channels, norm=model_norm, causal=causal,ratios=ratios)        
   

    def forward(self, audio_wave1):
        # 音频输入形状：[B, 1, T]
        # 音色特征提取
        wav_feat = self.encoder(audio_wave1)
        # 重建音频
        waveform = self.decoder(wav_feat)
        return waveform