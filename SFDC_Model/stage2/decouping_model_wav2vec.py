import numpy as np
import torch
from torch import nn
from transformers import WavLMModel
import os
import sys
# 获取当前文件的绝对路径（例如：/home/xinyue/code/ControlTCEmodel/stage2_decouping/train_decouping.py）
current_file_path = os.path.abspath(__file__)
# 计算项目根目录（/home/xinyue/code/ControlTCEmodel）
project_root = os.path.dirname(os.path.dirname(current_file_path))
# 将根目录添加到Python路径的最前面（确保优先搜索）
sys.path.insert(0, project_root)

import math
import torch.nn.functional as F

import torchaudio.functional as AF  # 用于音频处理
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

import os
import sys
# 获取当前文件的绝对路径（例如：/home/xinyue/code/ControlTCEmodel/stage2_decouping/train_decouping.py）
current_file_path = os.path.abspath(__file__)
# 计算项目根目录（/home/xinyue/code/ControlTCEmodel）
project_root = os.path.dirname(os.path.dirname(current_file_path))
# 将根目录添加到Python路径的最前面（确保优先搜索）
sys.path.insert(0, project_root)
# 使用绝对导入（基于包名 ControlTCEmodel）
from stage1.encodec_model import Encodec_model
import stage2.quantization as qt
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model
# from vector_quantize_pytorch import VectorQuantize
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
import pyworld as pw
import librosa
from scipy.interpolate import interp1d

class AcousticPreprocessor:
    def __init__(self, sr=16000, hop_size=320):
        self.sr = sr
        self.hop_size = hop_size

    def extract_features(self, wav):
        if isinstance(wav, torch.Tensor):
            wav = wav.squeeze().cpu().numpy()

        features = []
        if wav.ndim == 1:
            wav = np.expand_dims(wav, axis=0)

        for audio in wav:
            audio = audio.astype(np.float64)
            if np.max(np.abs(audio)) < 1e-5:
                # 纯静音处理
                length = max(1, int(np.ceil(len(audio) / self.hop_size)))
                features.append({
                    'f0': torch.zeros(length),
                    'energy': torch.zeros(length),
                })
                continue

            _f0, t = pw.harvest(audio, fs=self.sr, frame_period=1000 * self.hop_size / self.sr)
            _f0 = np.nan_to_num(_f0)
            f0 = pw.stonemask(audio, _f0, t, self.sr)
            f0 = np.nan_to_num(f0)

            # Clip f0到正常人类说话范围，防止极端值
            f0 = np.clip(f0, 50, 1100)

            if np.all(f0 == 0):
                length = max(1, len(f0))
                features.append({
                    'f0': torch.zeros(length),
                    'energy': torch.zeros(length),
                })
                continue

            sp = pw.cheaptrick(audio, f0, t, self.sr)
            sp = np.nan_to_num(sp)

            energy = np.sqrt((sp ** 2).sum(axis=1))
            energy = np.nan_to_num(energy)
            # Clip能量，防止过小
            energy = np.clip(energy, 1e-5, None)


            features.append({
                'f0': torch.tensor(f0, dtype=torch.float32),
                'energy': torch.tensor(energy, dtype=torch.float32),
            })
        return features

def sample_normalization(x, eps=1e-5):
    """
    Args:
        x: (B, T) tensor
        eps: Small value to prevent division by zero
    Returns:
        Normalized tensor: (B, T)
    """
    mean = x.mean(dim=1, keepdim=True)  # 按每个样本的时间维度做归一化
    std = x.std(dim=1, keepdim=True)
    std = torch.clamp(std, min=eps)

    x_norm = (x - mean) / std
    x_norm = torch.nan_to_num(x_norm)

    return x_norm


class ProsodyLoss(nn.Module):
    def __init__(self, gamma=0.5, contrast_temp=0.1):
        super().__init__()
        self.gamma = gamma
        self.contrast_temp = contrast_temp
        self.preprocessor = AcousticPreprocessor()
    def forward(self, wav, res):
        f0_pred = res['f0'].squeeze(1) # [bs, 1, T]
        energy_pred = res['energy'].squeeze(1) # [bs, 1, T]
    
        with torch.no_grad():
            features = self.preprocessor.extract_features(wav)  # 返回字典列表
            f0_list = [feat['f0'] for feat in features]  # 从字典中提取 f0 特征
            energy_list = [feat['energy'] for feat in features]  # 提取 energy 特征

            # 使用 torch.stack 将它们堆叠成 [bs, T] 的格式
            f0_true = torch.stack(f0_list, dim=0).to(f0_pred.device)  # 形状为 [bs, T]
            energy_true = torch.stack(energy_list, dim=0).to(energy_pred.device)  # 形状为 [bs, T]

        # 批量归一化
        f0_true_norm = sample_normalization(f0_true)
        energy_true_norm = sample_normalization(energy_true)

        f0_pred_norm = sample_normalization(f0_pred)
        energy_pred_norm = sample_normalization(energy_pred)
        
        if f0_pred_norm.shape[-1] != f0_true_norm.shape[-1]:
            f0_pred_norm = F.interpolate(f0_pred_norm.unsqueeze(1), size=f0_true_norm.shape[-1], mode='linear').squeeze(1)
        if energy_pred_norm.shape[-1] != energy_true_norm.shape[-1]:
            energy_pred_norm = F.interpolate(energy_pred_norm.unsqueeze(1), size=energy_true_norm.shape[-1], mode='linear').squeeze(1)
        
        f0_loss = F.mse_loss(f0_pred_norm, f0_true_norm, reduction='none')  # [bs, num_frames] 逐时间步计算
        energy_loss = F.mse_loss(energy_pred_norm, energy_true_norm, reduction='none')  # [bs, num_frames] 

        # 汇总损失
        total_f0_loss = f0_loss.mean(dim=1).mean()  # 对 batch 和帧求平均
        total_energy_loss = energy_loss.mean(dim=1).mean()  # 对 batch 和帧求平均


        # 数值稳定性处理
        epsilon = 1e-8
        total_base_loss = total_f0_loss + total_energy_loss + epsilon
        gamma = self.gamma * (total_f0_loss / total_base_loss)

        # 总损失组合
        total_loss = (
            gamma * total_f0_loss +
            (1 - gamma) * total_energy_loss
        )

        print(f"gamma: {gamma}, f0_loss: {total_f0_loss}, energy_loss: {total_energy_loss}, total_loss:{total_loss}")
        return total_loss


    
class TimbreEncoder(nn.Module):
    def __init__(self, input_dim=512, output_dim=256):
        super().__init__()
        # 通道调整：将 [bs, 512, T] -> [bs, T, 512]
        self.permute = lambda x: x.permute(0, 2, 1)
        
        # 注意力池化（聚焦关键音色片段）
        self.attention = nn.MultiheadAttention(
            embed_dim=512, num_heads=8, batch_first=True
        )
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # 投影到音色嵌入
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        # 输入形状: [bs, 512, T]
        x = self.permute(x)  # [bs, T, 512]
        # 注意力池化（Query用全局平均初始化）
        query = self.global_pool(x.permute(0, 2, 1)).permute(0, 2, 1)  # [bs, 1, 512]
        x, _ = self.attention(query, x, x)  # [bs, 1, 512]
        # 压缩到音色嵌入
        x = x.squeeze(1)  # [bs, 512]
        x = self.fc(x)    # [bs, output_dim]
        return x

class ResBlock(nn.Module):
    def __init__(self, channels, dilation=1, use_attention=False):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm1d(channels, affine=True)
        self.norm2 = nn.InstanceNorm1d(channels, affine=True)
        self.activation = nn.GELU()
        self.use_attention = use_attention
        if use_attention:
            self.attn = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.use_attention:
            x_attn, _ = self.attn(x.permute(0, 2, 1), x.permute(0, 2, 1), x.permute(0, 2, 1))
            x = x_attn.permute(0, 2, 1)
        return x + residual

# class AttentionPooling(nn.Module):
#     def __init__(self, input_dim, num_heads=4):
#         super().__init__()
#         self.num_heads = num_heads
#         self.attention = nn.MultiheadAttention(
#             embed_dim=input_dim,
#             num_heads=num_heads,
#             batch_first=True
#         )
#         # 可学习的全局查询向量
#         self.query = nn.Parameter(torch.randn(1, 1, input_dim))  # (1, 1, C)
        
#     def forward(self, x):
#         """
#         x: [B, C, T] → 池化为 [B, C]
#         """
#         # 转置为 [B, T, C]
#         x = x.permute(0, 2, 1)  # (B, T, C)
        
#         # 扩展查询向量到 batch 维度
#         query = self.query.expand(x.size(0), -1, -1)  # (B, 1, C)
        
#         # 计算注意力 (query 是全局查询，key/value 是输入特征)
#         attn_output, _ = self.attention(
#             query=query,  # (B, 1, C)
#             key=x,        # (B, T, C)
#             value=x       # (B, T, C)
#         )
        
#         # 输出形状: (B, 1, C) → 压缩为 (B, C)
#         return attn_output.squeeze(1)

class EmotionEncoder(nn.Module):
    def __init__(self, input_dim=512, base_channels=128, output_dim=256):
        super().__init__()
        
        # ==== Stage 1: 韵律特征预测 ====
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm1d(base_channels),
            nn.GELU(),
        )

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, kernel_size=5, padding=2),
            ResBlock(base_channels, dilation=1, use_attention=True),
            ResBlock(base_channels, dilation=2),
            nn.Conv1d(base_channels, base_channels*2, kernel_size=3, padding=1),
            nn.InstanceNorm1d(base_channels*2)  # ← 替换
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=base_channels*2,
            nhead=8,
            dim_feedforward=512,
            batch_first=True,
            norm_first=True,
            activation=nn.GELU()
        )
        self.temporal_model = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # f0 回归头
        self.f0_proj = nn.Sequential(
            nn.Conv1d(base_channels*2, base_channels//2, 3, padding=1),
            nn.InstanceNorm1d(base_channels//2),
            nn.Conv1d(base_channels//2, 1, 3, padding=1)
        )

        # energy 回归头
        self.energy_proj = nn.Sequential(
            nn.Conv1d(base_channels*2, base_channels//2, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_channels//2, 1, 3, padding=1),
            nn.Softplus()
        )


        # ==== Stage 2: 情感特征融合模块 ====
        self.fusion_net = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, output_dim, kernel_size=3, padding=1),
            nn.InstanceNorm1d(output_dim)
        )
        # self.attention_pooling = AttentionPooling(output_dim)  # 注意这里使用output_dim
        
    
    def forward(self, x):
        # x: [B, 512, T]
        x = self.input_proj(x)                     # [B, 128, T]
        x = self.feature_extractor(x)              # [B, 256, T]
        x = x.permute(0, 2, 1)                     # [B, T, 256]
        x = self.temporal_model(x)                 # [B, T, 256]
        x = x.permute(0, 2, 1)                     # [B, 256, T]

        f0 = self.f0_proj(x)                       # [B, 1, T]
        energy = self.energy_proj(x)               # [B, 1, T]

        # 融合韵律特征构建情感特征
        rhythm_feat = torch.cat([f0, energy], dim=1)  # [B, 2, T]
        emo_feat = self.fusion_net(rhythm_feat)                # [B, 256, T]
        
        # 使用注意力池化去除时间维度
        # pooled_emo_feat = self.attention_pooling(emo_feat)     # [B, 256]
        
        return {
            'f0': f0,
            'energy': energy,
            'feat': emo_feat,          # [B, 256, T]
            # 'pooled_feat': pooled_emo_feat  # [B, 256]
        }

        
class SemanticEncoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=768, output_dim=512):
        super().__init__()
        # model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn" #1024
        model_name = "facebook/wav2vec2-base-960h"
        # print("========================================")
        # model_name = "facebook/wav2vec2-large-xlsr-53"
        self.wav2vec = Wav2Vec2Model.from_pretrained(model_name).to(device)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        
        # 特征映射 + 激活增强
        self.transform = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        # 可学习的放缩系数，初始放大10倍
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, x):

        with torch.no_grad():
            inputs = self.feature_extractor(
                x,
                sampling_rate=16000,
                return_tensors="pt",
                padding="longest",
                do_normalize=True
            )
            outputs = self.wav2vec(
                input_values=inputs.input_values.to(device),
                output_hidden_states=True,
                # attention_mask=inputs.attention_mask.to(device),
                return_dict=True
            )
            x = outputs.hidden_states[-1]  # [bs, T, 1024]

        x = x.permute(0, 2, 1)              # [bs, 1024, T]
        x = self.transform(x)              # [bs, 512, T]
        x = x * self.scale                 # 放大幅度
        return x                   # [bs, 512, T], [bs, 512]
    
class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.1):
        super().__init__()
        self.temp = torch.tensor(temp)
        self.eps = 1e-8

    def forward(self, anchor, pos, neg):
        # ===== 1. 输入验证 =====
        assert not torch.isnan(anchor).any(), "Input contains NaN!"
        assert not torch.isinf(anchor).any(), "Input contains Inf!"
        
        # ===== 2. 安全归一化 =====
        anchor = nn.functional.normalize(anchor, p=2, dim=-1)
        pos = nn.functional.normalize(pos, p=2, dim=-1)
        neg = nn.functional.normalize(neg, p=2, dim=-1)
        
        # ===== 3. 调试打印 =====
        if not torch.allclose(anchor.norm(dim=-1), torch.ones_like(anchor.norm(dim=-1)), atol=1e-3):
            print("WARNING: Anchor norm deviation:", anchor.norm(dim=-1))

        # ===== 4. 相似度计算 =====
        temp = torch.clamp(self.temp, min=0.01)  # 温度限制
        pos_sim = torch.einsum('bd,bpd->bp', anchor, pos) / temp
        neg_sim = torch.einsum('bd,bnd->bn', anchor, neg) / temp

        # ===== 5. 数值稳定化 =====
        max_val = torch.maximum(
            pos_sim.max(dim=1, keepdim=True)[0], 
            neg_sim.max(dim=1, keepdim=True)[0]
        ).detach()  # 阻止梯度回传

        # ===== 6. 损失计算 =====
        pos_exp = torch.exp(pos_sim - max_val).sum(dim=1)
        neg_exp = torch.exp(neg_sim - max_val).sum(dim=1)
        denominator = pos_exp + neg_exp + self.eps
        losses = -torch.log(pos_exp / denominator)
        
        return losses.mean()
    
    


class Decouping_model(nn.Module):
    def __init__(self):     
        super().__init__()

        seanet = Encodec_model().to(device) 
        # 加载 SEANet 预训练权重
        model_checkpoint = torch.load("/home/xinyue/code/TCEmodel/stage1/checkpoints_512/bs24_epoch100_lr3e-4.pt", map_location=device)
        seanet.load_state_dict(model_checkpoint['model_state_dict'])      

        # 加载 SEANet 编码器
        self.encoder = seanet.encoder
        # 冻结 SEANet 参数
        for p in self.encoder.parameters():
            p.requires_grad = False      
        
        # 核心模块
        self.speaker_enc = TimbreEncoder()
        self.emotion_enc = EmotionEncoder()
        self.content_enc = SemanticEncoder()
        self.speaker_loss = ContrastiveLoss(temp=0.06)
        self.emotion_loss = ContrastiveLoss(temp=0.12)
        self.content_loss = ContrastiveLoss(temp=0.10)
        self.emotion_f0_loss = ProsodyLoss(gamma=0.5)

        # 处理音频

        # 提取特征
       
        self.Timquantizer = qt.ResidualVectorQuantizer(
            dimension = 256,
            n_q = 4,
            bins = 512,
        )
        self.Emoquantizer = qt.ResidualVectorQuantizer(
            dimension = 256,
            n_q = 4,
            bins = 512,
        )
        self.Conquantizer = qt.ResidualVectorQuantizer(
            dimension = 512,
            n_q = 4,
            bins = 1024,

        )
       
        #   # 7️⃣ 量化情感特征
        # self.fusion = FeatureFusion()
        
    def forward(self, wav1, wav2, wav3,wav4, wav5,wav6,wav7,list):
        """
        wav1: 主要输入语音
        wav2: 说话人匹配音频
        wav3: 情感匹配音频
        """
        # 编码阶段
        B_2,N_2,C_2,T_2 = wav2.shape
        B_3,N_3,C_3,T_3 = wav3.shape
        B_4,N_4,C_4,T_4 = wav4.shape
        B_5,N_5,C_5,T_5 = wav5.shape
        wav2 = wav2.reshape(-1, C_2, T_2)
        wav3 = wav3.reshape(-1, C_3, T_3)
        wav4 = wav4.reshape(-1, C_4, T_4)
        wav5 = wav5.reshape(-1, C_5, T_5)
        with torch.no_grad():
            wav_feat1 = self.encoder(wav1)
            wav_feat2 = self.encoder(wav2)
            wav_feat3 = self.encoder(wav3)
            wav_feat4 = self.encoder(wav4)
            wav_feat5 = self.encoder(wav5)
        
        speaker_feat1 = self.speaker_enc(wav_feat1)
        speaker_feat2 = self.speaker_enc(wav_feat2).reshape(B_2, N_2, 256)
        speaker_feat3 = self.speaker_enc(wav_feat3).reshape(B_3, N_3, 256)
        loss_speaker = self.speaker_loss(speaker_feat1, speaker_feat2, speaker_feat3)
        
        
        emotion_feat1 = self.emotion_enc(wav_feat1)
        emotion_feat4 = self.emotion_enc(wav_feat4)['feat'].reshape(B_4, N_4, 256,-1)
        emotion_feat5 = self.emotion_enc(wav_feat5)['feat'].reshape(B_5, N_5, 256,-1)
        loss_emotion = self.emotion_loss(self.global_pooling(emotion_feat1['feat']),self.global_pooling(emotion_feat4),self.global_pooling(emotion_feat5))
        
        content_feat1 = self.content_enc(list)
        content_pos_list = []
        for pos in wav6:
            content_feat6 = self.content_enc(pos)
            content_pos_list.append(content_feat6.mean(-1))
        content6 = torch.stack(content_pos_list)
        content_neg_list = []
        for neg in wav7:
            content_feat7 = self.content_enc(neg)
            content_neg_list.append(content_feat7.mean(-1))
        content7 = torch.stack(content_neg_list)
        loss_content = self.content_loss(content_feat1.mean(-1), content6, content7)
        
        loss_f0_emotion = self.emotion_f0_loss(wav1,emotion_feat1)
        
        speaker_feat_quantized, _, loss_ws = self.Timquantizer(speaker_feat1.unsqueeze(-1))  # [B, 256]
        emotion_feat_quantized, _, loss_we = self.Emoquantizer(emotion_feat1['feat'])  # [B, 256]
        content_feat_quantized, _, loss_wc = self.Conquantizer(content_feat1)  # [B, 256]

    
        return {
            "speaker_feat": speaker_feat_quantized.squeeze(-1),
            "emotion_feat": emotion_feat_quantized,
            "content_feat": content_feat_quantized,
            "loss_ws": loss_ws,
            "loss_we": loss_we,
            "loss_wc": loss_wc,
            "loss_speaker": loss_speaker.mean(),
            "loss_emotion": loss_emotion.mean(),
            "loss_emotion_f0": loss_f0_emotion,
            "loss_content": loss_content.mean(),
        }
    
    def extract_feature(self, wav,wav_numpy):
        """
        wav1: 主要输入语音
        wav2: 说话人匹配音频
        wav3: 情感匹配音频
        """
        # 编码阶段
        with torch.no_grad():
            wav_feat = self.encoder(wav)
        
        speaker_feat = self.speaker_enc(wav_feat)
        emotion_feat = self.emotion_enc(wav_feat)
        
        content_feat = self.content_enc(wav_numpy)
        
        speaker_feat_quantized, _, _ = self.Timquantizer(speaker_feat.unsqueeze(-1))  # [B, 256]
        emotion_feat_quantized, _, _ = self.Emoquantizer(emotion_feat['feat'])  # [B, 256]
        content_feat_quantized, _, _ = self.Conquantizer(content_feat)  # [B, 256]
    
        return {
            "speaker_feat": speaker_feat_quantized.squeeze(-1),
            "emotion_feat": emotion_feat_quantized,
            "content_feat": content_feat_quantized,
        }
    
    def extract_feature_gen(self, wav1,wav2,wav3):
        """
        wav1: 主要输入语音
        wav2: 说话人匹配音频
        wav3: 情感匹配音频
        """
        # 编码阶段
        with torch.no_grad():
            tim_feat = self.encoder(wav1)
            emo_feat = self.encoder(wav2)
        
        speaker_feat = self.speaker_enc(tim_feat)
        emotion_feat = self.emotion_enc(emo_feat)
        
        content_feat = self.content_enc(wav3)
        
        speaker_feat_quantized, _, _ = self.Timquantizer(speaker_feat.unsqueeze(-1))  # [B, 256]
        emotion_feat_quantized, _, _ = self.Emoquantizer(emotion_feat['feat'])  # [B, 256]
        content_feat_quantized, _, _ = self.Conquantizer(content_feat)  # [B, 256]
    
        return {
            "speaker_feat": speaker_feat_quantized.squeeze(-1),
            "emotion_feat": emotion_feat_quantized,
            "content_feat": content_feat_quantized,
        }
    def extract_feature_gen_p(self, wav1,wav2,wav3,wav4):
        """
        wav1: 主要输入语音
        wav2: 说话人匹配音频
        wav3: 情感匹配音频
        """
        # 编码阶段
        with torch.no_grad():
            tim_feat = self.encoder(wav1)
            emo_feat1 = self.encoder(wav2)
            emo_feat2 = self.encoder(wav3)
        
        speaker_feat = self.speaker_enc(tim_feat)
        emotion_feat1 = self.emotion_enc(emo_feat1)
        emotion_feat2 = self.emotion_enc(emo_feat2)
        
        content_feat = self.content_enc(wav4)
        
        speaker_feat_quantized, _, _ = self.Timquantizer(speaker_feat.unsqueeze(-1))  # [B, 256]
        emotion_feat_quantized1, _, _ = self.Emoquantizer(emotion_feat1['feat'])  # [B, 256]
        emotion_feat_quantized2, _, _ = self.Emoquantizer(emotion_feat2['feat'])  # [B, 256]
        content_feat_quantized, _, _ = self.Conquantizer(content_feat)  # [B, 256]
    
        return {
            "speaker_feat": speaker_feat_quantized.squeeze(-1),
            "emotion_feat1": emotion_feat_quantized1,
            "emotion_feat2": emotion_feat_quantized2,
            "content_feat": content_feat_quantized,
        }
    def global_pooling(self,x):
        # x: [B, C, T] 或 [B, C, L]
        mean_pool = x.mean(dim=-1)
        max_pool = x.max(dim=-1).values  # 注意 .values
        return torch.cat([mean_pool, max_pool], dim=-1)  # 拼接成 [B, 2C]