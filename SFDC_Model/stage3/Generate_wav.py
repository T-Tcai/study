import numpy as np
import torch
from torch import nn
from transformers import WavLMModel
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
from stage2.decouping_model_wav2vec import Decouping_model
from stage1.encodec_model import Encodec_model
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model
from vector_quantize_pytorch import VectorQuantize

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
import math


class EnhancedEmotionEncoder(nn.Module):
    def __init__(self, emo_dim=256, hid_dim=512):
        super().__init__()
        self.emo_net = nn.Sequential(
            nn.Conv1d(emo_dim, hid_dim, kernel_size=5, padding=2),
            nn.InstanceNorm1d(hid_dim),
            nn.GELU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=5, padding=2),
            nn.InstanceNorm1d(hid_dim),
            nn.GELU(),
        )
        self.res_conv = nn.Conv1d(emo_dim, hid_dim, 1)

        # 注意力权重生成网络
        self.attn_proj = nn.Sequential(
            nn.Conv1d(hid_dim, 1, kernel_size=1),  # 投影到标量注意力得分
            nn.Softmax(dim=-1),  # 沿时间维度归一化
        )

        # 降维投影（保持一致性）
        self.fc = nn.Linear(hid_dim, hid_dim)

    def forward(self, emo_feat):  # [B, 256, T]
        x = self.emo_net(emo_feat) + self.res_conv(emo_feat)  # [B, hid_dim, T]

        attn_weight = self.attn_proj(x)  # [B, 1, T]
        attn_feat = (x * attn_weight).sum(dim=-1)  # [B, hid_dim]

        x_reduced = self.fc(attn_feat)  # [B, hid_dim]
        return x_reduced


class SemanticNet(nn.Module):
    def __init__(self, sem_dim, hid_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(sem_dim, hid_dim, 5, padding=2),  # 卷积层 1：扩大感受野
            nn.InstanceNorm1d(hid_dim),
            nn.GELU(),
            nn.Conv1d(hid_dim, hid_dim, 3, padding=1),  # 卷积层 2：细化局部信息
            nn.InstanceNorm1d(hid_dim),
            nn.GELU(),
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hid_dim,
                nhead=4,
                dim_feedforward=hid_dim * 2,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=2,
        )

    def forward(self, sem_feat):  # 输入：[B, sem_dim, T]
        x = self.conv(sem_feat)  # [B, hid_dim, T]
        x = x.transpose(1, 2)  # [B, T, hid_dim]
        x = self.transformer(x)  # [B, T, hid_dim]
        return x.transpose(1, 2)  # [B, hid_dim, T]


class DynamicFusion(nn.Module):
    """统一投影到语义维度的双层融合模块"""

    def __init__(self, sem_dim=512, emo_dim=256, voice_dim=256, hid_dim=512):
        super().__init__()
        self.n_channels = hid_dim
        # self.emo_gate_bias = nn.Parameter(torch.tensor(0.3))  # 初始偏置值，可训练
        # 语义特征处理（增加深度）
        self.sem_net = SemanticNet(sem_dim, hid_dim)

        # 情感特征处理（带残差连接）
        self.emo_proj = EnhancedEmotionEncoder(emo_dim=emo_dim, hid_dim=hid_dim)

        # 音色特征处理（增强非线性）
        self.voice_proj = nn.Sequential(
            nn.Linear(voice_dim, hid_dim * 2),
            nn.GELU(),
            nn.Linear(hid_dim * 2, hid_dim),
        )

        # -------- 独立门控 --------
        self.gate_net = nn.ModuleDict(
            {
                "sem": self._build_gate_branch(hid_dim),
                "emo": self._build_gate_branch(hid_dim),
                "voice": self._build_gate_branch(hid_dim),
            }
        )
        # 融合后微调层（可选）
        self.final_fuse = nn.Sequential(
            nn.Conv1d(hid_dim, hid_dim, 1), nn.InstanceNorm1d(hid_dim), nn.GELU()
        )
        self.norm = nn.LayerNorm(hid_dim)

    def _build_gate_branch(self, channels):
        # 输出多个通道的门控（n_channels 个通道，每个通道控制时间步）
        return nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv1d(channels, self.n_channels, 1),  # 输出 n_channels 通道
        )

    def forward(self, sem_feat, emo_feat, voice_feat):
        """
        输入:
            sem_feat: [B,512,T2]  # 语义特征
            emo_feat: [B,256,T1]  # 情感特征
            voice_feat: [B,256]    # 音色特征
        """
        sem = self.sem_net(sem_feat)  # [B, hid_dim, T2]

        # ---- 情感全局 embedding ----
        emo_emb = self.emo_proj(emo_feat)  # [B, hid_dim]
        # emo_aligned = F.interpolate(emo_emb, sem.size(-1), mode='nearest')  # [B, hid_dim, T2]
        emo_aligned = emo_emb.unsqueeze(-1).expand(-1, -1, sem.size(-1))

        # ---- 音色 ----
        voice = self.voice_proj(voice_feat).unsqueeze(-1).expand(-1, -1, sem.size(-1))

        sem_gate = self.gate_net["sem"](sem)  # [B, hid_dim, T2]
        emo_gate = self.gate_net["emo"](emo_aligned)  # [B, hid_dim, T2]
        voice_gate = self.gate_net["voice"](voice)  # [B, hid_dim, T2]
        sem_gate = self.norm(sem_gate.transpose(1, 2)).transpose(
            1, 2
        )  # [B, T, C] → LayerNorm → [B, C, T]
        emo_gate = self.norm(emo_gate.transpose(1, 2)).transpose(1, 2)
        voice_gate = self.norm(voice_gate.transpose(1, 2)).transpose(1, 2)

        sem_gate = torch.sigmoid(sem_gate)
        emo_gate = torch.sigmoid(emo_gate)
        voice_gate = torch.sigmoid(voice_gate)

        # ---- 融合 ----
        fused = (
            sem_gate * sem  # [B, hid_dim, T2]
            + emo_gate * emo_aligned  # [B, hid_dim, T2]
            + voice_gate * voice  # [B, hid_dim, T2]
        )
        fused = self.final_fuse(fused)

        # ---- Gate 平衡 loss ----
        gate_mean = torch.stack(
            [
                sem_gate.mean(dim=[1, 2]),
                emo_gate.mean(dim=[1, 2]),
                voice_gate.mean(dim=[1, 2]),
            ],
            dim=1,
        )  # [B, 3]
        gate_balance_loss = gate_mean.std(dim=1).mean()

        print("Gate avg weights (sem / emo / voice):", gate_mean.detach().cpu().numpy())
        return fused, gate_balance_loss
        # def forward_v1(self, sem_feat, emo_feat, voice_feat,a):
        """
        输入:
            sem_feat: [B,512,T2]  # 语义特征
            emo_feat: [B,256,T1]  # 情感特征
            voice_feat: [B,256]    # 音色特征
        """
        sem = self.sem_net(sem_feat)  # [B, hid_dim, T2]

        # ---- 情感全局 embedding ----
        emo_emb = self.emo_proj(emo_feat)  # [B, hid_dim]
        # emo_aligned = F.interpolate(emo_emb, sem.size(-1), mode='nearest')  # [B, hid_dim, T2]
        emo_aligned = emo_emb.unsqueeze(-1).expand(-1, -1, sem.size(-1))

        # ---- 音色 ----
        voice = self.voice_proj(voice_feat).unsqueeze(-1).expand(-1, -1, sem.size(-1))

        sem_gate = self.gate_net["sem"](sem)  # [B, hid_dim, T2]
        emo_gate = self.gate_net["emo"](emo_aligned)  # [B, hid_dim, T2]
        voice_gate = self.gate_net["voice"](voice)  # [B, hid_dim, T2]
        sem_gate = self.norm(sem_gate.transpose(1, 2)).transpose(
            1, 2
        )  # [B, T, C] → LayerNorm → [B, C, T]
        emo_gate = self.norm(emo_gate.transpose(1, 2)).transpose(1, 2)
        voice_gate = self.norm(voice_gate.transpose(1, 2)).transpose(1, 2)

        sem_gate = torch.sigmoid(sem_gate)
        emo_gate = torch.sigmoid(emo_gate)
        voice_gate = torch.sigmoid(voice_gate)

        # ---- 融合 ----
        fused = (
            a * sem_gate * sem  # [B, hid_dim, T2]
            + emo_gate * emo_aligned  # [B, hid_dim, T2]
            + voice_gate * voice  # [B, hid_dim, T2]
        )
        fused = self.final_fuse(fused)

        # ---- Gate 平衡 loss ----
        gate_mean = torch.stack(
            [
                sem_gate.mean(dim=[1, 2]),
                emo_gate.mean(dim=[1, 2]),
                voice_gate.mean(dim=[1, 2]),
            ],
            dim=1,
        )  # [B, 3]
        gate_balance_loss = gate_mean.std(dim=1).mean()

        print("Gate avg weights (sem / emo / voice):", gate_mean.detach().cpu().numpy())
        return fused, gate_balance_loss


class TemporalBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(dim, dim, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class MultiScaleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 三个不同感受野：常规、宽卷积、空洞
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(channels, channels, kernel_size=3, padding=1, dilation=1),
                nn.Conv1d(channels, channels, kernel_size=5, padding=2, dilation=1),
                nn.Conv1d(channels, channels, kernel_size=3, padding=4, dilation=4),
            ]
        )
        self.act = nn.GELU()
        self.fuse = nn.Conv1d(channels * 3, channels, kernel_size=1)

    def forward(self, x):
        outs = []
        for conv in self.convs:
            out = conv(x)
            out = self.act(out)
            outs.append(out)
        out = torch.cat(outs, dim=1)  # [B, C*3, T]
        return self.fuse(out)  # [B, C, T]


class VoiceStyleInjector(nn.Module):
    def __init__(self, spk_dim=256, emo_dim=256, style_dim=128, n_layers=8):
        super().__init__()
        self.n_layers = n_layers
        self.style_dim = style_dim

        self.voice_mlp = nn.Sequential(
            nn.Linear(spk_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, n_layers * style_dim * 2),
        )

        self.emo_conv = nn.Sequential(
            nn.Conv1d(emo_dim, 512, 3, padding=1),
            TemporalBlock(512),
            nn.InstanceNorm1d(512),
        )
        self.emo_attention = nn.MultiheadAttention(
            embed_dim=512, num_heads=4, batch_first=True
        )

        self.emo_mlp = nn.Sequential(
            nn.Linear(512, 1024), nn.GELU(), nn.Linear(1024, n_layers * style_dim * 2)
        )

    def forward(self, voice_feat, emo_feat):
        B, _, T = emo_feat.size()

        voice_params = self.voice_mlp(voice_feat).view(
            B, self.n_layers, 2 * self.style_dim
        )
        voice_mu, voice_logvar = voice_params.chunk(2, dim=-1)
        voice_style = voice_mu

        emo_hidden = self.emo_conv(emo_feat)
        query = emo_hidden.mean(dim=2, keepdim=True).transpose(1, 2)
        key_value = emo_hidden.transpose(1, 2)
        pooled, _ = self.emo_attention(query, key_value, key_value)
        emo_params = self.emo_mlp(pooled.squeeze(1)).view(
            B, self.n_layers, 2 * self.style_dim
        )
        emo_mu, emo_logvar = emo_params.chunk(2, dim=-1)
        emo_style = emo_mu

        return voice_style, emo_style

    # def forward_v1(self, voice_feat, emo_feat1,emo_feat2,a):
    #     B, _, T = emo_feat1.size()

    #     voice_params = self.voice_mlp(voice_feat).view(B, self.n_layers, 2 * self.style_dim)
    #     voice_mu, voice_logvar = voice_params.chunk(2, dim=-1)
    #     voice_style = voice_mu

    #     emo_hidden1 = self.emo_conv(emo_feat1)
    #     query1 = emo_hidden1.mean(dim=2, keepdim=True).transpose(1, 2)
    #     key_value1 = emo_hidden1.transpose(1, 2)
    #     pooled1, _ = self.emo_attention(query1, key_value1, key_value1)
    #     emo_params1 = self.emo_mlp(pooled1.squeeze(1)).view(B, self.n_layers, 2 * self.style_dim)
    #     emo_mu1, emo_logvar1 = emo_params1.chunk(2, dim=-1)
    #     emo_style1 = emo_mu1

    #     emo_hidden2 = self.emo_conv(emo_feat2)
    #     query2 = emo_hidden2.mean(dim=2, keepdim=True).transpose(1, 2)
    #     key_value2 = emo_hidden2.transpose(1, 2)
    #     pooled2, _ = self.emo_attention(query2, key_value2, key_value2)
    #     emo_params2 = self.emo_mlp(pooled2.squeeze(1)).view(B, self.n_layers, 2 * self.style_dim)
    #     emo_mu2, emo_logvar2 = emo_params2.chunk(2, dim=-1)
    #     emo_style2 = emo_mu2

    #     emo_style = (1 - a) * emo_style1 + a * emo_style2

    #     return voice_style, emo_style


class StyleAdaptiveNorm(nn.Module):
    def __init__(self, dim, style_dim=128):
        super().__init__()
        self.norm = nn.InstanceNorm1d(dim, affine=False)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=style_dim, num_heads=4, batch_first=True
        )
        self.style_proj = nn.Sequential(
            nn.Linear(style_dim, 128), nn.GELU(), nn.Linear(128, dim * 3)
        )

    def forward(self, x, voice_style=None, emo_style=None):
        if voice_style is not None and emo_style is not None:
            alpha = 50
            voice_seq = voice_style.unsqueeze(1)
            emo_seq = emo_style.unsqueeze(1)

            mix_query = alpha * voice_seq + (1 - alpha) * emo_seq

            # mix_query = mix_query / (abs(alpha) + abs(1 - alpha) + 1e-6)

            fused_style, _ = self.cross_attn(mix_query, emo_seq, emo_seq)
            fused_style = fused_style.squeeze(1)
        elif voice_style is not None:
            fused_style = voice_style
        elif emo_style is not None:
            fused_style = emo_style
        else:
            raise ValueError("必须至少提供一种风格")

        params = self.style_proj(fused_style).unsqueeze(-1)
        gamma, beta, residual = params.chunk(3, dim=1)

        return self.norm(x) * (1 + gamma.tanh()) + beta + 0.2 * residual.tanh() * x

    # def forward(self, x, voice_style=None, emo_style=None):
    #     if voice_style is not None and emo_style is not None:

    #         voice_seq = voice_style.unsqueeze(1)
    #         emo_seq = emo_style.unsqueeze(1)
    #         fused_style, _ = self.cross_attn(voice_seq, emo_seq, emo_seq)
    #         fused_style = fused_style.squeeze(1)
    #     elif voice_style is not None:
    #         fused_style = voice_style
    #     elif emo_style is not None:
    #         fused_style = emo_style
    #     else:
    #         raise ValueError("必须至少提供一种风格")

    #     params = self.style_proj(fused_style).unsqueeze(-1)
    #     gamma, beta, residual = params.chunk(3, dim=1)

    #     return self.norm(x) * (1 + gamma.tanh()) + beta + 0.2 * residual.tanh() * x


class WaveGenerator(nn.Module):
    def __init__(
        self, in_dim=512, style_dim=128, out_dim=512, n_layers=8, emo_layers=3
    ):
        super().__init__()
        self.n_layers = n_layers
        self.emo_layers = emo_layers

        self.encoder_conv = nn.Conv1d(in_dim, 512, 3, padding=1)
        self.encoder_norm = StyleAdaptiveNorm(512, style_dim=style_dim)
        self.encoder_act = nn.GELU()

        self.res_blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "multi_scale": MultiScaleBlock(512),
                        "norm": StyleAdaptiveNorm(512, style_dim=style_dim),
                        "act": nn.GELU(),
                        "final_conv": nn.Conv1d(512, 512, 3, padding=1),
                    }
                )
                for _ in range(n_layers)
            ]
        )

        self.out_proj = nn.Sequential(
            nn.Conv1d(512, out_dim, 1), nn.InstanceNorm1d(out_dim), nn.GELU()
        )

    def forward(self, x, voice_styles=None, emo_styles=None):
        B, _, T = x.shape
        if voice_styles is not None and emo_styles is not None:
            assert voice_styles.shape == emo_styles.shape

        # Apply initial encoder
        x = self.encoder_conv(x)
        x = self.encoder_norm(x, None, emo_styles[:, 0])  # Use the first emotion style
        x = self.encoder_act(x)

        # Residual blocks with MultiScaleBlock and StyleAdaptiveNorm
        for i in range(self.n_layers):
            residual = x
            x = self.res_blocks[i]["multi_scale"](x)

            # Get style vectors for current layer
            cur_voice = voice_styles[:, i] if voice_styles is not None else None
            cur_emo = emo_styles[:, i] if emo_styles is not None else None

            # Apply StyleAdaptiveNorm to the output of MultiScaleBlock
            x = self.res_blocks[i]["norm"](x, cur_voice, cur_emo)
            x = self.res_blocks[i]["act"](x)
            x = self.res_blocks[i]["final_conv"](x)

            # Add residual connection
            x = residual + 0.5 * x

        return self.out_proj(x)


# 辅助模块
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.InstanceNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.InstanceNorm1d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generate_model(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16_000,
        channels: int = 1,
        dimension: int = 256,
        bins: int = 1024,
        max_duration: int = 10,
    ):
        super().__init__()
        self.douplemodel = Decouping_model().to(device)

        # 加载 SEANet 预训练权重
        model_checkpoint = torch.load(
            "/home/xinyue/code/TCEmodel/stage2/checkpoints_f0_en/bs12_epoch30_lr3e-4.pt",
            map_location=device,
        )
        self.douplemodel.load_state_dict(model_checkpoint["model_state_dict"])
        # 加载 SEANet 编码器
        for p in self.douplemodel.parameters():
            p.requires_grad = False

        self.fusion_net = DynamicFusion()
        self.style_net = VoiceStyleInjector()
        self.generator = WaveGenerator()

        seanet = Encodec_model().to(device)
        # 加载 SEANet 预训练权重
        model_checkpoint = torch.load(
            "/home/xinyue/code/TCEmodel/stage1/checkpoints_512/bs24_epoch100_lr3e-4.pt",
            map_location=device,
        )
        seanet.load_state_dict(model_checkpoint["model_state_dict"])
        # 加载 SEANet 编码器
        self.decoder = seanet.decoder
        # 冻结 SEANet 参数
        for p in self.decoder.parameters():
            p.requires_grad = True

    def forward(self, wave, wav_numpt):
        """
        wav1: 主要输入语音
        wav2: 说话人匹配音频
        wav3: 情感匹配音频
        """
        # 编码条件
        with torch.no_grad():
            feat = self.douplemodel.extract_feature(wave, wav_numpt)
            timbre = feat["speaker_feat"].detach()  # [B,256]
            emotion = feat["emotion_feat"].detach()  # [B,256,T]
            semantic = feat["content_feat"].detach()  # [B,512,T]

        # 三通道动态融合（关键修改点）✅
        fused, gate_balance_loss = self.fusion_net(semantic, emotion, timbre)

        # 计算对抗损失

        voice_styles, emo_styles = self.style_net(timbre, emotion)
        # 分层特征注入生成
        gen_feat = self.generator(fused, voice_styles, emo_styles)  # [B,512,T]

        # 解码器输出
        output = self.decoder(gen_feat)

        # 最终长度调整
        output = F.interpolate(
            output, size=wave.shape[-1], mode="linear", align_corners=True
        )

        with torch.no_grad():
            feat_fake = self.douplemodel.extract_feature(
                output, list(output.squeeze().cpu().numpy())
            )
        gen_spk_loss = (
            1
            - F.cosine_similarity(
                feat["speaker_feat"], feat_fake["speaker_feat"]
            ).mean()
        )

        # 4. 情感一致性损失
        gen_emo_loss = (
            1
            - F.cosine_similarity(
                feat["emotion_feat"], feat_fake["emotion_feat"], dim=-1
            ).mean()
        )

        gen_con_loss = (
            1
            - F.cosine_similarity(
                feat["content_feat"], feat_fake["content_feat"], dim=-1
            ).mean()
        )

        # #门控损失
        # target_gate = torch.ones_like(gate_mean) / 3  # 均衡期望
        # loss_gate_reg = F.mse_loss(gate_mean, target_gate).mean(-1)
        # loss_gate += 0.1 * loss_gate_reg
        return output, gen_con_loss, gen_spk_loss, gen_emo_loss, gate_balance_loss

    def generate_audio(self, tim_wav, emo_wav, con_wav):
        """
        wav1: 主要输入语音
        wav2: 说话人匹配音频
        wav3: 情感匹配音频
        """
        # 编码条件
        with torch.no_grad():
            feat = self.douplemodel.extract_feature_gen(
                tim_wav, emo_wav, con_wav
            )  # [B,512,T]
        # 三通道动态融合（关键修改点）✅
        fused, _ = self.fusion_net(
            feat["content_feat"], feat["emotion_feat"], feat["speaker_feat"]
        )

        voice_styles, emo_styles = self.style_net(
            feat["speaker_feat"], feat["emotion_feat"]
        )
        # 分层特征注入生成
        gen_feat = self.generator(fused, voice_styles, emo_styles)  # [B,512,T]
        # 解码器输出
        output = self.decoder(gen_feat)
        return output

    # def p_test(self, tim_wav,emo_wav1,con_wav,a):
    #     """
    #     wav1: 主要输入语音
    #     wav2: 说话人匹配音频
    #     wav3: 情感匹配音频
    #     """
    #     # 编码条件
    #     with torch.no_grad():
    #         feat = self.douplemodel.extract_feature_gen(tim_wav,emo_wav1,con_wav)           # [B,512,T]
    #      # 三通道动态融合（关键修改点）✅

    #     fused, _ = self.fusion_net.forward_v1(feat['content_feat'], feat['emotion_feat'], feat['speaker_feat'],a)

    #     voice_styles, emo_styles = self.style_net.forward(feat['speaker_feat'], feat['emotion_feat'])
    #     # 分层特征注入生成
    #     gen_feat = self.generator(fused,voice_styles, emo_styles)  # [B,512,T]
    #     # 解码器输出
    #     output = self.decoder(gen_feat)
    #     return output

    # def convert_audio(self, emo_wav,tim_wav,con_wav):
    #     """
    #     wave: [B, 1, T] 输入波形
    #     return: 生成的波形 [B, 1, T]
    #     """
    #     with torch.no_grad():
    #         emo_res = self.douplemodel.extract_feature(emo_wav)
    #         tim_res = self.douplemodel.extract_feature(tim_wav)
    #         con_res = self.douplemodel.extract_feature(con_wav)

    #         timbre = tim_res['speaker_feat'].squeeze(-1).detach()
    #         emotion = emo_res['emotion_feat'].detach()
    #         semantic = con_res['content_feat'].detach()
    #         fused, _ = self.fusion_net(semantic, emotion)
    #         print("convert")
    #         styles = self.style_net(timbre)
    #         gen_wav = self.generator(fused, styles)
    #         print(gen_wav.shape)
    #         # Step 2: 上采样 fused 到 T
    #         output = self.decoder(gen_wav)
    #         print(output.shape)

    #     return output
