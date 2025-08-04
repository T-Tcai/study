import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy

class TemporalMI_Estimator(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化三个互信息计算模块
        self.mi_sc = MI_Block(256, 512)  # 音色-内容
        self.mi_se = MI_Block(256, 256)   # 音色-情感
        self.mi_ec = MI_Block(256, 512) # 情感-内容


    def forward(self, speaker, emotion, content):
        """前向计算过程
        输入:
            speaker: [bs, 256]          (静态音色特征)
            emotion: [bs, 256, T1]      (动态情感特征)
            content: [bs, 512, T2]      (动态语义特征)
        """
        # 对齐音色与内容的时间步 (T2)
        
        speaker_align_con = speaker.unsqueeze(-1).expand(-1,-1,content.shape[-1])
        speaker_align_emo = speaker.unsqueeze(-1).expand(-1,-1,emotion.shape[-1])
        emotion_align = F.interpolate(emotion, content.shape[-1], mode='linear')
        
        loss_sc = self.mi_sc(speaker_align_con, content)

        # 对齐音色与情感的时间步 (T1)
        loss_se = self.mi_se(speaker_align_emo, emotion)

        # 对齐情感与内容的时间步
        loss_ec = self.mi_ec(emotion_align, content)

        # 互信息数值估计
        mi_sc = self.mi_sc.estimate_mi(speaker_align_con, content)
        mi_se = self.mi_se.estimate_mi(speaker_align_emo, emotion)
        mi_ec = self.mi_ec.estimate_mi(emotion_align, content)

        return (loss_sc, loss_se, loss_ec), (mi_sc, mi_se, mi_ec)

class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size=256):
        super(MINE, self).__init__()
        self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))
    
    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        x_flat = x_samples.permute(0, 2, 1).contiguous().view(-1, x_samples.size(1))  # [bs*T, 128]
        y_flat = y_samples.permute(0, 2, 1).contiguous().view(-1, y_samples.size(1))  # [bs*T, 128]
        # shuffle and concatenate
        sample_size = y_flat.size(0)
        random_index = torch.randperm(sample_size)
        y_shuffle = y_flat[random_index]
        
        T0 = self.T_func(torch.cat([x_flat, y_flat], dim = -1))
        T1 = self.T_func(torch.cat([x_flat, y_shuffle], dim = -1))

        lower_bound = T0.mean() - torch.log(T1.exp().mean())

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound
    
    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)
    
class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        # CLUB MLP 改成 2 层 + smaller hidden
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, y_dim),
        )
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, y_dim),
            nn.Tanh(),
        )

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
     
    def loglikeli(self, x_samples, y_samples):
        x_flat = x_samples.permute(0, 2, 1).contiguous().view(-1, x_samples.size(1))  # [bs*t, x_dim]
        y_flat = y_samples.permute(0, 2, 1).contiguous().view(-1, y_samples.size(1))  # [bs*t, y_dim]
        mu, logvar = self.get_mu_logvar(x_flat)
        return (-(mu - y_flat) ** 2 / (logvar.exp() + 1e-6) - logvar).sum(dim=-1).mean()

    

    def forward(self, x_samples, y_samples):
        x_flat = x_samples.permute(0, 2, 1).contiguous().view(-1, x_samples.size(1))  # [bs*t, x_dim]
        y_flat = y_samples.permute(0, 2, 1).contiguous().view(-1, y_samples.size(1))  # [bs*t, y_dim]
        mu, logvar = self.get_mu_logvar(x_flat)
        
        sample_size = x_flat.shape[0]
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_flat)**2 / logvar.exp()
        negative = - (mu - y_flat[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)



class MI_Block(nn.Module):
    def __init__(self, dim_x, dim_y, hidden_size=128):
        super().__init__()
        
        self.mine = MINE(dim_x, dim_y, hidden_size)
        self.club = CLUBSample(dim_x, dim_y, hidden_size)
    def _normalize_features(self, x, y):
        # x: [bs, C, T]
        x = F.layer_norm(x, x.shape[1:])  # 归一化 C 维度
        y = F.layer_norm(y, y.shape[1:])  # 归一化 C 维度
        return x, y
 
        # x = F.layer_norm(x.permute(0,2,1), [x.size(1)]).permute(0,2,1)
        # y = F.layer_norm(y.permute(0,2,1), [y.size(1)]).permute(0,2,1)
        # return x, y
    def forward(self, x, y):
        x_flat, y_flat = self._normalize_features(x, y)
        mine_val = self.mine(x_flat, y_flat)
        club_val = self.club(x_flat, y_flat)
        mi_lb_loss = -mine_val
        mi_ub_loss = self.club.learning_loss(x_flat, y_flat)
        loss = mi_lb_loss + mi_ub_loss + F.relu(club_val - mine_val)
        return loss

    def estimate_mi(self, x, y):
        
        # 2. 归一化
        x_flat, y_flat = self._normalize_features(x, y)
        return F.relu(self.club(x_flat, y_flat)).mean()
