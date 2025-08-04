import os
import random
import librosa
import pickle
import torch
import logging
import torch.nn.functional as F
import torchaudio
logger = logging.getLogger(__name__)
import librosa

def time_stretch_waveform(waveform, target_length):
    """使用 Librosa 进行时间缩放，使得 waveform 长度变为 target_length"""
    y = waveform.squeeze().numpy()  # 转换为 NumPy 数组
    current_length = len(y)
    scale =  current_length / target_length  # 计算缩放比例

    # 只在合理范围内调整，防止过度失真
    if scale < 0.85:
        # 补零的长度
        padding = target_length - current_length
        # 在末尾补零
        wave = torch.cat([waveform, torch.zeros(1, padding)], dim=1)
    else:
        y_stretched = librosa.effects.time_stretch(y, rate=scale)  # 进行时间伸缩
        wave = torch.tensor(y_stretched).unsqueeze(0)
    return wave # 转换回 PyTorch Tensor

class CustomAudioDataset(torch.utils.data.Dataset):
    def __init__(self, config, transform=None,mode='train'):
        assert mode in ['train', 'test'], 'dataset mode must be train or test'
        if mode == 'train':
            with open(config['datasets']['train_path'], 'rb') as f:
                files_dict = pickle.load(f)
                self.audio_files = files_dict
        # elif mode == 'test':
            # with open(config['datasets']['test_path'], 'rb') as f:
                # files_dict = pickle.load(f)
                # self.audio_files = None
        self.mode = mode
        self.transform = transform
        self.fixed_length = config['datasets']['fixed_length']
        self.tensor_cut = config['datasets']['tensor_cut']

    def __len__(self):
        return self.fixed_length if self.fixed_length and len(self.audio_files) > self.fixed_length else len(self.audio_files)  
    
    def get(self, idx=None):
        """uncropped, untransformed getter with random sample feature"""
        
        if idx is not None and idx > len(self.audio_files):
            raise StopIteration
        if idx is None:
            idx = random.randrange(len(self))
        
        if self.mode == 'train':
            waveform = self.audio_files[idx]
        else:
            waveform = self.audio_files[idx]
        return waveform

    def __getitem__(self, idx):
        # waveform, sample_rate = torchaudio.load(self.audio_files.iloc[idx, :].values[0])
        # """you can preprocess the waveform's sample rate to save time and memory"""
        # if sample_rate != self.sample_rate:
        #     waveform = convert_audio(waveform, sample_rate, self.sample_rate, self.channels)
        waveform = self.get(idx)
        if self.mode == 'train':
            wave1 = waveform[1]
            
            speaker_pos = waveform[2]
            speaker_neg = waveform[3]
            selected_speaker_pos = random.sample(speaker_pos, 5)
            selected_speaker_neg = random.sample(speaker_neg, 100)

            emotion_pos = waveform[4]
            emotion_neg = waveform[5]
            selected_emotion_pos = random.sample(emotion_pos, 10)
            selected_emotion_neg = random.sample(emotion_neg, 80)
            
            content_pos = waveform[6]
            content_neg = waveform[7]
            selected_content_pos = random.sample(content_pos, 5)
            selected_content_neg = random.sample(content_neg, 50)
            # 转换为NumPy数组列表（一维）
            selected_content_pos_ = [tensor.numpy().squeeze() for tensor in selected_content_pos]
            selected_content_neg_ = [tensor.numpy().squeeze() for tensor in selected_content_neg]
            
            
            selected_speaker_pos_pad = []
            for t in selected_speaker_pos:
                if t.size()[0] > self.tensor_cut:
                    start = random.randint(0, t.size()[0]-self.tensor_cut-1) # random start point
                    selected_speaker_pos_pad.append(t[start:start+self.tensor_cut]) # cut tensor
                else:
                    repeat_times = self.tensor_cut // t.size()[0]  # 计算需要重复的次数
                    remainder = self.tensor_cut % t.size()[0]  # 剩余部分
                    repeated = t.repeat(repeat_times)  # 沿时间维度重复
                    selected_speaker_pos_pad.append(torch.cat([repeated, t[:remainder]], dim=0))  # 追加剩余部分 
            selected_speaker_pos_pad_tensor = torch.stack(selected_speaker_pos_pad).unsqueeze(1)
            
            # ================== 处理 selected_speaker_neg ==================
            selected_speaker_neg_pad = []
            for t in selected_speaker_neg:
                if self.tensor_cut > 0:
                    # 裁剪或填充逻辑
                    if t.size()[0] > self.tensor_cut:  # 假设时间维度在第二维 (C, T)
                        start = random.randint(0, t.size()[0] - self.tensor_cut - 1)
                        selected_speaker_neg_pad.append(t[start:start+self.tensor_cut])
                    else:
                        # 重复填充逻辑
                        repeat_times = self.tensor_cut // t.size()[0]
                        remainder = self.tensor_cut % t.size()[0]
                        repeated = t.repeat(repeat_times)  # 沿时间维度重复
                        selected_speaker_neg_pad.append(torch.cat([repeated, t[:remainder]], dim=0))
            selected_speaker_neg_pad_tensor = torch.stack(selected_speaker_neg_pad).unsqueeze(1)
           
            
            # ================== 处理 selected_emotion_pos ==================
            selected_emotion_pos_pad = []
            for t in selected_emotion_pos:
                if self.tensor_cut > 0:
                    # 裁剪或填充逻辑
                    if t.size()[0] > self.tensor_cut:  # 假设时间维度在第二维 (C, T)
                        start = random.randint(0, t.size()[0] - self.tensor_cut - 1)
                        selected_emotion_pos_pad.append(t[start:start+self.tensor_cut])
                    else:
                        repeat_times = self.tensor_cut // t.size()[0]
                        remainder = self.tensor_cut % t.size()[0]
                        repeated = t.repeat(repeat_times)
                        selected_emotion_pos_pad.append(torch.cat([repeated, t[:remainder]], dim=0))
            selected_emotion_pos_pad_tensor = torch.stack(selected_emotion_pos_pad).unsqueeze(1)

            # ================== 处理 selected_emotion_neg ==================
            selected_emotion_neg_pad = []
            for t in selected_emotion_neg:
                if self.tensor_cut > 0:
                    # 裁剪或填充逻辑
                    if t.size()[0] > self.tensor_cut:  # 假设时间维度在第二维 (C, T)
                        start = random.randint(0, t.size()[0] - self.tensor_cut - 1)
                        selected_emotion_neg_pad.append(t[start:start+self.tensor_cut])
                    else:
                        repeat_times = self.tensor_cut // t.size()[0]
                        remainder = self.tensor_cut % t.size()[0]
                        repeated = t.repeat(repeat_times)
                        selected_emotion_neg_pad.append(torch.cat([repeated, t[:remainder]], dim=0))
            selected_emotion_neg_pad_tensor = torch.stack(selected_emotion_neg_pad).unsqueeze(1)
            
            # reference = wave1
            # selected_content_pos_pad = []
            # # 假设第一个片段是基准
            # for t in selected_content_pos:
            #     t = t.unsqueeze(0)
            #     aligned_t = time_stretch_waveform(t, reference.shape[1])
            #     selected_content_pos_pad.append(aligned_t)
            # selected_content_pos_pad_tensor = torch.stack(selected_content_pos_pad)
            
            # selected_content_neg_pad = []
            # for t in selected_content_neg:
            #     if self.tensor_cut > 0:
            #         if t.size()[0] > self.tensor_cut:
            #             start = random.randint(0, t.size()[0] - self.tensor_cut - 1)
            #             selected_content_neg_pad.append(t[start:start+self.tensor_cut])
            #         else:
            #             repeat_times = self.tensor_cut // t.size()[0]
            #             remainder = self.tensor_cut % t.size()[0]
            #             repeated = t.repeat(repeat_times)
            #             selected_content_neg_pad.append(torch.cat([repeated, t[:remainder]], dim=0))
            # selected_content_neg_pad_tensor = torch.stack(selected_content_neg_pad).unsqueeze(1)
            return wave1,selected_speaker_pos_pad_tensor,selected_speaker_neg_pad_tensor,selected_emotion_pos_pad_tensor,selected_emotion_neg_pad_tensor,selected_content_pos_,selected_content_neg_
        
        else:
            wave_test = waveform[1]
            return wave_test
        
def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.permute(1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    batch = batch.permute(0, 2, 1)
    return batch
def collate_fn(batch):
    if len(batch[0]) == 7:
        tensors_wav = []
        tensors_speaker_pos = []
        tensors_speaker_neg = []
        tensors_emotion_pos = []
        tensors_emotion_neg = []
        tensors_content_pos = []
        tensors_content_neg = []
        wav2vec_numpy = []
        for waveform in batch:
            tensors_wav += [waveform[0]]
            tensors_speaker_pos += [waveform[1]]
            tensors_speaker_neg += [waveform[2]]
            tensors_emotion_pos += [waveform[3]]
            tensors_emotion_neg += [waveform[4]]
            tensors_content_pos += [waveform[5]]
            tensors_content_neg += [waveform[6]]
            wav2vec_numpy += [waveform[0].squeeze().numpy()]
        tensors_wav = pad_sequence(tensors_wav)
        tensors_speaker_pos = torch.stack(tensors_speaker_pos)
        tensors_speaker_neg = torch.stack(tensors_speaker_neg)
        tensors_emotion_pos = torch.stack(tensors_emotion_pos)
        tensors_emotion_neg = torch.stack(tensors_emotion_neg)
        return tensors_wav,tensors_speaker_pos,tensors_speaker_neg,tensors_emotion_pos,tensors_emotion_neg,tensors_content_pos,tensors_content_neg,wav2vec_numpy
    else:
        tensors_x = []
        for waveform in batch:
            tensors_x += [waveform]
        tensors_x = pad_sequence(tensors_x)
        return tensors_x
 