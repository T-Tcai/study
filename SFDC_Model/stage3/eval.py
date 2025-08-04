import torch
import torchaudio
import librosa
import soundfile as sf
import yaml
from Generate_wav import Generate_model
import pickle
import random

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model = Generate_model().to(device)
model_checkpoint = torch.load("/home/xinyue/code/TCEmodel/stage3/checkpoints_960h_enhance/bs72_epoch600_lr3e-4.pt", map_location=device)
model.load_state_dict(model_checkpoint['model_state_dict'])   
model.eval()
#直接输入wav
#
# spk_wav = (
#     "/home/xinyue/code/test/data/seen/Happy/0012_001046.wav"
# )
# # "/mnt/mydisk/yxy_data/data/MEAD/M003/audio/angry/level_3/030.wav"
# emo_wav = (
#     "/home/xinyue/code/test/data/unseen/Happy/0013_000706.wav"
# )
# # "/mnt/mydisk/yxy_data/data/MEAD/W009/audio/sad/level_3/021.wav"
# con_wav = (
#     "/home/xinyue/code/test/data/seen/Sad/0017_001346.wav"
# )
# # "/mnt/mydisk/yxy_data/data/MEAD/W014/audio/happy/level_3/028.wav"

spk_wav = (
    "/home/xinyue/code/TCEmodel/Data/ESD/0017/Surprise/0017_001477.wav"
)
# "/mnt/mydisk/yxy_data/data/MEAD/M003/audio/angry/level_3/030.wav"
emo_wav1 = (
    "/home/xinyue/code/test/data/seen/Angry/0011_000440.wav"
)
emo_wav2 = (
    "/home/xinyue/code/test/data/seen/Happy/0014_000789.wav"
)
# "/mnt/mydisk/yxy_data/data/MEAD/W009/audio/sad/level_3/021.wav"
con_wav = (
    "/home/xinyue/code/TCEmodel/Data/ESD/0012/Neutral/0012_000175.wav"
)

speaker_wav,_= librosa.load(spk_wav, sr=16000)
emotion_wav1,_ = librosa.load(emo_wav1, sr=16000)
emotion_wav2,_ = librosa.load(emo_wav2, sr=16000)
content_wav,_ = librosa.load(con_wav, sr=16000)
speaker_wav = torch.from_numpy(speaker_wav).unsqueeze(0).to(device)
emotion_wav1 = torch.from_numpy(emotion_wav1).unsqueeze(0).to(device)
emotion_wav2 = torch.from_numpy(emotion_wav2).unsqueeze(0).to(device)
content_wav = torch.from_numpy(content_wav).unsqueeze(0).to(device)

# with open("/home/xinyue/code/ControlTCEmodel/data/train_data.pkl", 'rb') as f:
#     files_dict = pickle.load(f)
# idx1 = random.randrange(len(files_dict))
# idx2 = random.randrange(len(files_dict))
# idx3 = random.randrange(len(files_dict))
# key1 = files_dict[idx1][0]
# key2 = files_dict[idx2][0]
# key3 = files_dict[idx3][0]

# content1 = files_dict[idx1][2]
# content2 = files_dict[idx2][2]
# content3 = files_dict[idx3][2]

# speaker_wav = files_dict[idx1][1].to(device)
# emotion_wav = files_dict[idx2][1].to(device)
# content_wav = files_dict[idx3][1].to(device)

# print(f"speaker_wav:{key1}")
# print(f"emotion_wav:{key2}")
# print(f"content_wav:{key3}")
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
output_dir = "generated_audios"

os.makedirs(output_dir, exist_ok=True)
with torch.no_grad():
    f0_means = []
    energy_means = []
    alpha_list = [0.1, 1, 5, 10, 15, 20]
    for alpha in alpha_list:
        output_wav = model.p_test(speaker_wav.unsqueeze(0),emotion_wav1.unsqueeze(0),content_wav.squeeze(),alpha)
        torchaudio.save(f'/home/xinyue/code/TCEmodel/stage3/eval/tim.wav', speaker_wav.cpu(), 16000)
        torchaudio.save(f'/home/xinyue/code/TCEmodel/stage3/eval/emo1.wav', emotion_wav1.cpu(), 16000)
        torchaudio.save(f'/home/xinyue/code/TCEmodel/stage3/eval/con.wav', content_wav.cpu(), 16000)
        torchaudio.save(os.path.join(output_dir, f"Angry_Happy_audio_con_{alpha:.1f}.wav"), output_wav.squeeze(0).cpu(), 16000)
        output_wav = output_wav.cpu().numpy()
        # F0: 使用 librosa.yin
        f0 = librosa.yin(output_wav, fmin=50, fmax=500, sr=16000)
        f0_voiced = f0[f0 > 0]
        f0_mean = np.mean(f0_voiced)
        f0_means.append(f0_mean)

        # 能量：frame-wise RMS
        energy = librosa.feature.rms(y=output_wav)[0]
        energy_mean = np.mean(energy ** 2)
        energy_means.append(energy_mean)
    print(f0_means)
    print(energy_means)
    # 连续性可视化
    plt.figure(figsize=(10, 4))
    plt.plot(alpha_list, f0_means, marker='o', label="F0")
    plt.plot(alpha_list, energy_means, marker='s', label="energy")
    plt.xlabel("alpha α")
    plt.ylabel("Value")
    plt.title("The trend of alpha α")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig('continuity_visualization_Angry_Happy.png')





