import os
import random
import librosa
import pickle
import torch
import logging
import torch.nn.functional as F
logger = logging.getLogger(__name__)
# Load model directly

class CustomAudioDataset(torch.utils.data.Dataset):
    def __init__(self, config, transform=None,mode='train'):
        assert mode in ['train', 'test'], 'dataset mode must be train or test'
        if mode == 'train':
            with open(config['datasets']['train_path'], 'rb') as f:
                files_dict = pickle.load(f)
                self.audio_files = files_dict
        elif mode == 'test':
            with open(config['datasets']['test_path'], 'rb') as f:
                files_dict = pickle.load(f)
                self.audio_files = files_dict
                
        self.mode = mode
        self.transform = transform
        self.fixed_length = config['datasets']['fixed_length']

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
            wave = waveform[1]
            if wave.size()[1] > 32000:
                start = random.randint(0, wave.size()[1]-32000-1) # random start point
                wave = wave[:, start:start+32000] # cut tensor
            return wave
            
        else:
            wave = waveform[1]
            return wave
        
def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.permute(1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    batch = batch.permute(0, 2, 1)
    return batch
def collate_fn(batch):
    tensors_wav = []
    wav2vec_numpy = []
    for waveform in batch:
        tensors_wav += [waveform]
        wav2vec_numpy += [waveform.squeeze().numpy()]
    tensors_wav = pad_sequence(tensors_wav)
    return tensors_wav,wav2vec_numpy

 