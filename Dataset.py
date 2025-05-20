import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
from skimage.transform import resize


class CustomAudioDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.labels = torch.tensor(dataframe['class'].values, dtype=torch.long)

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.dataframe.iloc[idx]['FilePath']
        spec = self.get_spectrogram(file_path)
        audio = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        return audio, label

    @staticmethod
    def get_spectrogram(file_path):
        sr = 22050
        duration = 5

        # Set the size of the spectrogram images
        img_height = 128
        img_width = 256

        # Load the audio file
        signal, sr = librosa.load(file_path, sr=sr, duration=duration)

        # Compute the spectrogram
        spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=2048, hop_length=512, n_mels=128)

        # Convert the spectrogram to dB scale
        spec_db = librosa.power_to_db(spec, ref=np.max)
        spec_db = (spec_db - spec_db.mean()) / spec_db.std()

        # Resize the spectrgoram to the desired shape
        spec_resized = librosa.util.fix_length(spec_db, size=duration * sr // 512 + 1)
        spec_resized = resize(spec_resized, (img_height, img_width), anti_aliasing=True)
        return spec_resized
