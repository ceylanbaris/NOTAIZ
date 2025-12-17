import librosa
import numpy as np
from scipy.spatial.distance import cosine

def extract_spectrogram(path):
    y, sr = librosa.load(path, sr=22050, mono=True)
    S = np.abs(librosa.stft(y))
    S_db = librosa.amplitude_to_db(S)
    return np.mean(S_db, axis=1)

def spectrogram_similarity(s1, s2):
    return 1 - cosine(s1, s2)
