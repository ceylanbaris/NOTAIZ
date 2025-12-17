import librosa
import numpy as np
from scipy.spatial.distance import cosine

def extract_mfcc(path):
    y, sr = librosa.load(path, sr=22050, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(mfcc, axis=1)

def mfcc_similarity(m1, m2):
    return 1 - cosine(m1, m2)
