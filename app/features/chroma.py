import librosa
import numpy as np

def extract_chroma(path):
    # Load audio
    y, sr = librosa.load(path, sr=22050, mono=True)

    # Extract chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # Flatten
    chroma_flat = chroma.flatten()

    return chroma_flat


def pad_or_trim(vec, target_len):
    """Resize vector safely by padding or trimming."""
    if len(vec) > target_len:
        return vec[:target_len]
    else:
        return np.pad(vec, (0, target_len - len(vec)), mode='constant')


def chroma_similarity(v1, v2):
    # Pick the same length for both
    target_len = min(len(v1), len(v2))

    v1 = pad_or_trim(v1, target_len)
    v2 = pad_or_trim(v2, target_len)

    # Normalize
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Cosine similarity
    return float(np.dot(v1, v2))
