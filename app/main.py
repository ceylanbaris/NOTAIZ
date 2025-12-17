from fastapi import FastAPI, UploadFile, File
from starlette.middleware.base import BaseHTTPMiddleware
import numpy as np
import librosa
from sklearn.decomposition import PCA
from fastdtw import fastdtw

# -----------------------------------------------------
# Büyük dosya upload sorunu çözmek için middleware
# -----------------------------------------------------
class LargeUploadMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Tüm request body'yi oku (large MP3 için şart)
        await request.body()
        response = await call_next(request)
        return response


# -----------------------------------------------------
# FastAPI App
# -----------------------------------------------------
app = FastAPI()
app.add_middleware(LargeUploadMiddleware)


# -----------------------------------------------------
# Helper Functions
# -----------------------------------------------------

def load_audio(file: UploadFile, sr=22050):
    audio, _ = librosa.load(file.file, sr=sr, mono=True)
    return audio


def pad_or_trim(a, b):
    min_len = min(len(a), len(b))
    return a[:min_len], b[:min_len]


def compute_mfcc(audio, sr=22050, n=13):
    return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n)


def compute_chroma(audio, sr=22050):
    return librosa.feature.chroma_stft(y=audio, sr=sr)


def compute_spectral(audio, sr=22050):
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    return centroid, bandwidth, rolloff


def apply_pca(feature):
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(feature.T)
    return reduced


def cosine_similarity(a, b):
    a = a.flatten()
    b = b.flatten()
    min_len = min(len(a), len(b))
    a = a[:min_len]
    b = b[:min_len]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def dtw_distance(a, b):
    a = a.flatten()
    b = b.flatten()
    min_len = min(len(a), len(b))
    a = a[:min_len]
    b = b[:min_len]
    dist, _ = fastdtw(a, b)
    return float(dist)


def normalize(value, max_value):
    v = 1 - (value / max_value)
    return max(0, min(v, 1))


# -----------------------------------------------------
# Main Endpoint
# -----------------------------------------------------

@app.post("/analyze")
async def analyze(file1: UploadFile = File(...), file2: UploadFile = File(...)):

    try:
        # Load audio
        y1 = load_audio(file1)
        y2 = load_audio(file2)

        # Equal length
        y1, y2 = pad_or_trim(y1, y2)

        # Features
        mfcc1 = compute_mfcc(y1)
        mfcc2 = compute_mfcc(y2)

        chroma1 = compute_chroma(y1)
        chroma2 = compute_chroma(y2)

        centroid1, bw1, roll1 = compute_spectral(y1)
        centroid2, bw2, roll2 = compute_spectral(y2)

        # PCA Reduce
        mfcc1_r = apply_pca(mfcc1)
        mfcc2_r = apply_pca(mfcc2)

        chroma1_r = apply_pca(chroma1)
        chroma2_r = apply_pca(chroma2)

        spectral1 = np.concatenate([centroid1, bw1, roll1], axis=0)
        spectral2 = np.concatenate([centroid2, bw2, roll2], axis=0)

        spectral1_r = apply_pca(spectral1)
        spectral2_r = apply_pca(spectral2)

        # Similarities
        cos_mfcc = cosine_similarity(mfcc1_r, mfcc2_r)
        cos_chroma = cosine_similarity(chroma1_r, chroma2_r)
        cos_spec = cosine_similarity(spectral1_r, spectral2_r)

        dtw_mfcc = dtw_distance(mfcc1_r, mfcc2_r)
        dtw_chroma = dtw_distance(chroma1_r, chroma2_r)

        # Normalize
        dtw_mfcc_n = normalize(dtw_mfcc, 50000)
        dtw_chroma_n = normalize(dtw_chroma, 50000)

        # Final score
        final_score = (
            (cos_mfcc * 0.35)
            + (cos_chroma * 0.25)
            + (cos_spec * 0.15)
            + (dtw_mfcc_n * 0.15)
            + (dtw_chroma_n * 0.10)
        ) * 100

        risk = (
            "Düşük Risk" if final_score < 40
            else "Orta Risk" if final_score < 75
            else "Yüksek Risk"
        )

        return {
            "similarity_percent": round(final_score, 2),
            "risk": risk,
            "cosine_mfcc": cos_mfcc,
            "cosine_chroma": cos_chroma,
            "cosine_spectral": cos_spec,
            "dtw_mfcc_norm": dtw_mfcc_n,
            "dtw_chroma_norm": dtw_chroma_n,
            "message": "Analiz tamamlandı."
        }

    except Exception as e:
        return {"detail": f"Analiz hatası: {str(e)}"}
