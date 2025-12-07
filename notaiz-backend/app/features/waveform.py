import librosa
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def generate_waveform(path):
    y, sr = librosa.load(path, sr=None)

    plt.figure(figsize=(10, 3))
    plt.plot(y)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")
