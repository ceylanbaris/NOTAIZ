import librosa
import numpy as np

def generate_waveform(audio_path, length=200):
    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)

        if len(y) == 0:
            print(f"[WARN] Audio empty or read-error: {audio_path}")
            return [0] * length

        # normalize
        y = y / np.max(np.abs(y))

        # fix length
        if len(y) < length:
            y = np.pad(y, (0, length - len(y)))
        else:
            y = y[:length]

        return y.tolist()

    except Exception as e:
        print(f"[ERROR] Waveform generation failed for: {audio_path}")
        print(e)
        return [0] * length
