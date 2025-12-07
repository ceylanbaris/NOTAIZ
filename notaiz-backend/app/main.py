from fastapi import FastAPI, UploadFile, File
from utils.save_file import save_upload
from features.chroma import extract_chroma, chroma_similarity
from features.waveform import generate_waveform

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Notaiz backend çalışıyor!"}

@app.post("/analyze")
async def analyze(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    # Dosya kaydet
    path1 = save_upload(file1)
    path2 = save_upload(file2)

    # Chroma
    chroma1 = extract_chroma(path1)
    chroma2 = extract_chroma(path2)

    score = chroma_similarity(chroma1, chroma2)

    # Waveform
    wave1 = generate_waveform(path1)
    wave2 = generate_waveform(path2)

    return {
        "score": score,
        "waveform1": wave1,
        "waveform2": wave2
    }
