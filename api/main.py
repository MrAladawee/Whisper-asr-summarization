from fastapi import FastAPI, UploadFile, File, Query
import tempfile
import torch
import warnings
from transformers import pipeline

warnings.filterwarnings("ignore")

app = FastAPI(title="Whisper ASR + Summarization API")

device = 0 if torch.cuda.is_available() else -1

asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device=device,
    generate_kwargs={"language": "en"}
)

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=device
)

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    max_length: int = Query(15, ge=5, le=100),
    min_length: int = Query(5, ge=1, le=50),
):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(await file.read())
        audio_path = f.name

    transcription = asr(audio_path)["text"]

    summary = summarizer(
        transcription,
        max_length=max_length,
        min_length=min_length,
        do_sample=False
    )[0]["summary_text"]

    return {
        "transcription": transcription,
        "summary": summary,
        "params": {
            "min_length": min_length,
            "max_length": max_length
        }
    }
