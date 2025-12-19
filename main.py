import os
import torch
import tempfile
from transformers import pipeline
from datasets import load_dataset, Audio
import warnings

warnings.filterwarnings("ignore")

def main():
    print("Версия PyTorch:", torch.__version__)
    print("CUDA доступна:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Имя GPU:", torch.cuda.get_device_name(0))
    else:
        print("Будет использоваться CPU (работать будет медленнее).")

    os.environ["HF_DATASETS_AUDIO_ALLOW_TORCHCODEC"] = "0"

    librispeech = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy",
        split="validation[:1]"
    )

    librispeech = librispeech.cast_column("audio", Audio(decode=False))
    sample = librispeech[0]

    audio_bytes = sample["audio"]["bytes"]
    print("Оригинальный текст:")
    print(sample["text"])

    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        generate_kwargs={"language": "en"}
    )

    with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name

    result = asr(temp_path)
    print("ТРАНСКРИПЦИЯ ОТ WHISPER:")
    print(result["text"])

    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

    summary = summarizer(
        result["text"],
        max_length=15,
        min_length=5,
        do_sample=False
    )

    print("СУММАРИЗАЦИЯ:")
    print(summary[0]["summary_text"])

if __name__ == "__main__":
    main()
