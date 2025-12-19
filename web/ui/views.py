import requests
from django.shortcuts import render

FASTAPI_URL = "http://127.0.0.1:8000/transcribe"

def index(request):
    context = {}

    if request.method == "POST":
        audio = request.FILES.get("audio")
        min_length = request.POST.get("min_length", 5)
        max_length = request.POST.get("max_length", 15)

        response = requests.post(
            FASTAPI_URL,
            files={"file": audio},
            params={
                "min_length": min_length,
                "max_length": max_length
            }
        )

        if response.status_code == 200:
            data = response.json()
            context["transcription"] = data["transcription"]
            context["summary"] = data["summary"]
        else:
            context["error"] = "Ошибка при обработке аудио"

    return render(request, "index.html", context)
