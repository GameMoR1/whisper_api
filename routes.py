import os
import tempfile
import subprocess
import whisper
import torch

from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse

router = APIRouter()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Можно реализовать менеджер моделей, но для простоты — одна модель
model_cache = {}

def get_model(model_name):
    if model_name not in model_cache:
        model_cache[model_name] = whisper.load_model(model_name, device=DEVICE)
    return model_cache[model_name]

def build_atempo_filters(speed):
    filters = []
    while speed > 2.0:
        filters.append("atempo=2.0")
        speed /= 2.0
    while speed < 0.5:
        filters.append("atempo=0.5")
        speed /= 0.5
    filters.append(f"atempo={speed:.5f}")
    return ",".join(filters)

@router.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model_name: str = Form("base"),
    initial_prompt: str = Form(None),
    upgrade_transcribation: bool = Form(False),
    up_speed: float = Form(1.0)
):
    # Сохраняем файл во временный файл
    fd, input_path = tempfile.mkstemp()
    os.close(fd)
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Конвертация и ускорение через ffmpeg
    fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    atempo_filter = build_atempo_filters(up_speed)
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-acodec", "libmp3lame", "-ar", "44100", "-ac", "2",
        "-filter:a", atempo_filter,
        mp3_path
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        os.remove(input_path)
        os.remove(mp3_path)
        return JSONResponse({"error": f"FFmpeg error: {e.stderr.decode()}"}, status_code=500)

    try:
        model = get_model(model_name)
        transcribe_kwargs = {
            "fp16": (DEVICE == "cuda"),
            "beam_size": 5
        }
        if initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt
        result = model.transcribe(mp3_path, **transcribe_kwargs)

        def format_timestamp(seconds):
            m, s = divmod(int(seconds), 60)
            return f"{m:02d}:{s:02d}"

        formatted_text = "\n".join(
            f"[{format_timestamp(seg['start'])}] {seg['text'].strip()}"
            for seg in result['segments']
        )

        # Если нужно улучшение через GPT — вставьте сюда свой код

        return {"text": formatted_text}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        os.remove(input_path)
        os.remove(mp3_path)
        torch.cuda.empty_cache()
