import os
import tempfile
import subprocess
import torch
import json
import threading
from datetime import datetime, timedelta
from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
import g4f

router = APIRouter()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
model_status = {name: {"loaded": False, "progress": 0, "error": None} for name in WHISPER_MODELS}
model_cache = {}

LOG_PATH = "transcribe_log.json"

def download_model(name):
    try:
        model_status[name]["progress"] = 10
        import whisper
        model_cache[name] = whisper.load_model(name, device=DEVICE)
        model_status[name]["loaded"] = True
        model_status[name]["progress"] = 100
    except Exception as e:
        model_status[name]["error"] = str(e)
        model_status[name]["progress"] = 0

def preload_models():
    for name in WHISPER_MODELS:
        threading.Thread(target=download_model, args=(name,), daemon=True).start()

preload_models()

def get_model(model_name):
    return model_cache.get(model_name)

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

def log_transcribe(entry):
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            logs = json.load(f)
    else:
        logs = []
    logs.append(entry)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(logs[-1000:], f, ensure_ascii=False, indent=2)

def get_logs():
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def get_stats_24h():
    logs = get_logs()
    now = datetime.utcnow()
    hours = [0]*24
    for entry in logs:
        dt = datetime.fromisoformat(entry["datetime"])
        diff = now - dt
        if 0 <= diff.total_seconds() < 86400:
            h = int((now - dt).total_seconds() // 3600)
            hours[23-h] += 1
    return hours

@router.get("/", response_class=HTMLResponse)
async def ui_root():
    return HTMLResponse(open("static/index.html", encoding="utf-8").read())

@router.get("/api/gpu")
async def api_gpu():
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        gpus = []
        for i in range(count):
            gpus.append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total_MB": torch.cuda.get_device_properties(i).total_memory // (1024 * 1024),
                "memory_used_MB": torch.cuda.memory_allocated(i) // (1024 * 1024)
            })
        return gpus
    else:
        return []

@router.get("/api/models")
async def api_models():
    return [
        {
            "name": name,
            "loaded": model_status[name]["loaded"],
            "progress": model_status[name]["progress"],
            "error": model_status[name]["error"]
        }
        for name in WHISPER_MODELS
    ]

@router.get("/api/logs")
async def api_logs():
    return get_logs()[::-1]

@router.get("/api/stats24")
async def api_stats24():
    return get_stats_24h()

@router.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model_name: str = Form("base"),
    initial_prompt: str = Form(None),
    upgrade_transcribation: bool = Form(False),
    up_speed: float = Form(1.0)
):
    # Проверка: модель загружена?
    if not model_status.get(model_name, {}).get("loaded", False):
        return JSONResponse(
            {"error": f"Модель '{model_name}' ещё не установлена. Подождите завершения загрузки."},
            status_code=400
        )

    fd, input_path = tempfile.mkstemp()
    os.close(fd)
    with open(input_path, "wb") as f:
        f.write(await file.read())

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
        if model is None:
            return JSONResponse(
                {"error": f"Модель '{model_name}' ещё не установлена. Подождите завершения загрузки."},
                status_code=400
            )
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

        # Улучшение через GPT, если требуется
        if upgrade_transcribation:
            try:
                gpt_prompt = (
                    "Вот расшифровка диалога между двумя спикерами: сотрудником и клиентом. "
                    "Раздели текст по репликам спикеров, подпиши кто говорит (Сотрудник или Клиент), "
                    "исправь явные ошибки и сделай текст более читабельным. "
                    "Сохрани тайминги в формате [mm:ss] перед каждой репликой. ЕСЛИ разговора нет, а был автоответчик или чтото другое, в первой строке напиши *false*, в другом случае *true*\n\n"
                    f"{formatted_text}"
                )
                improved_text = g4f.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": gpt_prompt}],
                ).strip()
                result_text = improved_text
            except Exception as e:
                result_text = formatted_text
        else:
            result_text = formatted_text

        entry = {
            "datetime": datetime.utcnow().isoformat(),
            "model": model_name,
            "up_speed": up_speed,
            "filename": file.filename,
            "result_len": len(result_text),
            "initial_prompt": initial_prompt,
            "upgrade_transcribation": upgrade_transcribation
        }
        log_transcribe(entry)
        return {"text": result_text}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        os.remove(input_path)
        os.remove(mp3_path)
        torch.cuda.empty_cache()
