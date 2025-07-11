import os
import tempfile
import threading
import time
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
import torch
import whisper
import g4f
from templates import render_status, render_logs, render_tasks, render_stats, render_history
from stats_manager import get_gpu_stats, get_cpu_ram_stats, StatsHistory

router = APIRouter()

models_manager = None
logs_manager = None
tasks_manager = None
stats_history = StatsHistory()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
loaded_models = {}

def get_model(model_name: str, download_dir: str = None, device="cuda", device_index=None):
    key = (model_name, download_dir, device_index)
    if key not in loaded_models:
        loaded_models[key] = whisper.load_model(
            model_name, device=device, download_root=download_dir
        )
    return loaded_models[key]

def format_timestamp(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"

def format_segments(segments):
    lines = []
    for seg in segments:
        start = format_timestamp(seg['start'])
        text = seg['text'].strip()
        lines.append(f"[{start}] {text}")
    return "\n".join(lines)

def gpt_chat(prompt: str) -> str:
    response = g4f.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.strip()

def background_queue_worker():
    while True:
        queue = tasks_manager.get_queue()
        for task in queue:
            gpu_id = tasks_manager.assign_gpu_to_task(task.id)
            if gpu_id is not None:
                threading.Thread(
                    target=process_task,
                    args=(task, gpu_id),
                    daemon=True
                ).start()
        time.sleep(1)

def process_task(task, gpu_id):
    tmp_path = task.filename
    try:
        model = get_model(task.model, device="cuda", device_index=gpu_id)
        transcribe_kwargs = {"fp16": True, "beam_size": 5}
        result = model.transcribe(tmp_path, **transcribe_kwargs)
        formatted_text = format_segments(result['segments'])
        tasks_manager.update_task_done(task.id, formatted_text)
    except Exception as e:
        tasks_manager.update_task_error(task.id, str(e))
    finally:
        tasks_manager.release_gpu(gpu_id)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        torch.cuda.empty_cache()

@router.get("/", response_class=HTMLResponse)
async def root():
    count, gpus = len(get_gpu_stats()), get_gpu_stats()
    models_status = models_manager.get_status()
    logs = logs_manager.get_logs()[-50:]
    queue = tasks_manager.get_queue()
    processing = tasks_manager.get_processing()
    history = tasks_manager.get_history()
    stats = get_cpu_ram_stats()
    gpu_stats = get_gpu_stats()
    stats_history.add_stats(gpu_stats, stats, len(history))
    html = """
    <html>
    <head>
    <title>Whisper API — Мониторинг и управление</title>
    <meta charset="utf-8">
    <link rel="stylesheet" href="/static/style.css">
    <link href="https://fonts.googleapis.com/css?family=Inter:400,600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
    setInterval(function(){
        fetch('/status').then(r=>r.text()).then(html=>document.getElementById('status').innerHTML=html);
        fetch('/logs').then(r=>r.text()).then(html=>document.getElementById('logs').innerHTML=html);
        fetch('/tasks').then(r=>r.text()).then(html=>document.getElementById('tasks').innerHTML=html);
        fetch('/stats').then(r=>r.text()).then(html=>document.getElementById('stats').innerHTML=html);
        fetch('/history').then(r=>r.text()).then(html=>document.getElementById('history').innerHTML=html);
    }, 1000);
    </script>
    </head>
    <body>
    <div class="container">
    <h2>Whisper API — Мониторинг и управление</h2>
    <div id="stats">""" + render_stats(gpu_stats, stats) + """</div>
    <div id="status">""" + render_status(count, gpus, models_status) + """</div>
    <h3>Выполняются</h3>
    <div id='tasks'>""" + render_tasks(processing, queue) + """</div>
    <h3>История</h3>
    <div id='history'>""" + render_history(history, stats_history) + """</div>
    <h3>Логи работы API</h3>
    <div id='logs'>""" + render_logs(logs) + """</div>
    </div></body></html>
    """
    return html

@router.get("/status", response_class=HTMLResponse)
async def status():
    count, gpus = len(get_gpu_stats()), get_gpu_stats()
    models_status = models_manager.get_status()
    return render_status(count, gpus, models_status)

@router.get("/logs", response_class=HTMLResponse)
async def logs():
    logs = logs_manager.get_logs()[-50:]
    return render_logs(logs)

@router.get("/tasks", response_class=HTMLResponse)
async def tasks():
    queue = tasks_manager.get_queue()
    processing = tasks_manager.get_processing()
    return render_tasks(processing, queue)

@router.get("/stats", response_class=HTMLResponse)
async def stats():
    stats = get_cpu_ram_stats()
    gpu_stats = get_gpu_stats()
    return render_stats(gpu_stats, stats)

@router.get("/history", response_class=HTMLResponse)
async def history():
    history = tasks_manager.get_history()
    return render_history(history, stats_history)

@router.get("/history_json")
async def history_json():
    gpu = []
    req = []
    timestamps = []
    for i, t in enumerate(stats_history.timestamps):
        if i < len(stats_history.gpu_stats):
            gpus = stats_history.gpu_stats[i]
            if isinstance(gpus, list) and gpus:
                gpu.append(max([g['gpu_util'] for g in gpus]))
            else:
                gpu.append(0)
        else:
            gpu.append(0)
        if i < len(stats_history.request_counts):
            req.append(stats_history.request_counts[i])
        else:
            req.append(0)
        if i < len(stats_history.timestamps):
            timestamps.append(stats_history.timestamps[i].strftime("%H:%M"))
        else:
            timestamps.append("")
    return JSONResponse({"gpu": gpu, "req": req, "timestamps": timestamps})

@router.post("/transcribe/")
async def transcribe(
    file: UploadFile = File(...),
    model_name: str = Form("base"),
    model_dir: str = Form(None),
    initial_prompt: str = Form(None),
    upgrade_transcribation: bool = Form(False)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        model = get_model(model_name, model_dir)
        transcribe_kwargs = {
            "fp16": (DEVICE == "cuda"),
            "beam_size": 5
        }
        if initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt

        result = model.transcribe(tmp_path, **transcribe_kwargs)
        formatted_text = format_segments(result['segments'])

        if upgrade_transcribation:
            gpt_prompt = (
                "Вот расшифровка диалога между двумя спикерами: сотрудником и клиентом. "
                "Раздели текст по репликам спикеров, подпиши кто говорит (Сотрудник или Клиент), "
                "исправь явные ошибки и сделай текст более читабельным. "
                "Сохрани тайминги в формате [mm:ss] перед каждой репликой. "
                "Учитывай, что в таймингах, которые уже есть - ошибки. "
                "Иногда реплика незакончена, а тайминг подписан. Не обрывай реплики таймингами. "
                "Если спикер не закончил говорить, не пиши тайминг.\n\n"
                f"{formatted_text}"
            )
            try:
                improved_text = gpt_chat(gpt_prompt)
                return JSONResponse({"text": improved_text})
            except Exception as e:
                return JSONResponse({
                    "text": formatted_text,
                    "warning": f"Ошибка улучшения через GPT: {str(e)}"
                })
        else:
            return JSONResponse({"text": formatted_text})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        os.remove(tmp_path)
        torch.cuda.empty_cache()
