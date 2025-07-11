import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
import torch
import whisper
import g4f
from templates import render_status, render_logs, render_tasks

router = APIRouter()

models_manager = None
logs_manager = None
tasks_manager = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
loaded_models = {}

def get_model(model_name: str, download_dir: str = None):
    key = (model_name, download_dir)
    if key not in loaded_models:
        loaded_models[key] = whisper.load_model(model_name, device=DEVICE, download_root=download_dir)
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

def get_gpu_info():
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        gpus = []
        for i in range(count):
            gpus.append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total_MB": torch.cuda.get_device_properties(i).total_memory // (1024 * 1024)
            })
        return count, gpus
    else:
        return 0, []

@router.get("/", response_class=HTMLResponse)
async def root():
    count, gpus = get_gpu_info()
    models_status = models_manager.get_status()
    logs = logs_manager.get_logs()[-50:]
    tasks = tasks_manager.get_tasks()

    html = """
    <html>
    <head>
    <title>Whisper API — Мониторинг и управление</title>
    <meta charset="utf-8">
    <link rel="stylesheet" href="/static/style.css">
    <link href="https://fonts.googleapis.com/css?family=Inter:400,600&display=swap" rel="stylesheet">
    <script>
    setInterval(function(){
        fetch('/status').then(r=>r.text()).then(html=>document.getElementById('status').innerHTML=html);
        fetch('/logs').then(r=>r.text()).then(html=>document.getElementById('logs').innerHTML=html);
        fetch('/tasks').then(r=>r.text()).then(html=>document.getElementById('tasks').innerHTML=html);
    }, 1000);
    </script>
    </head>
    <body>
    <div class="container">
    <h2>Whisper API — Мониторинг и управление</h2>
    <div id="status">
    """
    html += render_status(count, gpus, models_status)
    html += "</div>"
    html += "<h3>Активные процессы</h3><div id='tasks'>"
    html += render_tasks(tasks)
    html += "</div>"
    html += "<h3>Логи работы API</h3><div id='logs'>"
    html += render_logs(logs)
    html += "</div></div></body></html>"
    return html

@router.get("/status", response_class=HTMLResponse)
async def status():
    count, gpus = get_gpu_info()
    models_status = models_manager.get_status()
    return render_status(count, gpus, models_status)

@router.get("/logs", response_class=HTMLResponse)
async def logs():
    logs = logs_manager.get_logs()[-50:]
    return render_logs(logs)

@router.get("/tasks", response_class=HTMLResponse)
async def tasks():
    tasks = tasks_manager.get_tasks()
    return render_tasks(tasks)

@router.post("/transcribe/")
async def transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_name: str = Form("base"),
    model_dir: str = Form(None),
    initial_prompt: str = Form(None),
    upgrade_transcribation: bool = Form(False)
):
    if not models_manager.is_model_loaded(model_name):
        return JSONResponse(
            {"error": f"Модель '{model_name}' ещё не скачана. Попробуйте позже."},
            status_code=503
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    task_id = tasks_manager.add_task(file.filename, model_name)
    logs_manager.log(f"Задача {task_id}: получен файл {file.filename} для транскрибации ({model_name})", "INFO")

    def process_task():
        try:
            tasks_manager.update_task(task_id, "processing")
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
                    logs_manager.log(f"Задача {task_id}: улучшенная транскрибация завершена", "INFO")
                except Exception as e:
                    logs_manager.log(f"Задача {task_id}: ошибка улучшения через GPT: {str(e)}", "ERROR")
            else:
                logs_manager.log(f"Задача {task_id}: транскрибация завершена", "INFO")
            tasks_manager.update_task(task_id, "done")
        except Exception as e:
            logs_manager.log(f"Задача {task_id}: ошибка транскрибации: {str(e)}", "ERROR")
            tasks_manager.update_task(task_id, "error", str(e))
        finally:
            os.remove(tmp_path)
            torch.cuda.empty_cache()

    background_tasks.add_task(process_task)
    return JSONResponse({"status": "processing", "task_id": task_id, "detail": "Файл принят, задача поставлена в очередь"})
