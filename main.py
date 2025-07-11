import os
import tempfile
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from models import ModelManager
from tasks import TaskManager
from gpu_monitor import GPUMonitor
from log_manager import LogManager

# --- Инициализация компонентов ---
model_manager = ModelManager()
gpu_monitor = GPUMonitor()
log_manager = LogManager()
task_manager = TaskManager(model_manager, log_manager, gpu_monitor)

# Запускаем загрузку моделей при старте
model_manager.start_downloads()

app = FastAPI()

# Для отдачи файлов результатов и аудио
if not os.path.exists("uploads"):
    os.makedirs("uploads")
if not os.path.exists("results"):
    os.makedirs("results")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- UI ---
@app.get("/", response_class=HTMLResponse)
async def ui_root():
    # HTML + JS для динамического UI
    return HTMLResponse("""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Whisper GPU Transcriber</title>
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <style>
        body { background: #18191a; color: #f1f1f1; font-family: sans-serif; margin: 0; padding: 0; }
        .container { max-width: 1200px; margin: auto; padding: 24px; }
        h1 { color: #e4b34a; }
        .section { margin-bottom: 32px; }
        .gpu-table, .task-table, .log-table { width: 100%; border-collapse: collapse; margin-bottom: 12px; }
        th, td { padding: 8px 10px; border-bottom: 1px solid #333; }
        th { background: #222; }
        .btn { background: #e4b34a; color: #18191a; border: none; padding: 8px 14px; border-radius: 4px; cursor: pointer; }
        .btn:active { background: #c89f37; }
        .progress { background: #333; border-radius: 4px; overflow: hidden; height: 16px; }
        .progress-bar { background: #e4b34a; height: 100%; }
        @media (max-width: 600px) {
            .container { padding: 8px; }
            th, td { font-size: 13px; padding: 6px 4px; }
        }
    </style>
</head>
<body>
<div class="container">
    <h1>Whisper GPU Transcriber</h1>
    <div class="section">
        <h2>GPU</h2>
        <table class="gpu-table" id="gpu-table"></table>
    </div>
    <div class="section">
        <h2>Модели</h2>
        <table class="gpu-table" id="model-table"></table>
    </div>
    <div class="section">
        <h2>Загрузка файла для транскрибации</h2>
        <form id="upload-form">
            <input type="file" name="file" required>
            <select name="model_name" id="model-select"></select>
            <input type="text" name="initial_prompt" placeholder="Initial prompt (опционально)">
            <label>
                <input type="checkbox" name="upgrade_transcribation"> Улучшить через GPT
            </label>
            <button class="btn" type="submit">Загрузить и транскрибировать (приоритет)</button>
        </form>
        <div id="upload-status"></div>
    </div>
    <div class="section">
        <h2>Активные задачи</h2>
        <table class="task-table" id="active-tasks"></table>
        <h2>Очередь</h2>
        <table class="task-table" id="queued-tasks"></table>
    </div>
    <div class="section">
        <h2>Логи</h2>
        <table class="log-table" id="log-table"></table>
    </div>
</div>
<script>
async function fetchData() {
    // GPU
    let gpu = await fetch('/api/gpu').then(r=>r.json());
    let gpuTable = document.getElementById('gpu-table');
    gpuTable.innerHTML = '<tr><th>ID</th><th>Имя</th><th>Память (МБ)</th><th>Использовано (МБ)</th><th>Загрузка (%)</th></tr>' +
        gpu.map(g=>`<tr>
            <td>${g.id}</td>
            <td>${g.name}</td>
            <td>${g.memory_total_MB}</td>
            <td>${g.memory_used_MB}</td>
            <td>${g.utilization_percent ?? '-'}</td>
        </tr>`).join('');

    // Модели
    let models = await fetch('/api/models').then(r=>r.json());
    let modelTable = document.getElementById('model-table');
    modelTable.innerHTML = '<tr><th>Модель</th><th>Скачана</th><th>Прогресс</th></tr>' +
        models.map(m=>`<tr>
            <td>${m.name}</td>
            <td>${m.downloaded ? "✅" : "⏳"}</td>
            <td>
                <div class="progress"><div class="progress-bar" style="width:${m.progress}%;"></div></div>
                ${m.progress}%
            </td>
        </tr>`).join('');
    // Для селектора моделей
    let select = document.getElementById('model-select');
    select.innerHTML = models.map(m=>`<option value="${m.name}">${m.name}</option>`).join('');

    // Активные задачи
    let active = await fetch('/api/tasks/active').then(r=>r.json());
    let activeTable = document.getElementById('active-tasks');
    activeTable.innerHTML = '<tr><th>ID</th><th>Модель</th><th>GPU</th><th>Файл</th><th>Статус</th><th>Время</th><th>Результат</th></tr>' +
        active.map(t=>`<tr>
            <td>${t.id}</td>
            <td>${t.model_name}</td>
            <td>${t.gpu_id}</td>
            <td>${t.file_path}</td>
            <td>${t.status}</td>
            <td>${t.started_at ? (new Date(t.started_at*1000)).toLocaleTimeString() : ''}</td>
            <td>${t.result_path ? `<a href="/api/result/${t.result_path}" target="_blank">Скачать</a>` : ''}</td>
        </tr>`).join('');

    // Очередь задач
    let queued = await fetch('/api/tasks/queued').then(r=>r.json());
    let queuedTable = document.getElementById('queued-tasks');
    queuedTable.innerHTML = '<tr><th>ID</th><th>Модель</th><th>GPU</th><th>Файл</th><th>Статус</th><th>Время</th></tr>' +
        queued.map(t=>`<tr>
            <td>${t.id}</td>
            <td>${t.model_name}</td>
            <td>${t.gpu_id}</td>
            <td>${t.file_path}</td>
            <td>${t.status}</td>
            <td>${(new Date(t.created_at*1000)).toLocaleTimeString()}</td>
        </tr>`).join('');

    // Логи
    let logs = await fetch('/api/logs').then(r=>r.json());
    let logTable = document.getElementById('log-table');
    logTable.innerHTML = '<tr><th>Время</th><th>Сообщение</th></tr>' +
        logs.map(l=>`<tr><td>${l.timestamp}</td><td>${l.message}</td></tr>`).join('');
}
setInterval(fetchData, 1000);
fetchData();

document.getElementById('upload-form').onsubmit = async function(e) {
    e.preventDefault();
    let form = new FormData(this);
    let statusDiv = document.getElementById('upload-status');
    statusDiv.textContent = "Загрузка...";
    let resp = await fetch('/api/transcribe', {method: 'POST', body: form});
    let data = await resp.json();
    if (data.error) {
        statusDiv.textContent = "Ошибка: " + data.error;
    } else {
        statusDiv.textContent = "Задача добавлена!";
    }
}
</script>
</body>
</html>
    """)

# --- API: GPU ---
@app.get("/api/gpu")
async def api_gpu():
    return gpu_monitor.get_all_info()

# --- API: Модели ---
@app.get("/api/models")
async def api_models():
    return model_manager.get_statuses()

# --- API: Логи ---
@app.get("/api/logs")
async def api_logs():
    return log_manager.get_logs(100)

# --- API: Активные задачи ---
@app.get("/api/tasks/active")
async def api_active_tasks():
    return [
        {
            "id": t.id,
            "model_name": t.model_name,
            "gpu_id": t.gpu_id,
            "file_path": t.file_path,
            "status": t.status,
            "started_at": t.started_at,
            "result_path": t.result_path
        }
        for t in task_manager.get_active_tasks()
    ]

# --- API: Очередь ---
@app.get("/api/tasks/queued")
async def api_queued_tasks():
    return [
        {
            "id": t.id,
            "model_name": t.model_name,
            "gpu_id": t.gpu_id,
            "file_path": t.file_path,
            "status": t.status,
            "created_at": t.created_at
        }
        for t in task_manager.get_queued_tasks()
    ]

# --- API: Получить файл результата ---
@app.get("/api/result/{filename}")
async def api_result(filename: str):
    path = os.path.join("results", filename)
    if os.path.exists(path):
        return FileResponse(path, filename=filename)
    return JSONResponse({"error": "Файл не найден"}, status_code=404)

# --- API: Добавить задачу на транскрибацию ---
@app.post("/api/transcribe")
async def api_transcribe(
    file: UploadFile = File(...),
    model_name: str = Form("base"),
    initial_prompt: str = Form(None),
    upgrade_transcribation: bool = Form(False)
):
    # Проверяем, скачана ли модель
    if not model_manager.is_downloaded(model_name):
        return JSONResponse({"error": f"Модель {model_name} ещё не скачана. Ожидайте завершения загрузки."}, status_code=400)
    # Сохраняем файл корректно!
    file_ext = os.path.splitext(file.filename)[1]
    fd, tmp_path = tempfile.mkstemp()
    os.close(fd)
    save_path = os.path.join("uploads", f"{os.path.splitext(file.filename)[0]}_{os.path.basename(tmp_path)}{file_ext}")
    with open(save_path, "wb") as f:
        f.write(await file.read())
    # Добавляем задачу с максимальным приоритетом
    task = task_manager.add_task(
        file_path=save_path,
        model_name=model_name,
        initial_prompt=initial_prompt,
        upgrade_transcribation=upgrade_transcribation,
        priority=True
    )
    return {"task_id": task.id}

# --- Для отдачи файлов (uploads/results) ---
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/results", StaticFiles(directory="results"), name="results")
