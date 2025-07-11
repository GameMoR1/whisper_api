import os
import tempfile
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

@router.get("/", response_class=HTMLResponse)
async def root():
    count, gpus = len(get_gpu_stats()), get_gpu_stats()
    models_status = models_manager.get_status()
    logs = logs_manager.get_logs()[-50:] if logs_manager else []
    queue = tasks_manager.get_queue() if tasks_manager else []
    processing = tasks_manager.get_processing() if tasks_manager else []
    history = tasks_manager.get_history() if tasks_manager else []
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
    </head>
    <body>
    <div class="container">
    <h2>Whisper API — Мониторинг и управление</h2>
    <div id="stats">""" + render_stats(gpu_stats, stats) + """</div>
    <div id="status">""" + render_status(count, gpus, models_status) + """</div>
    <h3>Выполняются</h3>
    <div id='tasks'>""" + render_tasks(processing, queue) + """</div>
    <h3>История</h3>
    <div class='charts-row'>
        <div class='chart-block'>
            <h4>История загрузки GPU</h4>
            <canvas id='gpu_history_chart' height='120'></canvas>
        </div>
        <div class='chart-block'>
            <h4>История количества запросов</h4>
            <canvas id='req_history_chart' height='120'></canvas>
        </div>
    </div>
    <h3>Логи работы API</h3>
    <div id='logs'>""" + render_logs(logs) + """</div>
    </div>
    <script>
    let gpuChart, reqChart;

    function updateCharts() {
        fetch('/history_json').then(r => r.json()).then(data => {
            const labels = data.timestamps;
            const gpu = data.gpu;
            const req = data.req;

            if (!gpuChart) {
                gpuChart = new Chart(document.getElementById('gpu_history_chart').getContext('2d'), {
                    type: 'line',
                    data: {labels: labels, datasets: [{label:'GPU (%)', data: gpu, borderColor:'#4fd1c5', backgroundColor:'rgba(79,209,197,0.1)'}]},
                    options: {scales:{y:{min:0,max:100}}, animation: false}
                });
            } else {
                gpuChart.data.labels = labels;
                gpuChart.data.datasets[0].data = gpu;
                gpuChart.update('none');
            }

            if (!reqChart) {
                reqChart = new Chart(document.getElementById('req_history_chart').getContext('2d'), {
                    type: 'line',
                    data: {labels: labels, datasets: [{label:'Запросы', data: req, borderColor:'#f6ad55', backgroundColor:'rgba(246,173,85,0.1)'}]},
                    options: {scales:{y:{beginAtZero:true}}, animation: false}
                });
            } else {
                reqChart.data.labels = labels;
                reqChart.data.datasets[0].data = req;
                reqChart.update('none');
            }
        });
    }

    updateCharts();
    setInterval(updateCharts, 1000);

    setInterval(function(){
        fetch('/status').then(r=>r.text()).then(html=>document.getElementById('status').innerHTML=html);
        fetch('/logs').then(r=>r.text()).then(html=>document.getElementById('logs').innerHTML=html);
        fetch('/tasks').then(r=>r.text()).then(html=>document.getElementById('tasks').innerHTML=html);
        fetch('/stats').then(r=>r.text()).then(html=>document.getElementById('stats').innerHTML=html);
    }, 1000);
    </script>
    </body>
    </html>
    """
    return html

@router.get("/status", response_class=HTMLResponse)
async def status():
    count, gpus = len(get_gpu_stats()), get_gpu_stats()
    models_status = models_manager.get_status()
    return render_status(count, gpus, models_status)

@router.get("/logs", response_class=HTMLResponse)
async def logs():
    logs = logs_manager.get_logs()[-50:] if logs_manager else []
    return render_logs(logs)

@router.get("/tasks", response_class=HTMLResponse)
async def tasks():
    queue = tasks_manager.get_queue() if tasks_manager else []
    processing = tasks_manager.get_processing() if tasks_manager else []
    return render_tasks(processing, queue)

@router.get("/stats", response_class=HTMLResponse)
async def stats():
    stats = get_cpu_ram_stats()
    gpu_stats = get_gpu_stats()
    return render_stats(gpu_stats, stats)

@router.get("/history", response_class=HTMLResponse)
async def history():
    history = tasks_manager.get_history() if tasks_manager else []
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
    initial_prompt: str = Form(None),
    upgrade_transcribation: bool = Form(False),
    up_speed: str = Form(None)
):
    # Проверяем статус модели
    if not models_manager.is_model_loaded(model_name):
        if models_manager.is_model_loading(model_name):
            return JSONResponse({"error": f"Модель '{model_name}' ещё загружается. Попробуйте позже."}, status_code=503)
        else:
            return JSONResponse({"error": f"Модель '{model_name}' не готова или произошла ошибка."}, status_code=503)

    # Получаем модель на CPU (всегда), для транскрибации копируем на GPU если нужно
    base_model = models_manager.get_model(model_name, device="cpu")
    if base_model is None:
        return JSONResponse({"error": f"Модель '{model_name}' не найдена."}, status_code=503)

    # Копируем модель на GPU только для транскрибации
    model = base_model
    if DEVICE == "cuda":
        model = base_model.to("cuda")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
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
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
