def render_status(count, gpus, models_status):
    html = f"<p>Доступно GPU: <span class='gpu-count'>{count}</span></p>"
    if count > 0:
        html += "<ul class='gpu-list'>"
        for gpu in gpus:
            html += f"<li>ID: {gpu['id']} | <b>{gpu['name']}</b> | {gpu['mem_used']}/{gpu['mem_total']} MB, <span class='ok'>{gpu['gpu_util']}%</span></li>"
        html += "</ul>"
    else:
        html += "<p class='no-gpu'>GPU не обнаружены</p>"
    html += "<h3>Whisper модели</h3><table class='models-table'><tr><th>Модель</th><th>Статус</th><th>Ошибка</th></tr>"
    for name, info in models_status.items():
        status = "<span class='ok'>✅</span>" if info["loaded"] else "<span class='wait'>⏳</span>"
        error = f"<span class='err'>{info['error']}</span>" if info["error"] else ""
        html += f"<tr><td>{name}</td><td>{status}</td><td>{error}</td></tr>"
    html += "</table>"
    return html

def render_logs(logs):
    html = "<table class='logs-table'><tr><th>Время</th><th>Уровень</th><th>Сообщение</th></tr>"
    for entry in logs:
        level_class = "log-info" if entry['level'] == "INFO" else "log-error"
        html += f"<tr class='{level_class}'><td>{entry['timestamp']}</td><td>{entry['level']}</td><td>{entry['message']}</td></tr>"
    html += "</table>"
    return html

def render_tasks(processing, queue):
    html = "<div class='tasks-block'>"
    html += "<table class='tasks-table'><tr><th>ID</th><th>Файл</th><th>Модель</th><th>GPU</th><th>Старт</th><th>Статус</th></tr>"
    for task in processing:
        html += f"<tr><td>{task.id[:8]}</td><td>{os.path.basename(task.filename)}</td><td>{task.model}</td><td>{task.gpu_id}</td><td>{task.started_at.strftime('%H:%M:%S') if task.started_at else ''}</td><td><span class='proc'>Выполняется</span></td></tr>"
    html += "</table>"
    html += "<table class='tasks-table'><tr><th>ID</th><th>Файл</th><th>Модель</th><th>Время постановки</th><th>Статус</th></tr>"
    for task in queue:
        html += f"<tr><td>{task.id[:8]}</td><td>{os.path.basename(task.filename)}</td><td>{task.model}</td><td>{task.created_at.strftime('%H:%M:%S')}</td><td><span class='wait'>В очереди</span></td></tr>"
    html += "</table>"
    html += "</div>"
    return html

def render_stats(gpu_stats, cpu_stats):
    html = "<div class='stats-row'>"
    for gpu in gpu_stats:
        html += f"""
        <div class='stat-block'>
            <h4>GPU {gpu['id']} <span style='font-size:0.8em;color:#aaa'>{gpu['name']}</span></h4>
            <div>Загрузка: <span class='ok'>{gpu['gpu_util']}%</span></div>
            <div>Память: <span>{gpu['mem_used']} / {gpu['mem_total']} MB</span></div>
        </div>
        """
    html += f"""
    <div class='stat-block'>
        <h4>CPU</h4>
        <div>Load: <span>{cpu_stats['cpu_load_1']:.2f} / {cpu_stats['cpu_load_5']:.2f} / {cpu_stats['cpu_load_15']:.2f}</span></div>
        <div>RAM: <span>{cpu_stats['ram_used']} / {cpu_stats['ram_total']} MB</span></div>
    </div>
    """
    html += "</div>"
    return html

def render_history(history, stats_history):
    # Графики будут отрисованы на клиенте через JS (Chart.js)
    html = """
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
    <script>
    // Формируем данные для графиков (получаем их через эндпоинт /history_json)
    fetch('/history_json').then(r=>r.json()).then(data=>{
        let ctx1 = document.getElementById('gpu_history_chart').getContext('2d');
        let ctx2 = document.getElementById('req_history_chart').getContext('2d');
        let labels = data.timestamps;
        let gpu = data.gpu;
        let req = data.req;
        new Chart(ctx1, {
            type: 'line',
            data: {labels: labels, datasets:[{label:'GPU (%)', data: gpu, borderColor:'#4fd1c5', backgroundColor:'rgba(79,209,197,0.1)'}]},
            options: {scales:{y:{min:0,max:100}}}
        });
        new Chart(ctx2, {
            type: 'line',
            data: {labels: labels, datasets:[{label:'Запросы', data: req, borderColor:'#f6ad55', backgroundColor:'rgba(246,173,85,0.1)'}]},
            options: {scales:{y:{beginAtZero:true}}}
        });
    });
    </script>
    """
    return html
