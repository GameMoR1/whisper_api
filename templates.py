import os

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
        if info.get("loaded"):
            status = "<span class='ok'>✅</span>"
        elif info.get("loading"):
            status = "<span class='wait'>⏳</span>"
        else:
            status = "<span class='err'>❌</span>"
        error = f"<span class='err'>{info['error']}</span>" if info.get("error") else ""
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
        html += f"<tr><td>{task.id[:8]}</td><td>{os.path.basename(task.filename)}</td><td>{task.model}</td><td>{task.gpu_id}</td><td>{task.started_at if task.started_at else ''}</td><td><span class='proc'>Выполняется</span></td></tr>"
    html += "</table>"
    html += "<table class='tasks-table'><tr><th>ID</th><th>Файл</th><th>Модель</th><th>Время постановки</th><th>Статус</th></tr>"
    for task in queue:
        html += f"<tr><td>{task.id[:8]}</td><td>{os.path.basename(task.filename)}</td><td>{task.model}</td><td>{task.created_at}</td><td><span class='wait'>В очереди</span></td></tr>"
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
    # Графики рисуются JS-скриптом на главной странице, тут только контейнеры
    return ""
