def render_status(count, gpus, models_status):
    html = f"<p>Доступно GPU: <span class='gpu-count'>{count}</span></p>"
    if count > 0:
        html += "<ul class='gpu-list'>"
        for gpu in gpus:
            html += f"<li>ID: {gpu['id']} | <b>{gpu['name']}</b> | {gpu['memory_total_MB']} MB</li>"
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

def render_tasks(tasks):
    html = "<table class='tasks-table'><tr><th>ID</th><th>Файл</th><th>Модель</th><th>Статус</th><th>Ошибка</th><th>Создано</th></tr>"
    for task in tasks:
        status_map = {
            "pending": "<span class='wait'>В очереди</span>",
            "processing": "<span class='proc'>Выполняется</span>",
            "done": "<span class='ok'>Готово</span>",
            "error": "<span class='err'>Ошибка</span>"
        }
        status = status_map.get(task["status"], task["status"])
        error = f"<span class='err'>{task['error']}</span>" if task["error"] else ""
        html += f"<tr><td>{task['id'][:8]}</td><td>{task['filename']}</td><td>{task['model']}</td><td>{status}</td><td>{error}</td><td>{task['created_at']}</td></tr>"
    html += "</table>"
    return html
