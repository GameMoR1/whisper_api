import threading
import uuid
from datetime import datetime

class TasksManager:
    def __init__(self):
        self.tasks = {}
        self.lock = threading.Lock()

    def add_task(self, filename, model_name):
        task_id = str(uuid.uuid4())
        with self.lock:
            self.tasks[task_id] = {
                "id": task_id,
                "filename": filename,
                "model": model_name,
                "status": "pending",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": None
            }
        return task_id

    def update_task(self, task_id, status, error=None):
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id]["status"] = status
                self.tasks[task_id]["error"] = error

    def get_tasks(self):
        with self.lock:
            return [task for task in self.tasks.values() if task["status"] in ("pending", "processing")]
