import threading
import uuid
from datetime import datetime, timedelta

class Task:
    def __init__(self, filename, model_name):
        self.id = str(uuid.uuid4())
        self.filename = filename
        self.model = model_name
        self.status = "pending"
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.started_at = None
        self.finished_at = None
        self.error = None
        self.result_text = None
        self.gpu_id = None

class TasksManager:
    def __init__(self, max_history_hours=48):
        self.tasks = []
        self.lock = threading.Lock()
        self.max_history = timedelta(hours=max_history_hours)
        self.gpu_states = {}

    def set_gpus(self, gpu_ids):
        with self.lock:
            for gpu_id in gpu_ids:
                if gpu_id not in self.gpu_states:
                    self.gpu_states[gpu_id] = None

    def add_task(self, filename, model_name):
        with self.lock:
            task = Task(filename, model_name)
            self.tasks.append(task)
            return task.id

    def get_task(self, task_id):
        with self.lock:
            for task in self.tasks:
                if task.id == task_id:
                    return task
        return None

    def get_queue(self):
        with self.lock:
            return [task for task in self.tasks if task.status == "pending"]

    def get_processing(self):
        with self.lock:
            return [task for task in self.tasks if task.status == "processing"]

    def get_history(self):
        cutoff = (datetime.now() - self.max_history).strftime("%Y-%m-%d %H:%M:%S")
        with self.lock:
            return [task for task in self.tasks if task.status in ("done", "error") and task.created_at >= cutoff]

    def assign_gpu_to_task(self, task_id):
        with self.lock:
            for gpu_id, assigned_task in self.gpu_states.items():
                if assigned_task is None:
                    self.gpu_states[gpu_id] = task_id
                    task = self.get_task(task_id)
                    if task:
                        task.gpu_id = gpu_id
                        task.status = "processing"
                        task.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    return gpu_id
            return None

    def release_gpu(self, gpu_id):
        with self.lock:
            self.gpu_states[gpu_id] = None

    def update_task_done(self, task_id, result_text=None):
        with self.lock:
            task = self.get_task(task_id)
            if task:
                task.status = "done"
                task.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                task.result_text = result_text
                if task.gpu_id is not None:
                    self.gpu_states[task.gpu_id] = None
                    task.gpu_id = None

    def update_task_error(self, task_id, error_msg):
        with self.lock:
            task = self.get_task(task_id)
            if task:
                task.status = "error"
                task.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                task.error = error_msg
                if task.gpu_id is not None:
                    self.gpu_states[task.gpu_id] = None
                    task.gpu_id = None
