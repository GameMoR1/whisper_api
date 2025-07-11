import threading
import uuid
import sqlite3
from datetime import datetime, timedelta

class Task:
    def __init__(self, id, filename, model, status, created_at, started_at, finished_at, error, result_text, gpu_id):
        self.id = id
        self.filename = filename
        self.model = model
        self.status = status
        self.created_at = created_at
        self.started_at = started_at
        self.finished_at = finished_at
        self.error = error
        self.result_text = result_text
        self.gpu_id = gpu_id

class TasksManager:
    def __init__(self, db_path="tasks.sqlite", max_history_hours=48):
        self.db_path = db_path
        self.max_history = timedelta(hours=max_history_hours)
        self.lock = threading.Lock()
        self.gpu_states = {}  # gpu_id -> task_id или None
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    filename TEXT,
                    model TEXT,
                    status TEXT,
                    created_at TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    error TEXT,
                    result_text TEXT,
                    gpu_id INTEGER
                )
            """)

    def set_gpus(self, gpu_ids):
        with self.lock:
            for gpu_id in gpu_ids:
                if gpu_id not in self.gpu_states:
                    self.gpu_states[gpu_id] = None

    def add_task(self, filename, model_name):
        task_id = str(uuid.uuid4())
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO tasks (id, filename, model, status, created_at, started_at, finished_at, error, result_text, gpu_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (task_id, filename, model_name, "pending", now, None, None, None, None, None))
            conn.commit()
        return task_id

    def get_task(self, task_id):
        with self.lock, sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
            row = cur.fetchone()
            if row:
                return Task(*row)
        return None

    def get_queue(self):
        with self.lock, sqlite3.connect(self.db_path) as conn:
            cutoff = (datetime.now() - self.max_history).strftime("%Y-%m-%d %H:%M:%S")
            cur = conn.execute("SELECT * FROM tasks WHERE status = 'pending' AND created_at >= ? ORDER BY created_at", (cutoff,))
            return [Task(*row) for row in cur.fetchall()]

    def get_processing(self):
        with self.lock, sqlite3.connect(self.db_path) as conn:
            cutoff = (datetime.now() - self.max_history).strftime("%Y-%m-%d %H:%M:%S")
            cur = conn.execute("SELECT * FROM tasks WHERE status = 'processing' AND created_at >= ? ORDER BY started_at", (cutoff,))
            return [Task(*row) for row in cur.fetchall()]

    def get_history(self):
        with self.lock, sqlite3.connect(self.db_path) as conn:
            cutoff = (datetime.now() - self.max_history).strftime("%Y-%m-%d %H:%M:%S")
            cur = conn.execute("SELECT * FROM tasks WHERE status IN ('done', 'error') AND created_at >= ? ORDER BY finished_at DESC", (cutoff,))
            return [Task(*row) for row in cur.fetchall()]

    def assign_gpu_to_task(self, task_id):
        with self.lock:
            for gpu_id, assigned_task in self.gpu_states.items():
                if assigned_task is None:
                    self.gpu_states[gpu_id] = task_id
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute("UPDATE tasks SET status = 'processing', started_at = ?, gpu_id = ? WHERE id = ?", (now, gpu_id, task_id))
                        conn.commit()
                    return gpu_id
            return None

    def release_gpu(self, gpu_id):
        with self.lock:
            self.gpu_states[gpu_id] = None

    def update_task_done(self, task_id, result_text=None):
        with self.lock, sqlite3.connect(self.db_path) as conn:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn.execute("UPDATE tasks SET status = 'done', finished_at = ?, result_text = ? WHERE id = ?", (now, result_text, task_id))
            conn.commit()

    def update_task_error(self, task_id, error_msg):
        with self.lock, sqlite3.connect(self.db_path) as conn:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn.execute("UPDATE tasks SET status = 'error', finished_at = ?, error = ? WHERE id = ?", (now, error_msg, task_id))
            conn.commit()
