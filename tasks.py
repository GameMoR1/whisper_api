import threading
import queue
import uuid
import os
import time

class TaskStatus:
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"

class Task:
    def __init__(self, file_path, model_name, gpu_id, initial_prompt=None, upgrade_transcribation=False):
        self.id = str(uuid.uuid4())
        self.file_path = file_path
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.initial_prompt = initial_prompt
        self.upgrade_transcribation = upgrade_transcribation
        self.status = TaskStatus.QUEUED
        self.result_path = None
        self.result_text = None
        self.error = None
        self.created_at = time.time()
        self.started_at = None
        self.finished_at = None

class TaskManager:
    def __init__(self, model_manager, log_manager, gpu_monitor):
        self.model_manager = model_manager
        self.log_manager = log_manager
        self.gpu_monitor = gpu_monitor

        self.lock = threading.Lock()
        self.tasks = {}  # id -> Task
        self.queues = {gpu_id: queue.Queue() for gpu_id in self.gpu_monitor.get_gpu_ids()}
        self.active_tasks = {gpu_id: None for gpu_id in self.gpu_monitor.get_gpu_ids()}

        for gpu_id in self.gpu_monitor.get_gpu_ids():
            threading.Thread(target=self._worker, args=(gpu_id,), daemon=True).start()

    def add_task(self, file_path, model_name, initial_prompt=None, upgrade_transcribation=False, priority=False):
        with self.lock:
            # Находим свободный GPU или кладём в очередь к наименее загруженному
            gpu_id = self._choose_gpu()
            task = Task(file_path, model_name, gpu_id, initial_prompt, upgrade_transcribation)
            self.tasks[task.id] = task
            if priority:
                self._put_priority(gpu_id, task)
            else:
                self.queues[gpu_id].put(task)
            self.log_manager.log(f"Task {task.id} added to GPU {gpu_id} queue (model: {model_name})")
            return task

    def _put_priority(self, gpu_id, task):
        # Вставить задачу в начало очереди
        q = self.queues[gpu_id]
        with q.mutex:
            q.queue.appendleft(task)

    def _choose_gpu(self):
        # Выбираем GPU с наименьшей длиной очереди + без активной задачи
        min_len = None
        min_gpu = None
        for gpu_id, q in self.queues.items():
            length = q.qsize()
            if self.active_tasks[gpu_id] is None:
                length -= 0.5  # Свободный GPU — приоритетнее
            if min_len is None or length < min_len:
                min_len = length
                min_gpu = gpu_id
        return min_gpu

    def _worker(self, gpu_id):
        while True:
            task = self.queues[gpu_id].get()
            with self.lock:
                self.active_tasks[gpu_id] = task
                task.status = TaskStatus.RUNNING
                task.started_at = time.time()
            self.log_manager.log(f"Task {task.id} started on GPU {gpu_id}")
            try:
                result_path, result_text = self._run_task(task, gpu_id)
                task.status = TaskStatus.DONE
                task.result_path = result_path
                task.result_text = result_text
                task.finished_at = time.time()
                self.log_manager.log(f"Task {task.id} finished on GPU {gpu_id}")
            except Exception as e:
                task.status = TaskStatus.ERROR
                task.error = str(e)
                task.finished_at = time.time()
                self.log_manager.log(f"Task {task.id} failed on GPU {gpu_id}: {e}")
            finally:
                with self.lock:
                    self.active_tasks[gpu_id] = None
                self.gpu_monitor.clear_gpu_cache(gpu_id)

    def _run_task(self, task, gpu_id):
        # Импортируем whisper и torch здесь, чтобы избежать проблем с многопоточностью
        import whisper
        import torch

        model = self.model_manager.get_model(task.model_name)
        if model is None:
            raise Exception(f"Model {task.model_name} is not downloaded yet!")

        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        transcribe_kwargs = {
            "fp16": (device.startswith("cuda")),
            "beam_size": 5
        }
        if task.initial_prompt:
            transcribe_kwargs["initial_prompt"] = task.initial_prompt

        result = model.transcribe(task.file_path, **transcribe_kwargs)
        formatted_text = self._format_segments(result['segments'])

        # Сохраняем результат в файл
        base = os.path.splitext(os.path.basename(task.file_path))[0]
        result_path = f"{base}_result.txt"
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(formatted_text)

        return result_path, formatted_text

    def _format_segments(self, segments):
        def format_timestamp(seconds):
            m, s = divmod(int(seconds), 60)
            return f"{m:02d}:{s:02d}"
        lines = []
        for seg in segments:
            start = format_timestamp(seg['start'])
            text = seg['text'].strip()
            lines.append(f"[{start}] {text}")
        return "\n".join(lines)

    def get_active_tasks(self):
        with self.lock:
            return [task for task in self.active_tasks.values() if task is not None]

    def get_queued_tasks(self):
        with self.lock:
            tasks = []
            for q in self.queues.values():
                with q.mutex:
                    tasks += list(q.queue)
            return tasks

    def get_all_tasks(self):
        with self.lock:
            return list(self.tasks.values())
