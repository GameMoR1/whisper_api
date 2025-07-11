import threading
from datetime import datetime

class LogsManager:
    def __init__(self, max_logs=200):
        self.logs = []
        self.max_logs = max_logs
        self.lock = threading.Lock()

    def log(self, message, level="INFO"):
        with self.lock:
            entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "level": level,
                "message": message
            }
            self.logs.append(entry)
            if len(self.logs) > self.max_logs:
                self.logs = self.logs[-self.max_logs:]

    def get_logs(self):
        with self.lock:
            return list(self.logs)
