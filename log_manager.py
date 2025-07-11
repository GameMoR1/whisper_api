import threading
import time

class LogManager:
    def __init__(self, max_logs=1000):
        self.logs = []
        self.lock = threading.Lock()
        self.max_logs = max_logs

    def log(self, message):
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "message": message
        }
        with self.lock:
            self.logs.append(entry)
            if len(self.logs) > self.max_logs:
                self.logs = self.logs[-self.max_logs:]

    def get_logs(self, limit=100):
        with self.lock:
            return self.logs[-limit:]
