import threading
import sqlite3
from datetime import datetime

class LogsManager:
    def __init__(self, logfile="logs.sqlite", max_logs=200):
        self.logfile = logfile
        self.max_logs = max_logs
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.logfile) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    level TEXT,
                    message TEXT
                )
            """)

    def log(self, message, level="INFO"):
        with self.lock, sqlite3.connect(self.logfile) as conn:
            entry = (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                level,
                message
            )
            conn.execute("INSERT INTO logs (timestamp, level, message) VALUES (?, ?, ?)", entry)
            conn.commit()

    def get_logs(self):
        with self.lock, sqlite3.connect(self.logfile) as conn:
            cur = conn.execute("SELECT timestamp, level, message FROM logs ORDER BY id DESC LIMIT ?", (self.max_logs,))
            return [{"timestamp": t, "level": l, "message": m} for t, l, m in cur.fetchall()][::-1]
