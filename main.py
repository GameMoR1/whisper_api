import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from models_manager import ModelsManager
from logs_manager import LogsManager
from tasks_manager import TasksManager
from routes import router, background_queue_worker

# Пути для хранения данных и моделей вне папки с кодом
DATA_DIR = os.environ.get("WHISPER_API_DATA", os.path.expanduser("~/whisper_api_data"))
MODELS_DIR = os.environ.get("WHISPER_MODELS", os.path.expanduser("~/.cache/whisper_models"))

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]

app = FastAPI(title="Whisper API (OOP)")

models_manager = ModelsManager(WHISPER_MODELS, download_root=MODELS_DIR)
logs_manager = LogsManager(logfile=os.path.join(DATA_DIR, "logs.sqlite"))
tasks_manager = TasksManager(db_path=os.path.join(DATA_DIR, "tasks.sqlite"))

import routes
routes.models_manager = models_manager
routes.logs_manager = logs_manager
routes.tasks_manager = tasks_manager

app.include_router(router)
app.mount("/static", StaticFiles(directory="static"), name="static")

import threading
threading.Thread(target=background_queue_worker, daemon=True).start()
