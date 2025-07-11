from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from models_manager import ModelsManager
from logs_manager import LogsManager
from tasks_manager import TasksManager
from routes import router, background_queue_worker

app = FastAPI(title="Whisper API (OOP)")

models_manager = ModelsManager(
    ["tiny", "base", "small", "medium", "large-v2", "large-v3"],
    download_root="~/.cache/whisper_models"
)
logs_manager = LogsManager()
tasks_manager = TasksManager()

import routes
routes.models_manager = models_manager
routes.logs_manager = logs_manager
routes.tasks_manager = tasks_manager

app.include_router(router)
app.mount("/static", StaticFiles(directory="static"), name="static")

import threading
threading.Thread(target=background_queue_worker, daemon=True).start()
