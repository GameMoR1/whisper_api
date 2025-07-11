import threading
import whisper
import torch

WHISPER_MODELS = [
    "tiny", "base", "small", "medium", "large-v2", "large-v3"
]

class ModelStatus:
    def __init__(self, name):
        self.name = name
        self.downloaded = False
        self.progress = 0
        self.lock = threading.Lock()

class ModelManager:
    def __init__(self):
        self.models = {name: ModelStatus(name) for name in WHISPER_MODELS}
        self.loaded_models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def start_downloads(self):
        for name in WHISPER_MODELS:
            threading.Thread(target=self._download_model, args=(name,), daemon=True).start()

    def _download_model(self, name):
        status = self.models[name]
        with status.lock:
            try:
                # Whisper не предоставляет прогресс напрямую, поэтому просто выставляем 100% после загрузки
                model = whisper.load_model(name, device=self.device)
                self.loaded_models[name] = model
                status.downloaded = True
                status.progress = 100
            except Exception as e:
                status.downloaded = False
                status.progress = 0

    def is_downloaded(self, name):
        return self.models[name].downloaded

    def get_progress(self, name):
        return self.models[name].progress

    def get_model(self, name):
        if self.is_downloaded(name):
            return self.loaded_models[name]
        return None

    def get_statuses(self):
        return [
            {
                "name": status.name,
                "downloaded": status.downloaded,
                "progress": status.progress
            }
            for status in self.models.values()
        ]
