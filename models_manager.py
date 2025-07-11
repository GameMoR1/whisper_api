import threading
import whisper

WHISPER_MODELS = [
    "tiny", "base", "small", "medium", "large-v2", "large-v3"
]

class ModelsManager:
    def __init__(self):
        self.models = {name: {"loaded": False, "error": None} for name in WHISPER_MODELS}
        self.lock = threading.Lock()
        self._init_models_async()

    def _init_models_async(self):
        threading.Thread(target=self._download_all_models, daemon=True).start()

    def _download_all_models(self):
        for name in WHISPER_MODELS:
            try:
                whisper.load_model(name)
                with self.lock:
                    self.models[name]["loaded"] = True
                    self.models[name]["error"] = None
            except Exception as e:
                with self.lock:
                    self.models[name]["loaded"] = False
                    self.models[name]["error"] = str(e)

    def get_status(self):
        with self.lock:
            return {name: dict(info) for name, info in self.models.items()}

    def is_model_loaded(self, name):
        with self.lock:
            return self.models.get(name, {}).get("loaded", False)
