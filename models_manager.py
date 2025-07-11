import threading
import whisper

class ModelsManager:
    def __init__(self, model_names, download_root):
        self.status = {name: {"loaded": False, "loading": False, "error": None} for name in model_names}
        self.models = {}
        self.download_root = download_root
        self.lock = threading.Lock()
        for name in model_names:
            threading.Thread(target=self._load_model, args=(name,), daemon=True).start()

    def _load_model(self, name):
        with self.lock:
            self.status[name]["loading"] = True
        try:
            model = whisper.load_model(name, download_root=self.download_root)
            with self.lock:
                self.models[name] = model
                self.status[name]["loaded"] = True
        except Exception as e:
            with self.lock:
                self.status[name]["error"] = str(e)
        finally:
            with self.lock:
                self.status[name]["loading"] = False

    def is_model_loaded(self, name):
        with self.lock:
            return self.status.get(name, {}).get("loaded", False)

    def is_model_loading(self, name):
        with self.lock:
            return self.status.get(name, {}).get("loading", False)

    def get_status(self):
        with self.lock:
            return dict(self.status)

    def get_model(self, name):
        with self.lock:
            return self.models.get(name)
