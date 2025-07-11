import threading
import whisper

class ModelsManager:
    def __init__(self, model_names, download_root):
        self.status = {name: {"loaded": False, "loading": False, "error": None} for name in model_names}
        self.models = {}
        self.download_root = download_root
        self.lock = threading.Lock()
        for name in model_names:
            threading.Thread(target=self._load_model_cpu, args=(name,), daemon=True).start()

    def _load_model_cpu(self, name):
        with self.lock:
            self.status[name]["loading"] = True
        try:
            # Модель загружается только в CPU, чтобы не занимать видеопамять!
            model = whisper.load_model(name, device="cpu", download_root=self.download_root)
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

    def get_model(self, name, device="cpu"):
        # Возвращает модель на нужном устройстве (CPU или GPU)
        with self.lock:
            model = self.models.get(name)
        if model is None:
            return None
        if device == "cpu":
            return model
        # Если требуется GPU — копируем модель на GPU только для транскрибации
        import torch
        if torch.cuda.is_available():
            return model.to("cuda")
        return model
