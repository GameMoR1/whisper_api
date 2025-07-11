import threading
import whisper
import os

class ModelsManager:
    def __init__(self, model_names, download_root):
        self.status = {name: {"loaded": False, "loading": False, "error": None} for name in model_names}
        self.models = {}
        self.download_root = os.path.expanduser(download_root)
        self.lock = threading.Lock()
        os.makedirs(self.download_root, exist_ok=True)
        for name in model_names:
            threading.Thread(target=self._load_model_cpu, args=(name,), daemon=True).start()

    def _model_cached(self, name):
        # Whisper кладёт веса в папку download_root/ или ~/.cache/whisper/
        model_dir = os.path.join(self.download_root, name)
        # Проверяем по наличию файла weights
        for file in os.listdir(self.download_root):
            if name in file and file.endswith('.pt'):
                return True
        return False

    def _load_model_cpu(self, name):
        with self.lock:
            self.status[name]["loading"] = True
        try:
            if not self._model_cached(name):
                # Только если нет на диске — скачиваем
                model = whisper.load_model(name, device="cpu", download_root=self.download_root)
            else:
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
        with self.lock:
            model = self.models.get(name)
        if model is None:
            return None
        if device == "cpu":
            return model
        import torch
        if torch.cuda.is_available():
            return model.to("cuda")
        return model
