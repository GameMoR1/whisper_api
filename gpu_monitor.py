import torch

class GPUInfo:
    def __init__(self, gpu_id):
        self.gpu_id = gpu_id
        self.name = torch.cuda.get_device_name(gpu_id)
        self.memory_total = torch.cuda.get_device_properties(gpu_id).total_memory // (1024 * 1024)  # MB

    def get_memory_used(self):
        return torch.cuda.memory_allocated(self.gpu_id) // (1024 * 1024)  # MB

    def get_utilization(self):
        # Для загрузки ядра используем nvidia-smi через subprocess
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", f"--id={self.gpu_id}"],
                capture_output=True, text=True, timeout=1
            )
            util = int(result.stdout.strip())
            return util
        except Exception:
            return None

    def to_dict(self):
        return {
            "id": self.gpu_id,
            "name": self.name,
            "memory_total_MB": self.memory_total,
            "memory_used_MB": self.get_memory_used(),
            "utilization_percent": self.get_utilization()
        }

class GPUMonitor:
    def __init__(self):
        self.gpu_ids = list(range(torch.cuda.device_count()))

    def get_gpu_ids(self):
        return self.gpu_ids

    def get_all_info(self):
        return [GPUInfo(gpu_id).to_dict() for gpu_id in self.gpu_ids]

    def clear_gpu_cache(self, gpu_id):
        # Чистим память только на конкретном GPU
        torch.cuda.empty_cache()
