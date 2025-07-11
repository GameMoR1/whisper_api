import subprocess
import os
from datetime import datetime
from collections import deque

class StatsHistory:
    def __init__(self, max_minutes=48*60):
        self.gpu_stats = deque(maxlen=max_minutes)
        self.cpu_stats = deque(maxlen=max_minutes)
        self.request_counts = deque(maxlen=max_minutes)
        self.timestamps = deque(maxlen=max_minutes)

    def add_stats(self, gpu, cpu, req_count):
        self.gpu_stats.append(gpu)
        self.cpu_stats.append(cpu)
        self.request_counts.append(req_count)
        self.timestamps.append(datetime.now())

def get_gpu_stats():
    try:
        result = subprocess.check_output(
            "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits",
            shell=True
        ).decode().strip()
        stats = []
        for line in result.splitlines():
            idx, name, util, mem_used, mem_total = line.split(", ")
            stats.append({
                "id": int(idx),
                "name": name,
                "gpu_util": int(util),
                "mem_used": int(mem_used),
                "mem_total": int(mem_total)
            })
        return stats
    except Exception:
        return []

def get_cpu_ram_stats():
    load = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
    meminfo = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":")
                meminfo[k.strip()] = int(v.strip().split()[0])
        mem_total = meminfo.get("MemTotal", 0) // 1024
        mem_free = meminfo.get("MemAvailable", 0) // 1024
        mem_used = mem_total - mem_free
    except Exception:
        mem_total = mem_free = mem_used = 0
    return {
        "cpu_load_1": load[0],
        "cpu_load_5": load[1],
        "cpu_load_15": load[2],
        "ram_total": mem_total,
        "ram_used": mem_used,
        "ram_free": mem_free
    }
