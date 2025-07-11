import os
import tempfile
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import whisper
import torch
import g4f

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
loaded_models = {}

def get_model(model_name: str, download_dir: str = None):
    key = (model_name, download_dir)
    if key not in loaded_models:
        loaded_models[key] = whisper.load_model(model_name, device=DEVICE, download_root=download_dir)
    return loaded_models[key]

def format_timestamp(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"

def format_segments(segments):
    # Формат: [mm:ss] Текст
    lines = []
    for seg in segments:
        start = format_timestamp(seg['start'])
        text = seg['text'].strip()
        lines.append(f"[{start}] {text}")
    return "\n".join(lines)

def gpt_chat(prompt: str) -> str:
    """
    Sends a prompt to ChatGPT via g4f and returns the response text.
    """
    response = g4f.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.strip()

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def root():
    count, gpus = get_gpu_info()
    html = "<h2>API работает корректно</h2>"
    html += f"<p>Доступно GPU: {count}</p>"
    if count > 0:
        html += "<ul>"
        for gpu in gpus:
            html += f"<li>ID: {gpu['id']}, Name: {gpu['name']}, Memory: {gpu['memory_total_MB']} MB</li>"
        html += "</ul>"
    else:
        html += "<p>GPU не обнаружены</p>"
    return html

@app.post("/transcribe/")
async def transcribe(
    file: UploadFile = File(...),
    model_name: str = Form("base"),
    model_dir: str = Form(None),
    initial_prompt: str = Form(None),
    upgrade_transcribation: bool = Form(False)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        model = get_model(model_name, model_dir)
        transcribe_kwargs = {
            "fp16": (DEVICE == "cuda"),
            "beam_size": 5
        }
        if initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt

        result = model.transcribe(tmp_path, **transcribe_kwargs)
        formatted_text = format_segments(result['segments'])

        # Если требуется улучшение транскрипции через GPT
        if upgrade_transcribation:
            gpt_prompt = (
                "Вот расшифровка диалога между двумя спикерами: сотрудником и клиентом. "
                "Раздели текст по репликам спикеров, подпиши кто говорит (Сотрудник или Клиент), "
                "исправь явные ошибки и сделай текст более читабельным. "
                "Сохрани тайминги в формате [mm:ss] перед каждой репликой. "
                "Учитывай, что в таймингах, которые уже есть - ошибки. "
                "Иногда реплика незакончена, а тайминг подписан. Не обрывай реплики таймингами. "
                "Если спикер не закончил говорить, не пиши тайминг.\n\n"
                f"{formatted_text}"
            )
            try:
                improved_text = gpt_chat(gpt_prompt)
                return JSONResponse({"text": improved_text})
            except Exception as e:
                # Если GPT не сработал, возвращаем обычную транскрипцию с пометкой об ошибке
                return JSONResponse({
                    "text": formatted_text,
                    "warning": f"Ошибка улучшения через GPT: {str(e)}"
                })
        else:
            return JSONResponse({"text": formatted_text})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        os.remove(tmp_path)
        torch.cuda.empty_cache()
