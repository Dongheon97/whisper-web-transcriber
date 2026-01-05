import os
import json
import time
import uuid
import threading
from typing import Dict, Any, Optional
import subprocess
import re
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware


APP_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(APP_DIR, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

WHISPER_SH = "<whisper_script_path>"
TS_RE = re.compile(r"\[(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)\s*-->\s*(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)\]")

app = FastAPI()

# (Optional) If you serve index.html from the same server, CORS is not needed,
# but allowing it for development convenience in case you open from a different port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Single-slot job manager
# -----------------------------
MAX_QUEUE = 10

job_lock = threading.Lock()
job_cv = threading.Condition(job_lock)

current_job_id: Optional[str] = None
queue: list[str] = []

# In-memory job store (MVP)
jobs: Dict[str, Dict[str, Any]] = {}

@app.get("/", response_class=HTMLResponse)
def home():
    with open(os.path.join(APP_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()

def job_path(job_id: str) -> str:
    p = os.path.join(JOBS_DIR, job_id)
    os.makedirs(p, exist_ok=True)
    return p

def save_status(job_id: str) -> None:
    p = os.path.join(job_path(job_id), "status.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(jobs[job_id], f, ensure_ascii=False, indent=2)

def hms_to_sec(t: str) -> float:
    # Supports "MM:SS.mmm" and "HH:MM:SS.mmm"
    parts = t.split(":")
    if len(parts) == 2:
        mm, ss = parts
        return int(mm) * 60 + float(ss)
    if len(parts) == 3:
        hh, mm, ss = parts
        return int(hh) * 3600 + int(mm) * 60 + float(ss)
    raise ValueError(f"bad time format: {t}")


def ffprobe_duration_sec(path: str) -> float:
    # Returns duration in seconds (float)
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1",
        path
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out) if out else 0.0

def find_latest_txt(job_id: str) -> Optional[str]:
    jp = job_path(job_id)
    txts = [os.path.join(jp, f) for f in os.listdir(jp) if f.lower().endswith(".txt")]
    if not txts:
        return None
    txts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return txts[0]

def run_whisper_processing(job_id: str) -> None:
    jp = job_path(job_id)
    input_path = jobs[job_id]["input_path"]

    # Total duration (sec)
    try:
        total = ffprobe_duration_sec(input_path)
    except Exception:
        total = 0.0

    with job_lock:
        jobs[job_id]["state"] = "processing"
        jobs[job_id]["total_sec"] = total if total > 0 else jobs[job_id].get("total_sec", 0)
        jobs[job_id]["processed_sec"] = 0.0
        jobs[job_id]["progress_pct"] = 0.0
        jobs[job_id]["detail"] = "Starting whisper…"
        jobs[job_id]["error"] = None
        save_status(job_id)

    cmd = [WHISPER_SH, input_path, jp]

    proc = subprocess.Popen(
        cmd,
        cwd=jp,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    log_path = os.path.join(jp, "whisper.log")
    log_f = open(log_path, "a", encoding="utf-8")

    last_processed = 0.0

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line); log_f.flush()

            m = TS_RE.search(line)
            if not m:
                continue

            end_ts = m.group(2)
            try:
                sec = hms_to_sec(end_ts)
            except Exception:
                continue

            if sec <= last_processed:
                continue

            last_processed = sec
            with job_lock:
                jobs[job_id]["processed_sec"] = last_processed
                total_sec = float(jobs[job_id].get("total_sec") or 0.0)
                if total_sec > 0:
                    pct = min(99.9, (last_processed / total_sec) * 100.0)
                    jobs[job_id]["progress_pct"] = pct
                    jobs[job_id]["detail"] = f"{last_processed:.0f}/{total_sec:.0f} sec"
                else:
                    jobs[job_id]["detail"] = f"Processed ~{last_processed:.0f} sec"
                save_status(job_id)

        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"whisper script failed (exit={ret})")

        out_txt = find_latest_txt(job_id)
        if not out_txt:
            raise RuntimeError("No .txt output found in job directory")

        with job_lock:
            jobs[job_id]["state"] = "done"
            jobs[job_id]["progress_pct"] = 100.0
            jobs[job_id]["processed_sec"] = float(jobs[job_id].get("total_sec") or jobs[job_id]["processed_sec"])
            jobs[job_id]["detail"] = "Done"
            jobs[job_id]["result_txt"] = out_txt
            save_status(job_id)

    except Exception as e:
        with job_lock:
            jobs[job_id]["state"] = "failed"
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["detail"] = "Failed"
            save_status(job_id)
        raise

    finally:
        log_f.close()

def worker_loop():
    global current_job_id
    while True:
        with job_cv:
            # Wait if queue is empty
            while not queue:
                current_job_id = None
                job_cv.wait()

            # Pop next job
            job_id = queue.pop(0)
            current_job_id = job_id

            # Transition from queued -> processing
            jobs[job_id]["state"] = "processing"
            jobs[job_id]["detail"] = "Starting transcription…"
            save_status(job_id)

        # Process outside the lock (important)
        try:
            run_whisper_processing(job_id)
        except Exception as e:
            jobs[job_id]["state"] = "failed"
            jobs[job_id]["error"] = str(e)
            save_status(job_id)
        finally:
            with job_cv:
                current_job_id = None
                job_cv.notify_all()  # Notify that state has changed (optional)

# -----------------------------
# API
# -----------------------------
@app.get("/status")
def status():
    with job_lock:
        busy = current_job_id is not None
        return {
            "state": "busy" if busy else "idle",
            "queue_length": len(queue) + (1 if busy else 0)  # Include currently running job in count if desired
        }

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Receives a video upload and starts a dummy transcription job.
    Response matches the UI expectation:
      { "job_id": "...", "token": "..." }
    """
    global current_job_id

    # Single-slot enforcement
    with job_lock:
        # Limit queue only to 10: len(queue) >= MAX_QUEUE
        # To limit "currently processing + queue" to 10: (len(queue) + (1 if current_job_id else 0)) >= MAX_QUEUE
        if len(queue) >= MAX_QUEUE:
            raise HTTPException(status_code=429, detail="Queue is full. Try later.")

    job_id = uuid.uuid4().hex[:12]
    token = uuid.uuid4().hex  # Simple secret token
    jp = job_path(job_id)

    # Save uploaded file
    in_path = os.path.join(jp, "input_" + (file.filename or "video"))
    with open(in_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    # For demo, we don't probe duration.
    # Set a fixed "total_sec" to make progress meaningful.
    # You will replace this with ffprobe later.
    total_sec = 5400  # Pretend 90 minutes

    jobs[job_id] = {
        "input_path": in_path,
        "state": "uploading",
        "job_id": job_id,
        "token": token,
        "filename": file.filename or "uploaded_video",
        "created_at": time.time(),
        "total_sec": total_sec,
        "processed_sec": 0,
        "progress_pct": 0.0,
        "detail": "Queued",
        "error": None,
    }
    save_status(job_id)

    # Mark slot officially as this job_id
    with job_cv:
        jobs[job_id]["state"] = "queued"
        jobs[job_id]["detail"] = "Queued"
        save_status(job_id)

        queue.append(job_id)
        position = len(queue)  # 1-based position in waiting queue
        job_cv.notify()        # Wake up worker

    return {"job_id": job_id, "token": token, "queued": True, "position": position}


@app.get("/events")
def events(job_id: str, token: str):
    """
    SSE endpoint.
    Sends:
      event: progress  data: {processed_sec,total_sec,progress_pct,detail}
      event: done      data: {download_url}
      event: error     data: {message}
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="job not found")
    if jobs[job_id]["token"] != token:
        raise HTTPException(status_code=403, detail="forbidden")

    def gen():
        last_pct = -1.0
        # Keep streaming until done/failed
        while True:
            j = jobs.get(job_id)
            if not j:
                yield "event: error\ndata: {\"message\":\"job missing\"}\n\n"
                return

            state = j["state"]
            pct = float(j.get("progress_pct", 0.0))

            # Throttle redundant updates
            if pct != last_pct and state in ("processing", "uploading"):
                payload = {
                    "processed_sec": j.get("processed_sec", 0),
                    "total_sec": j.get("total_sec", 1),
                    "progress_pct": pct,
                    "detail": j.get("detail", "Processing…"),
                }
                yield f"event: progress\ndata: {json.dumps(payload)}\n\n"
                last_pct = pct

            if state == "done":
                download_url = f"/download?job_id={job_id}&token={token}"
                yield f"event: done\ndata: {json.dumps({'download_url': download_url})}\n\n"
                return

            if state == "failed":
                msg = j.get("error") or "processing failed"
                yield f"event: error\ndata: {json.dumps({'message': msg})}\n\n"
                return
            
            if state == "queued":
                # Calculate position of this job in the queue (pushed back if there's 1 currently processing)
                with job_lock:
                    ahead_in_queue = queue.index(job_id) if job_id in queue else 0
                    if current_job_id is not None and current_job_id != job_id:
                        ahead_total = ahead_in_queue + 1  # Include the 1 running job
                    else:
                        ahead_total = ahead_in_queue

                    payload = {
                        "position_ahead": ahead_total,  # Number of jobs ahead (including running)
                        "queue_length": len(queue) + (1 if current_job_id is not None else 0),
                        "detail": "Queued"
                    }
                yield f"event: queued\ndata: {json.dumps(payload)}\n\n"
                time.sleep(1.0)
                continue

            time.sleep(0.3)

    return StreamingResponse(gen(), media_type="text/event-stream")

@app.get("/download")
def download(job_id: str, token: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="job not found")
    if jobs[job_id]["token"] != token:
        raise HTTPException(status_code=403, detail="forbidden")

    out_txt = jobs[job_id].get("result_txt") or find_latest_txt(job_id)
    if not out_txt or not os.path.exists(out_txt):
        raise HTTPException(status_code=404, detail="result not ready")

    # Pre-calculate filename
    fname = os.path.splitext(jobs[job_id]["filename"])[0] + ".txt"
    jp = job_path(job_id)

    # ✅ Schedule deletion after 10 minutes (prevent duplicate scheduling)
    with job_lock:
        if not jobs[job_id].get("cleanup_scheduled"):
            jobs[job_id]["cleanup_scheduled"] = True
            save_status(job_id)

            def _cleanup():
                # Protection: don't delete if still processing/incomplete (optional)
                with job_lock:
                    j = jobs.get(job_id)
                    if not j:
                        return
                    if j.get("state") not in ("done", "failed"):
                        # Skip if still processing
                        j["cleanup_scheduled"] = False
                        save_status(job_id)
                        return

                shutil.rmtree(jp, ignore_errors=True)
                with job_lock:
                    jobs.pop(job_id, None)

            threading.Timer(600, _cleanup).start()  # 600 seconds = 10 minutes

    return FileResponse(out_txt, media_type="text/plain", filename=fname)


# Start one background worker thread
threading.Thread(target=worker_loop, daemon=True).start()
