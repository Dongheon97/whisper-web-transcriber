#!/usr/bin/env bash

source /home/<user>/anaconda3/bin/activate
conda activate whisper
cd <pacakge_path>/whisper_service
exec uvicorn app:app --host 0.0.0.0 --port 8000
