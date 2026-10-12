@echo off
cd /d "%~dp0"
rem Optional: DeepFace (age, gender, emotion) starts in its own window when Python and deepface are installed.
python -c "import deepface" >nul 2>&1 && start "DeepFace" python deepface_server.py
start "" http://localhost:8080
node server.mjs
