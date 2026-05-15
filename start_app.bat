@echo off
REM Activate virtual environment
if exist .venv\Scripts\activate (
    call .venv\Scripts\activate
) else if exist venv\Scripts\activate (
    call venv\Scripts\activate
)

REM Run the FastAPI gateway, which supervises the Streamlit workers
python run_app.py
