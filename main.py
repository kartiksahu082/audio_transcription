import os
import uuid
import logging
import time
from fastapi import FastAPI, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from pydub import AudioSegment
import whisper
import torch
import numpy as np

# ----------------------------
# 1. Configuration and Setup
# ----------------------------

# Configure Logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Audio Transcription API")

# Mount static files and templates directories
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Define upload folder
UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Supported input audio formats
SUPPORTED_INPUT_FORMATS = ["wav", "aiff", "aifc", "flac", "mp3", "ogg", "m4a"]

# ----------------------------
# 2. Model Initialization
# ----------------------------

# Initialize Whisper models as a global dictionary
models = {}
AVAILABLE_MODELS = ["turbo", "base", "small"]  # Define available models

@app.on_event("startup")
def load_whisper_models():
    """
    Loads all available Whisper models at application startup and moves them to GPU if available.
    """
    global models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading Whisper models on device: {device}")
    try:
        for model_size in AVAILABLE_MODELS:
            logger.info(f"Loading Whisper model '{model_size}'...")
            loaded_model = whisper.load_model(model_size)
            loaded_model = loaded_model.to(device)
            models[model_size] = loaded_model
            logger.info(f"Whisper model '{model_size}' loaded successfully.")
    except Exception as e:
        logger.exception(f"Failed to load Whisper models: {e}")
        raise e

# ----------------------------
# 3. Helper Functions
# ----------------------------

def normalize_audio(audio_segment):
    """
    Normalizes the audio to a target dBFS.
    """
    target_dBFS = -20.0
    change_in_dBFS = target_dBFS - audio_segment.dBFS
    return audio_segment.apply_gain(change_in_dBFS)

def transcribe_with_whisper(audio_path, language="en", model_size="turbo"):
    """
    Transcribes audio using the specified Whisper model.
    Returns the transcription text and time taken.
    """
    global models
    if model_size not in models:
        logger.error(f"Requested model '{model_size}' is not loaded.")
        raise ValueError(f"Model '{model_size}' is not available.")

    try:
        model = models[model_size]
        device = next(model.parameters()).device
        logger.info(f"Transcribing using model '{model_size}' on device: {device}")

        # Handle 'auto' language detection
        if language.lower() == "auto":
            logger.info("Language set to auto-detect.")
            start_time = time.time()
            result = model.transcribe(audio_path, fp16=torch.cuda.is_available())
            end_time = time.time()
        else:
            logger.info(f"Language set to: {language}")
            start_time = time.time()
            result = model.transcribe(audio_path, language=language, fp16=torch.cuda.is_available())
            end_time = time.time()

        transcription_time = end_time - start_time  # Time in seconds
        transcription = result.get("text", "").strip()
        logger.debug(f"Transcription result: {transcription}")

        return transcription, round(transcription_time, 2)

    except Exception as e:
        logger.error(f"Whisper transcription error: {e}")
        raise e

def cleanup_files(*file_paths):
    """
    Deletes specified files from the filesystem.
    """
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.debug(f"Deleted file: {path}")
            else:
                logger.warning(f"File not found for deletion: {path}")
        except Exception as e:
            logger.error(f"Error deleting file {path}: {e}")

# ----------------------------
# 4. API Endpoints
# ----------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Serves the homepage.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/transcribe/", response_class=JSONResponse)
async def transcribe_audio(request: Request, background_tasks: BackgroundTasks,
                          file: UploadFile = Form(...), 
                          language: str = Form("en"),
                          model: str = Form("turbo")):
    """
    Handles the audio file upload, processing, and transcription using Whisper.
    Returns both the transcription and the time taken for transcription.
    """
    content = await file.read()
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
    if len(content) > MAX_FILE_SIZE:
        logger.warning(f"Uploaded file size {len(content)} exceeds the 50MB limit.")
        raise HTTPException(status_code=400, detail="File size exceeds the 50MB limit.")

    if model not in AVAILABLE_MODELS:
        logger.warning(f"Unsupported model selection: {model}")
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported model selection: {model}."}
        )

    try:
        filename = file.filename
        file_extension = filename.split(".")[-1].lower()

        if file_extension not in SUPPORTED_INPUT_FORMATS:
            logger.warning(f"Unsupported file format: {file_extension}")
            return JSONResponse(
                status_code=400,
                content={"error": f"Unsupported file format: {file_extension}."}
            )

        unique_id = uuid.uuid4().hex
        saved_filename = f"{unique_id}.{file_extension}"
        file_location = os.path.join(UPLOAD_FOLDER, saved_filename)

        # Save the uploaded file
        with open(file_location, "wb") as f:
            f.write(content)
        logger.info(f"Saved uploaded file to {file_location}")

        # Convert to WAV if necessary
        if file_extension != "wav":
            wav_filename = f"{unique_id}.wav"
            wav_file_location = os.path.join(UPLOAD_FOLDER, wav_filename)
            try:
                audio = AudioSegment.from_file(file_location, format=file_extension)
                audio = normalize_audio(audio)
                audio.export(wav_file_location, format="wav")
                logger.info(f"Converted {file_extension} to WAV at {wav_file_location}")
            except Exception as e:
                logger.error(f"Error converting file to WAV: {e}")
                raise HTTPException(status_code=500, detail="Error processing audio file.")
        else:
            wav_file_location = file_location
            logger.info(f"Uploaded file is already in WAV format.")

        # Normalize audio
        try:
            audio = AudioSegment.from_file(wav_file_location)
            audio = normalize_audio(audio)
            normalized_path = os.path.join(UPLOAD_FOLDER, f"normalized_{unique_id}.wav")
            audio.export(normalized_path, format="wav")
            logger.info(f"Normalized audio saved at {normalized_path}")
        except Exception as e:
            logger.error(f"Error normalizing audio: {e}")
            raise HTTPException(status_code=500, detail="Error processing audio file.")

        # Transcribe using Whisper
        try:
            transcription, transcription_time = transcribe_with_whisper(normalized_path, language=language, model_size=model)
            logger.info(f"Transcription successful for {normalized_path} in {transcription_time} seconds.")
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise HTTPException(status_code=500, detail="Transcription failed.")

        # Cleanup files after processing
        background_tasks.add_task(cleanup_files, file_location, wav_file_location, normalized_path)

        return {
            "transcription": transcription,
            "transcription_time_seconds": transcription_time,
            "model_used": model
        }

    except HTTPException as he:
        raise he  # Re-raise HTTP exceptions to be handled by FastAPI
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "An unexpected error occurred during transcription."}
        )
