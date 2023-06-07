from typing import Annotated
from ml_functions import classify, transcribe
from fastapi import FastAPI, File, UploadFile

app = FastAPI()


@app.post("/audio")
async def create_upload_file(file: Annotated[bytes, File()]):
    transcribed_text = transcribe(file)

    candidate_labels = [
        "food",
        "medicine",
        "transportation",
        "household",
        "hygiene",
        "other",
    ]

    classification = classify(transcribed_text.text, candidate_labels)

    return {
        "transcribed_text": transcribed_text.text,
        "classification": classification,
    }
