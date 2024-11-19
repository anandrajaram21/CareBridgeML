from typing import Annotated
from fastapi import FastAPI, File, UploadFile

app = FastAPI()


@app.post("/audio")
async def transcribe_and_translate(file: Annotated[bytes, File()]):
    with open("audio.wav", "wb") as f:
        f.write(file)

    transcribed_text = transcribe("audio.wav")

    candidate_labels = [
        "food",
        "medicine",
        "transportation",
        "household",
        "hygiene",
        "other",
    ]

    classification = classify(transcribed_text, candidate_labels)

    return {
        "transcribed_text": transcribed_text,
        "classification": classification,
    }
