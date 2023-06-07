import openai
from transformers import pipeline
from dotenv import load_dotenv
import os

load_dotenv()


def classify(text, labels):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    candidate_labels = labels
    result = classifier(text, candidate_labels)

    return result["labels"][0]


def transcribe(audio):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    with open("audio.wav", "wb") as f:
        f.write(audio)

    with open("audio.wav", "rb") as f:
        transcript = openai.Audio.translate("whisper-1", f)

    return transcript
