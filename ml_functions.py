from transformers import pipeline
import torch

def classify(text, labels):
    # Formulate the prompt for GPT-3 or GPT-4
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)
    candidate_labels = labels
    result = classifier(text, candidate_labels)

    return result["labels"][0]


def transcribe(audio):
    # Load the Whisper model
    whisper = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        torch_dtype=torch.float16,
        device=0,  # Use GPU (CUDA)
    )
    # Transcribe the audio directly
    transcription = whisper(audio)["text"]
    return transcription

