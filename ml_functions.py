import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def classify(text, labels):
    # Formulate the prompt for GPT-3 or GPT-4
    prompt = (
        f"Classify the following text into one of the given categories.\n\n"
        f"Text: '{text}'\n"
        f"Categories: {', '.join(labels)}\n\n"
        f"Return only the name of the chosen category."
    )

    # Use the OpenAI API to get the classification
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful assistant for text classification."},
                  {"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.3
    )

    # Extract the chosen category from the response
    chosen_category = response['choices'][0]['message']['content'].strip()

    # Ensure the chosen category is valid
    if chosen_category not in labels:
        chosen_category = "Unknown"
    
    return chosen_category


def transcribe(audio):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    with open("audio.wav", "wb") as f:
        f.write(audio)

    with open("audio.wav", "rb") as f:
        transcript = openai.Audio.translate("whisper-1", f)

    return transcript

