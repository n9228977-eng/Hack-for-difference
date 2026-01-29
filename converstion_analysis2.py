# Install the requests package by executing the command "pip install requests"

import requests
import time

base_url = "https://api.assemblyai.com"

headers = {
    "authorization": "c4df9797813548eaa993c916486f1db5"
}
# You can upload a local file using the following code
with open("C:\\Users\\NBhadouria\\Downloads\\Candor Techspace.m4a", "rb") as f:
  response = requests.post(base_url + "/v2/upload",
                          headers=headers,
                          data=f)

audio_url = response.json()["upload_url"]

# audio_url = "https://assembly.ai/wildfires.mp3"
print(f"Audio URL: {audio_url}")

data = {
    "audio_url": audio_url,
    "speech_models": ["universal"],"language_code": "hi",

    "speaker_labels": True,  # Enable speaker labels
  # "language_detection": True,  # Enable language detection if you are processing files in a variety of languages
  "speech_understanding": {
    "request": {
      "translation": {
        "target_languages": ["en"],  # Translate to Spanish and German
        "formal": True  # Use formal language style
      }
    }
  }
}

url = base_url + "/v2/transcript"
response = requests.post(url, json=data, headers=headers)
# print(response.json())
transcript_id = response.json()['id']
polling_endpoint = base_url + "/v2/transcript/" + transcript_id

while True:
  transcription_result = requests.get(polling_endpoint, headers=headers).json()

  transcript_text = transcription_result['text']

  if transcription_result['status'] == 'completed':
    print(f"Transcript Text:", transcript_text)
 
    print("--- Translations ---")
    for language_code, translated_text in transcription_result['translated_texts'].items():
        print(f"{language_code.upper()}:")
        print(translated_text + "...\n")
    break

  elif transcription_result['status'] == 'error':
    raise RuntimeError(f"Transcription failed: {transcription_result['error']}")

  else:
    time.sleep(3)