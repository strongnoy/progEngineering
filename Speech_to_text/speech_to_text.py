from transformers import AutoProcessor, AutoModelForCTC
import torch
import librosa

# Load model directly

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")

audio_input, _ = librosa.load("harvard.wav", sr=16000)

inputs = processor(audio_input, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(input_values=inputs.input_values).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])

print("Распознанный текст:", transcription)