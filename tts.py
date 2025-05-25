import pyttsx3
import os

# Initialize TTS engine
engine = pyttsx3.init()

# Create output directory
output_dir = 'output_audio_3'
os.makedirs(output_dir, exist_ok=True)

# Manually defined speech messages
messages = [
    "Lighting Condition is good,person is moving, towards the camera, at 23.49 meters, chair is static,at 25.61"

]

# Generate audio for each message
for idx, message in enumerate(messages, start=1):
    filename = f'message_{idx}.mp3'  # Change to .wav if needed
    output_path = os.path.join(output_dir, filename)
    print(f"Saving: {output_path} -> \"{message}\"")
    engine.save_to_file(message, output_path)

# Run the speech engine to save all files
engine.runAndWait()

print("All messages have been saved as audio files.")