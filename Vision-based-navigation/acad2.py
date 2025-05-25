import cv2
import torch
import os
import numpy as np
from ultralytics import YOLO
from transformers import DPTImageProcessor, DPTForDepthEstimation
import traceback
import threading
import queue
import time

# Output directories
output_dir = 'vision_navigation_output'
predictions_dir = 'predictions_txt'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(predictions_dir, exist_ok=True)

# File to store all spoken feedback
speech_log_file = os.path.join(output_dir, "speech_output.txt")
with open(speech_log_file, "w") as f:  # Create or clear the file at startup
    f.write("Speech Log\n==========\n")

# Create a speech queue and worker thread to avoid TTS conflicts
speech_queue = queue.Queue()
tts_active = True

def tts_worker():
    """Worker thread that processes speech requests without blocking the main thread"""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        
        while tts_active:
            try:
                # Get message with 0.5 second timeout
                message = speech_queue.get(timeout=0.5)
                
                if message == "QUIT":
                    break
                    
                # Log the message
                with open(speech_log_file, "a") as f:
                    f.write(message + "\n")
                
                # Speak the message
                engine.say(message)
                engine.runAndWait()
                speech_queue.task_done()
                
            except queue.Empty:
                # No message to process, continue waiting
                continue
            except Exception as e:
                print(f"TTS Error: {e}")
                # Continue processing messages even if one fails
                speech_queue.task_done()
                
    except Exception as e:
        print(f"Could not initialize TTS engine: {e}")
        # Still process messages to log them even without speech
        while tts_active:
            try:
                message = speech_queue.get(timeout=0.5)
                if message == "QUIT":
                    break
                with open(speech_log_file, "a") as f:
                    f.write(f"[NO SPEECH] {message}\n")
                print(f"TTS would say: {message}")
                speech_queue.task_done()
            except queue.Empty:
                continue

# Start TTS worker thread
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

def log_and_speak(text):
    """Add text to speech queue without blocking"""
    speech_queue.put(text)

def give_distance_feedback(objects_with_distances):
    """Queue distance feedback without blocking"""
    if not objects_with_distances:
        return
    for obj_type, distance in objects_with_distances:
        distance_meters = round(distance, 2)
        if distance_meters <= 30:
            log_and_speak(f"{obj_type} is {distance_meters} meters ahead.")

def check_lighting_condition(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return "poor" if avg_brightness < 50 else "good"

# Load YOLOv8 and MiDaS models
print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")
print("Loading depth estimation model...")
feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

# Open camera or fallback video
print("Opening video source...")
cap = cv2.VideoCapture("C:\\Users\\BIT\\OneDrive\\Desktop\\v_prof.mp4")

if not cap.isOpened():
    print("Error: Could not open video file. Please check the path.")
    # Clean up TTS thread before exiting
    tts_active = False
    speech_queue.put("QUIT")
    tts_thread.join(timeout=1.0)
    exit()

frame_count = 0
previous_positions = {}
previous_depths = {}

def is_object_moving(cls, center, threshold=20):
    if cls not in previous_positions:
        previous_positions[cls] = center
        return False
    prev_center = previous_positions[cls]
    dist = np.linalg.norm(np.array(center) - np.array(prev_center))
    previous_positions[cls] = center
    return dist > threshold

# --- Frame processing loop ---
print("Starting frame processing...")
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or failed to grab frame.")
            break

        frame_count += 1
        print(f"\n{'='*40}\n Processing Frame {frame_count}\n{'='*40}")

        try:
            lighting = check_lighting_condition(frame)
            print(f" Lighting condition: {lighting}")
            
            if lighting == "poor":
                log_and_speak("Warning, poor lighting conditions detected.")

            results = model(frame)
            boxes = results[0].boxes
            annotated_frame = results[0].plot() if boxes else frame.copy()

            # Save YOLO prediction labels
            img_h, img_w = frame.shape[:2]
            pred_file = os.path.join(predictions_dir, f"frame_{frame_count:04d}.txt")
            with open(pred_file, "w") as f:
                if boxes:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        x1, y1, x2, y2 = box.xyxy[0]
                        cx = ((x1 + x2) / 2) / img_w
                        cy = ((y1 + y2) / 2) / img_h
                        w = (x2 - x1) / img_w
                        h = (y2 - y1) / img_h
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            # Depth estimation
            inputs = feature_extractor(images=frame, return_tensors="pt")
            with torch.no_grad():
                depth_map = depth_model(**inputs).predicted_depth.squeeze().numpy()

            objects_with_distances = []

            if not boxes:
                print("No objects detected.")
            else:
                print(" Detected Objects:")
                for result in results:
                    for box in result.boxes:
                        cls = model.names[int(box.cls[0])]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                        center_x = max(0, min(center_x, depth_map.shape[1] - 1))
                        center_y = max(0, min(center_y, depth_map.shape[0] - 1))

                        is_moving = is_object_moving(cls, (center_x, center_y))

                        try:
                            current_depth = float(depth_map[center_y, center_x])
                            direction = "static"

                            if cls == "person":
                                if cls in previous_depths:
                                    prev_depth = previous_depths[cls]
                                    delta = current_depth - prev_depth
                                    if abs(delta) > 0.2:
                                        direction = "toward camera" if delta < 0 else "away from camera"
                                previous_depths[cls] = current_depth
                            else:
                                direction = "N/A"

                            movement_status = "moving" if is_moving else "static"
                            print(f" - {cls:<15} | {movement_status:<7} | {direction:<17} | {current_depth:.2f} meters")

                            if direction in ["toward camera", "away from camera"]:
                                log_and_speak(f"{cls} is {movement_status}, {direction}, at {round(current_depth, 1)} meters")
                            else:
                                log_and_speak(f"{cls} is {movement_status} at {round(current_depth, 1)} meters")

                            objects_with_distances.append((cls, current_depth))

                        except Exception as e:
                            print(f"Error retrieving depth for {cls}: {e}")

            # Save visual outputs
            cv2.imwrite(os.path.join(output_dir, f'frame_{frame_count:04d}.jpg'), frame)
            cv2.imwrite(os.path.join(output_dir, f'detected_{frame_count:04d}.jpg'), annotated_frame)

            depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(output_dir, f'depth_{frame_count:04d}.jpg'), depth_vis)

            print("💾 Saved outputs for this frame.")

        except Exception as e:
            print(f"❌ An error occurred in Frame {frame_count}: {e}")
            traceback.print_exc()
            
        # Optional: Add a small delay to reduce CPU usage
        time.sleep(0.01)

finally:
    # Clean up resources
    cap.release()
    
    # Stop the TTS thread gracefully
    print("Stopping TTS system...")
    tts_active = False
    speech_queue.put("QUIT")
    tts_thread.join(timeout=1.0)
    
    print(f"\n✅ Finished. Total frames processed: {frame_count}. Check the '{output_dir}' and '{predictions_dir}' directories for outputs.")