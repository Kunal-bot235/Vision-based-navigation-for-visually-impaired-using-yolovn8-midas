
# Vision-Based Navigation System for the Visually Impaired

This project implements an advanced **vision-based navigation system** using **YOLOv8** for object detection and **MiDaS (DPT)** for depth estimation, aimed at assisting visually impaired individuals with real-time feedback on their surroundings.

---

## 🚀 Features

- 🎯 **Object Detection**: Utilizes YOLOv8 to identify and track objects in real-time video.
- 🌊 **Depth Estimation**: Integrates Intel's DPT (MiDaS) model for estimating the distance to detected objects.
- 🗣️ **Auditory Feedback**: Provides real-time voice guidance using `pyttsx3`.
- 🔄 **Movement & Direction Analysis**: Detects object movement and direction (e.g., toward or away from camera).
- ⚠️ **Lighting Condition Check**: Warns about poor lighting conditions to maintain detection accuracy.
- 📸 **Frame Logging**: Saves output images, depth maps, and detection annotations for each frame.

---

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Libraries & Frameworks:**
  - OpenCV
  - NumPy
  - PyTorch
  - Transformers (`DPTForDepthEstimation`, `DPTImageProcessor`)
  - Ultralytics YOLOv8
  - pyttsx3 (Text-to-Speech)
  - threading, queue (for concurrent TTS handling)

---

## 📁 Project Structure

```
.
├── vision_navigation_output/     # Stores processed frames and depth visualizations
├── predictions_txt/              # YOLO bounding box predictions in txt format
├── main.py                       # Core application script (shown above)
├── README.md                     # Project overview and usage instructions
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Kunal-bot235/Acad-Project-work-Khushi976-Vision-based-navigation.git
   cd Acad-Project-work-Khushi976-Vision-based-navigation
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv8 Model:**
   The script loads the lightweight YOLOv8n model:
   ```python
   model = YOLO("yolov8n.pt")
   ```
   You can replace `"yolov8n.pt"` with another YOLOv8 variant for improved accuracy.

4. **Run the program:**
   Update the `cv2.VideoCapture()` path in `main.py` to your desired input source (camera or video), then:
   ```bash
   python main.py
   ```

---

## 🎯 How It Works

- The video stream is processed frame-by-frame.
- Objects are detected using YOLOv8 and their positions are tracked.
- The MiDaS model estimates depth at each object's center.
- Directional movement (static, moving toward or away) is calculated.
- Real-time voice feedback is given using `pyttsx3`.
- Outputs including annotated frames and depth maps are saved.

---

## 🗃 Output Files

- **`vision_navigation_output/`**
  - `frame_####.jpg`: Original video frames.
  - `detected_####.jpg`: Frames with YOLOv8 bounding boxes.
  - `depth_####.jpg`: Colorized depth maps.

- **`predictions_txt/`**
  - `frame_####.txt`: YOLO formatted object detection outputs.

- **`speech_output.txt`**
  - Logged messages spoken via TTS.

---

## 📊 Future Improvements

- Use a more compact or mobile-friendly TTS engine.
- Deploy on Raspberry Pi with external camera.
- Add GPS and path guidance using maps.

---

## 🧑‍💻 Contributors

- **Kunal Kumar Singh** – [GitHub](https://github.com/Kunal-bot235)
- **Khushi** – [GitHub](https://github.com/Khushi976)

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- Ultralytics for YOLOv8
- Intel & Hugging Face for MiDaS depth model
- Python open-source community
