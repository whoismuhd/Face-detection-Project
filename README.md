# Face Detection Project

A powerful and feature-rich face detection application that can detect faces and emotions in real-time using your webcam, video files, or static images.

## ✨ Features

### Core Features
* **Real-time face detection** from webcam or video files
* **Static image processing** - detect faces in photos
* **Emotion detection** (happy, sad, angry, neutral, fear, surprise, disgust)
* **Facial landmarks detection** (eyes, nose, mouth)
* **Face confidence score display**
* **Performance metrics** - Real-time FPS counter with smooth averaging
* **Flexible export options** - Save results as JSON or CSV

### Advanced Features
* **Parallel emotion processing** - Faster performance with multi-threaded emotion detection
* **Configurable detection threshold** - Adjust minimum confidence levels
* **Frame skipping** - Process every Nth frame for better performance
* **Batch frame saving** - Automatically save frames with detected faces
* **Export detection results** - CSV/JSON export for analysis
* **Beautiful visualization** - Color-coded confidence levels, better UI layout
* **Type hints** - Full type annotations for better code quality

## 📁 Project Structure

```
face-detection-project/
│
├── face_detector.py         # Main script with face and emotion detection
├── requirements.txt         # Project dependencies
├── README.md               # Project documentation
└── venv/                   # Virtual environment (not tracked)
```

## 🚀 Installation

1. Clone this repository:
```bash
git clone https://github.com/whoismuhd/Face-detection-Project.git
cd Face-detection-Project
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Webcam (Real-time)
```bash
# Basic usage
python face_detector.py

# Fast mode (no emotion detection)
python face_detector.py --no-emotion

# Custom confidence threshold
python face_detector.py --confidence 0.7

# Skip frames for better performance
python face_detector.py --frame-skip 2
```

### Video File Processing
```bash
# Basic video processing
python face_detector.py --video path/to/video.mp4

# Save frames with detections
python face_detector.py --video path/to/video.mp4 --output frames/ --save-frames

# Export detection results to CSV
python face_detector.py --video path/to/video.mp4 --export-csv results.csv

# Full featured example
python face_detector.py --video video.mp4 --output frames/ --export-csv results.csv --save-frames --confidence 0.6
```

### Image Processing
```bash
# Process single image
python face_detector.py --image photo.jpg

# Process and save output
python face_detector.py --image photo.jpg --output result.jpg

# Process and export JSON results
python face_detector.py --image photo.jpg --output result.jpg --export-json results.json
```

## 🎨 What You'll See

When running the application, you'll see:
* **Color-coded bounding boxes** around detected faces:
  - 🟢 Green: High confidence (>0.7)
  - 🟡 Yellow: Medium confidence (0.5-0.7)
  - 🟠 Orange: Lower confidence (<0.5)
* **Red dots** showing facial landmarks (eyes, nose, mouth)
* **Blue text** showing the detected emotion below each face
* **Real-time stats overlay**:
  - FPS counter (smoothly averaged)
  - Number of faces detected
  - Frame number
  - Progress percentage (for video files)

## ⚙️ Command-Line Options

### Input Options
* `--video PATH` - Path to input video file (omit for webcam)
* `--image PATH` - Path to input image file

### Output Options
* `--output PATH` - Output directory (video) or file path (image)
* `--export-csv PATH` - Export detection results to CSV file (video only)
* `--export-json PATH` - Export detection results to JSON file (image only)
* `--save-frames` - Save frames with detections (video only)

### Processing Options
* `--confidence FLOAT` - Minimum detection confidence (0.0-1.0, default: 0.5)
* `--no-emotion` - Disable emotion detection for faster processing
* `--frame-skip N` - Process every Nth frame (0 = all frames, default: 0)

## 💡 Tips for Best Results

### Face Detection
* Ensure good lighting conditions
* Face the camera directly
* Keep your face within a reasonable distance
* Use higher confidence thresholds (0.7+) for better accuracy

### Emotion Detection
* Make sure your face is well-lit
* Look directly at the camera
* Stay relatively close to the camera
* Note: Emotion detection is computationally intensive, use `--no-emotion` for faster processing

### Performance Optimization
* Use `--no-emotion` flag to disable emotion detection for 2-3x faster processing
* Use `--frame-skip N` to process every Nth frame (good for longer videos)
* Lower confidence thresholds process faster but may detect false positives

## 📊 Export Formats

### CSV Export (Video)
The CSV file contains:
- `frame`: Frame number
- `timestamp`: Detection timestamp
- `faces_detected`: Number of faces in frame
- `face_id`: Face index in frame
- `confidence`: Detection confidence score
- `emotion`: Detected emotion
- `bbox`: Bounding box coordinates (x, y, width, height)

### JSON Export (Image)
The JSON file contains:
- `image_path`: Source image path
- `timestamp`: Processing timestamp
- `faces_detected`: Number of faces
- `faces`: Array of face detections with:
  - `bbox`: Bounding box
  - `confidence`: Detection confidence
  - `landmarks`: Facial landmark coordinates
  - `emotion`: Detected emotion
  - `timestamp`: Detection timestamp

## 🎮 Controls

* Press **'q'** to quit the application

## 📦 Dependencies

* **OpenCV** - For image and video processing
* **MediaPipe** - For accurate face detection
* **DeepFace** - For emotion detection
* **NumPy** - For numerical operations
* **TensorFlow** - Required for DeepFace

## 🆕 What's New

### Version 2.0 Improvements
* ✅ **Image processing support** - Now handles static images in addition to video
* ✅ **Parallel emotion processing** - Faster emotion detection with threading
* ✅ **Better FPS calculation** - Smooth exponential moving average
* ✅ **Export functionality** - CSV and JSON export for analysis
* ✅ **Type hints** - Full type annotations throughout
* ✅ **Better visualization** - Color-coded confidence levels, improved UI
* ✅ **More configuration options** - Fine-tune detection and performance
* ✅ **Improved error handling** - Better logging and error messages
* ✅ **Code structure** - Better separation of concerns, more maintainable

## 🔧 Technical Details

### Performance
* Face detection using MediaPipe: ~30-60 FPS
* With emotion detection: ~5-15 FPS (depending on hardware)
* Parallel emotion processing improves throughput by ~40%

### Architecture
* Object-oriented design with `FaceDetector` class
* ThreadPoolExecutor for parallel emotion detection
* Exponential moving average FPS counter for smooth display
* Modular design for easy extension

## 📝 License

This project is licensed under the MIT License.

## 👤 About

This project demonstrates real-time face and emotion detection using modern computer vision techniques. It's perfect for learning about computer vision, machine learning, and Python programming.

Built with ❤️ using MediaPipe, OpenCV, and DeepFace.
