# Face Detection Project

A powerful and feature-rich face detection application that can detect faces, recognize people, analyze emotions, estimate age/gender, and much more in real-time using your webcam, video files, or static images.

## ✨ Features

### Core Features
* **Real-time face detection** from webcam or video files
* **Static image processing** - detect faces in photos
* **Batch image processing** - process entire directories of images
* **Face recognition** - recognize known faces using a database
* **Emotion detection** (happy, sad, angry, neutral, fear, surprise, disgust)
* **Age and gender detection** - estimate age and gender of detected faces
* **Face quality scoring** - assess face quality (brightness, sharpness, size)
* **Facial landmarks detection** (eyes, nose, mouth)
* **Face confidence score display**
* **Performance metrics** - Real-time FPS counter with smooth averaging
* **Flexible export options** - Save results as JSON or CSV
* **Video export** - Save processed video with detections
* **Face crop export** - Export individual face crops

### Advanced Features
* **Face database management** - Store and manage known faces
* **Parallel processing** - Faster performance with multi-threaded analysis
* **Statistics tracking** - Track emotions, age, gender, recognized faces across video
* **Face blurring** - Privacy protection by blurring detected faces
* **Quality filtering** - Filter out low-quality faces automatically
* **Configuration file support** - Use JSON config files for settings
* **Configurable detection threshold** - Adjust minimum confidence levels
* **Frame skipping** - Process every Nth frame for better performance
* **Batch frame saving** - Automatically save frames with detected faces
* **Multiple visualization modes** - Minimal, detailed, or simple views
* **Export detection results** - CSV/JSON export for analysis
* **Beautiful visualization** - Color-coded confidence levels, better UI layout
* **Type hints** - Full type annotations for better code quality

## 📁 Project Structure

```
face-detection-project/
│
├── face_detector.py         # Main script with face detection
├── requirements.txt         # Project dependencies
├── config.example.json      # Example configuration file
├── face_database.json       # Face recognition database (auto-created)
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

# With age and gender detection
python face_detector.py --age-gender

# Face recognition (requires face database)
python face_detector.py --recognize

# Blur faces for privacy
python face_detector.py --blur-faces

# Quality filtering (only process high-quality faces)
python face_detector.py --min-quality 0.5

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

# Export processed video (with all detections drawn)
python face_detector.py --video input.mp4 --export-video output.mp4

# Export face crops
python face_detector.py --video video.mp4 --export-crops crops/

# Full featured example
python face_detector.py --video video.mp4 --output frames/ --export-csv results.csv --export-video processed.mp4 --age-gender --recognize --save-frames
```

### Image Processing
```bash
# Process single image
python face_detector.py --image photo.jpg

# Process and save output
python face_detector.py --image photo.jpg --output result.jpg

# Process with age/gender detection
python face_detector.py --image photo.jpg --output result.jpg --age-gender

# Process and export JSON results
python face_detector.py --image photo.jpg --output result.jpg --export-json results.json

# Export face crops
python face_detector.py --image photo.jpg --export-crops faces/

# Blur faces in image
python face_detector.py --image photo.jpg --output blurred.jpg --blur-faces
```

### Batch Image Processing
```bash
# Process all images in a directory
python face_detector.py --batch images/ --output processed/

# Batch process with age/gender detection
python face_detector.py --batch images/ --output processed/ --age-gender

# Export batch results to JSON
python face_detector.py --batch images/ --output processed/ --export-json batch_results.json

# Batch process with face recognition
python face_detector.py --batch images/ --output processed/ --recognize
```

### Face Recognition
```bash
# Add a face to the database
python face_detector.py --add-face "John Doe" --image john.jpg

# Use face recognition on video
python face_detector.py --video video.mp4 --recognize

# Use face recognition on images
python face_detector.py --image photo.jpg --recognize

# Custom database location
python face_detector.py --recognize --face-db my_database.json
```

### Configuration File
```bash
# Use configuration file (see config.example.json)
python face_detector.py --config config.json --video video.mp4

# Configuration file options:
# - confidence
# - enable_emotion
# - enable_age_gender
# - blur_faces
# - enable_recognition
# - min_quality
# - frame_skip
# - face_db
```

## 🎨 What You'll See

When running the application, you'll see:
* **Color-coded bounding boxes** around detected faces:
  - 🟢 Green: High confidence (>0.7)
  - 🟡 Yellow: Medium confidence (0.5-0.7)
  - 🟠 Orange: Lower confidence (<0.5)
* **Red dots** showing facial landmarks (eyes, nose, mouth)
* **Information labels** showing:
  - Recognized name (if face recognition enabled)
  - Detected emotion (if enabled)
  - Estimated age (if enabled)
  - Gender (if enabled)
  - Quality score (if detailed mode)
* **Real-time stats overlay**:
  - FPS counter (smoothly averaged)
  - Number of faces detected
  - Frame number
  - Total faces detected so far
  - Progress percentage (for video files)

## ⚙️ Command-Line Options

### Input Options
* `--video PATH` - Path to input video file (omit for webcam)
* `--image PATH` - Path to input image file
* `--batch PATH` - Directory containing images for batch processing

### Output Options
* `--output PATH` - Output directory (video/batch) or file path (image)
* `--export-csv PATH` - Export detection results to CSV file (video only)
* `--export-json PATH` - Export detection results to JSON file (image/batch)
* `--export-video PATH` - Export processed video to file (video only)
* `--export-crops PATH` - Directory to export individual face crops
* `--save-frames` - Save frames with detections (video only)

### Processing Options
* `--confidence FLOAT` - Minimum detection confidence (0.0-1.0, default: 0.5)
* `--no-emotion` - Disable emotion detection for faster processing
* `--age-gender` - Enable age and gender detection
* `--blur-faces` - Blur detected faces for privacy
* `--frame-skip N` - Process every Nth frame (0 = all frames, default: 0)
* `--min-quality FLOAT` - Minimum face quality score (0.0-1.0, default: 0.0)

### Face Recognition Options
* `--recognize` - Enable face recognition
* `--add-face NAME` - Add face to database (requires --image and name)
* `--face-db PATH` - Path to face database file (default: face_database.json)

### Configuration
* `--config PATH` - Path to JSON configuration file

## 💡 Tips for Best Results

### Face Detection
* Ensure good lighting conditions
* Face the camera directly
* Keep your face within a reasonable distance
* Use higher confidence thresholds (0.7+) for better accuracy

### Face Recognition
* Add multiple images of the same person for better recognition
* Use clear, frontal face photos
* Ensure good lighting in reference images
* Recognition works best with faces looking directly at camera

### Emotion Detection
* Make sure your face is well-lit
* Look directly at the camera
* Stay relatively close to the camera
* Note: Emotion detection is computationally intensive, use `--no-emotion` for faster processing

### Age/Gender Detection
* Works best with clear, frontal face views
* Requires good lighting conditions
* Slightly slower than emotion-only detection

### Quality Filtering
* Use `--min-quality 0.5` to filter out low-quality faces
* Quality score considers brightness, sharpness, and face size
* Helps reduce false positives and improve accuracy

### Performance Optimization
* Use `--no-emotion` flag to disable emotion detection for 2-3x faster processing
* Use `--frame-skip N` to process every Nth frame (good for longer videos)
* Lower confidence thresholds process faster but may detect false positives
* Age/gender detection adds processing time; use only when needed
* Face recognition adds moderate processing overhead

## 📊 Export Formats

### CSV Export (Video)
The CSV file contains:
- `frame`: Frame number
- `timestamp`: Detection timestamp
- `faces_detected`: Number of faces in frame
- `face_id`: Face index in frame
- `confidence`: Detection confidence score
- `emotion`: Detected emotion
- `age`: Estimated age (if enabled)
- `gender`: Estimated gender (if enabled)
- `recognized_as`: Recognized name (if enabled)
- `quality`: Face quality score
- `bbox`: Bounding box coordinates (x, y, width, height)

### JSON Export (Image/Batch)
The JSON file contains:
- `image_path`: Source image path
- `timestamp`: Processing timestamp
- `faces_detected`: Number of faces
- `faces`: Array of face detections with:
  - `bbox`: Bounding box
  - `confidence`: Detection confidence
  - `quality`: Quality metrics (score, brightness, sharpness, size)
  - `landmarks`: Facial landmark coordinates
  - `emotion`: Detected emotion (if enabled)
  - `age`: Estimated age (if enabled)
  - `gender`: Estimated gender (if enabled)
  - `recognized_as`: Recognized name (if enabled)
  - `similarity`: Recognition similarity score (if enabled)
  - `crop_path`: Path to exported crop (if enabled)
  - `timestamp`: Detection timestamp

### Face Crops
When using `--export-crops`, individual face images are saved as:
- `face_XXX_TIMESTAMP.jpg` - Where XXX is face index and TIMESTAMP is detection time
- Useful for creating face datasets or further analysis

## 📹 Video Export

The `--export-video` option saves a processed video with all detections drawn on frames. This is useful for:
* Creating demo videos
* Sharing results with others
* Reviewing detections frame-by-frame

The exported video maintains the original frame rate and resolution.

## 🔒 Privacy Features

### Face Blurring
Use the `--blur-faces` option to automatically blur detected faces in output images/videos. This is useful for:
* Privacy protection in public datasets
* Anonymizing faces in videos/images
* Creating privacy-compliant content

### Quality Filtering
Use `--min-quality` to automatically filter out low-quality face detections, which can help:
* Reduce false positives
* Improve accuracy of emotion/age/gender detection
* Focus processing on high-quality faces

## 🎭 Face Recognition

### Building a Face Database

1. **Add faces to database:**
```bash
python face_detector.py --add-face "John Doe" --image john1.jpg
python face_detector.py --add-face "Jane Smith" --image jane1.jpg
python face_detector.py --add-face "John Doe" --image john2.jpg  # Add multiple photos for better recognition
```

2. **Use recognition:**
```bash
python face_detector.py --video video.mp4 --recognize
python face_detector.py --image photo.jpg --recognize
```

3. **Database format:**
The face database is stored as JSON with face embeddings. Each entry contains:
- Face name/label
- Face embedding vector
- Timestamp when added

### Recognition Tips
* Add 2-5 clear photos per person for best results
* Use frontal face photos with good lighting
* Recognition similarity threshold: 0.6 (configurable in code)
* Database is saved automatically after adding faces

## 📈 Statistics

When processing videos, the application tracks and displays:
* Total frames processed
* Frames with detected faces
* Total faces detected
* Average faces per frame
* Emotion distribution (if enabled)
* Gender distribution (if enabled)
* Recognized faces count (if enabled)
* Average age (if enabled)

Statistics are printed at the end of video processing.

## ⚙️ Configuration File

Create a `config.json` file to set default options:

```json
{
  "confidence": 0.6,
  "enable_emotion": true,
  "enable_age_gender": false,
  "blur_faces": false,
  "enable_recognition": false,
  "min_quality": 0.3,
  "frame_skip": 0,
  "face_db": "face_database.json"
}
```

Use it with:
```bash
python face_detector.py --config config.json --video video.mp4
```

Command-line arguments override config file values.

## 🎮 Controls

* Press **'q'** to quit the application

## 📦 Dependencies

* **OpenCV** - For image and video processing
* **MediaPipe** - For accurate face detection
* **DeepFace** - For emotion, age, gender detection, and face recognition
* **NumPy** - For numerical operations
* **TensorFlow** - Required for DeepFace

## 🆕 What's New

### Version 4.0 Improvements (Latest)
* ✅ **Face recognition** - Recognize known faces using a database
* ✅ **Face database management** - Store and manage face embeddings
* ✅ **Face quality scoring** - Assess brightness, sharpness, and size
* ✅ **Quality filtering** - Filter out low-quality faces automatically
* ✅ **Face crop export** - Export individual face crops for analysis
* ✅ **Configuration file support** - Use JSON configs for settings
* ✅ **Enhanced statistics** - Track recognized faces in video
* ✅ **Better visualization** - Show recognition results and quality scores
* ✅ **Improved performance** - Better parallel processing

### Version 3.0 Improvements
* ✅ **Age and gender detection** - Estimate age and gender of detected faces
* ✅ **Batch image processing** - Process entire directories of images
* ✅ **Video export** - Save processed videos with all detections
* ✅ **Statistics tracking** - Track emotions, age, gender distributions
* ✅ **Face blurring** - Privacy protection feature
* ✅ **Better performance** - Improved parallel processing
* ✅ **Enhanced visualization** - More informative display with age/gender
* ✅ **Comprehensive statistics** - Detailed summary at end of processing

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
* With age/gender detection: ~3-10 FPS (depending on hardware)
* With face recognition: ~2-8 FPS (depending on database size)
* Parallel processing improves throughput by ~40%

### Architecture
* Object-oriented design with `FaceDetector` class
* ThreadPoolExecutor for parallel face analysis
* Exponential moving average FPS counter for smooth display
* Statistics tracking across video processing
* Face database with JSON storage
* Quality scoring using multiple metrics
* Modular design for easy extension

### Face Recognition
* Uses VGG-Face model from DeepFace
* Euclidean distance for similarity matching
* Configurable similarity threshold
* Supports multiple faces per person for better accuracy

## 📝 License

This project is licensed under the MIT License.

## 👤 About

This project demonstrates real-time face detection, recognition, emotion analysis, age/gender estimation, and more using modern computer vision techniques. It's perfect for learning about computer vision, machine learning, and Python programming.

Built with ❤️ using MediaPipe, OpenCV, and DeepFace.
