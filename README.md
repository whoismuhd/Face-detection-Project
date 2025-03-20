# Face Detection Project

A powerful face detection application that can detect faces and emotions in real-time using your webcam or video files.

## Features

* Real-time face detection from webcam or video files
* Emotion detection (happy, sad, angry, neutral, fear, surprise, disgust)
* Facial landmarks detection (eyes, nose, mouth)
* Face confidence score display
* FPS counter for performance monitoring
* Save detected frames with faces
* Simple command-line interface

## Project Structure

```
face-detection-project/
│
├── face_detector.py         # Main script with face and emotion detection
├── requirements.txt         # Project dependencies
├── .gitignore              # Git ignore file
└── README.md               # Project documentation
```

## Installation

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

## Usage

### Using Webcam (Default):
```bash
python face_detector.py
```

### Using Video File:
```bash
python face_detector.py --video path/to/video.mp4
```

### Save Detected Frames:
```bash
python face_detector.py --output output_folder
```

## What You'll See

When running the application, you'll see:
* Green boxes around detected faces
* Red dots showing facial landmarks (eyes, nose, mouth)
* Blue text showing the detected emotion below each face
* FPS counter in the top-left corner
* Number of faces detected in the frame

## Tips for Best Results

1. For better face detection:
   * Ensure good lighting
   * Face the camera directly
   * Keep your face within a reasonable distance

2. For better emotion detection:
   * Make sure your face is well-lit
   * Look directly at the camera
   * Stay relatively close to the camera

## Controls

* Press 'q' to quit the application

## Dependencies

* OpenCV - For image and video processing
* MediaPipe - For accurate face detection
* DeepFace - For emotion detection
* NumPy - For numerical operations
* TensorFlow - Required for DeepFace

## License

This project is licensed under the MIT License.

## About

This project demonstrates real-time face and emotion detection using modern computer vision techniques. It's perfect for learning about computer vision, machine learning, and Python programming.