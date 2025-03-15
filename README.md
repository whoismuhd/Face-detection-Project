# Face Detection Project

A simple face detection application using OpenCV that can detect faces in images and videos.

## Features

- Detect faces in static images
- Real-time face detection from webcam feed or video files
- Count faces in the frame
- Save images/frames with detected faces
- Command-line interface for easy usage

## Project Structure

```
face-detection-project/
│
├── face_detector.py         # Main script with face detection functions
├── requirements.txt         # Project dependencies
├── .gitignore               # Git ignore file
├── README.md                # Project documentation
│
└── examples/                # Example images for testing
    ├── example1.jpg
    └── example2.jpg
```

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/face-detection-project.git
   cd face-detection-project
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Detect faces in an image:

```bash
python face_detector.py --image examples/example1.jpg
```

### Detect faces using webcam:

```bash
python face_detector.py
```

### Detect faces in a video file:

```bash
python face_detector.py --video path/to/video.mp4
```

### Save detected faces:

```bash
python face_detector.py --image examples/example1.jpg --output output_images --save
```

### Additional options:

- `--no-display`: Don't draw rectangles around detected faces
- `--output`: Specify directory to save output images
- `--save`: Save images/frames with detected faces

## Examples

Running face detection on an image:
```bash
python face_detector.py --image examples/example1.jpg --save --output results
```

Starting webcam face detection:
```bash
python face_detector.py --save --output webcam_captures
```

## Controls

When using video or webcam mode:
- Press 'q' to quit the application

## Dependencies

- OpenCV
- NumPy

## License

This project is licensed under the MIT License - see the LICENSE file for details.