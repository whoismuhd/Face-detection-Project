import cv2
import numpy as np
import argparse
import os
from datetime import datetime

def detect_faces(image_path, output_dir=None, draw=True, save=False):
    """
    Detect faces in an image using OpenCV's Haar Cascade classifier.
    
    Args:
        image_path (str): Path to the input image
        output_dir (str, optional): Directory to save output images
        draw (bool): Whether to draw rectangles around detected faces
        save (bool): Whether to save the output image
        
    Returns:
        tuple: (image with faces marked, list of face coordinates)
    """
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None, []
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Draw rectangles around the faces if requested
    if draw:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
    # Save the output image if requested
    if save and output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filename = os.path.basename(image_path)
        base_name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{base_name}_detected{ext}")
        cv2.imwrite(output_path, img)
        print(f"Saved output image to {output_path}")
    
    return img, faces

def detect_faces_in_video(video_path=0, output_dir=None, save=False):
    """
    Detect faces in video stream or webcam.
    
    Args:
        video_path (int or str): 0 for webcam, or path to video file
        output_dir (str, optional): Directory to save frames with faces
        save (bool): Whether to save frames with detected faces
    """
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Create output directory if needed
    if save and output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    while True:
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display face count
        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Save frame if faces detected and save is enabled
        if save and output_dir and len(faces) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_path = os.path.join(output_dir, f"face_detected_{timestamp}.jpg")
            cv2.imwrite(output_path, frame)
        
        # Display the frame
        cv2.imshow('Face Detection', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Face Detection using OpenCV')
    parser.add_argument('--image', help='Path to input image')
    parser.add_argument('--video', help='Path to input video (omit for webcam)')
    parser.add_argument('--output', help='Directory to save output images/frames')
    parser.add_argument('--save', action='store_true', help='Save output images/frames')
    parser.add_argument('--no-display', action='store_true', help='Do not display rectangles around faces')
    
    args = parser.parse_args()
    
    if args.image:
        # Process a single image
        img, faces = detect_faces(
            args.image, 
            args.output, 
            not args.no_display, 
            args.save
        )
        
        if img is not None:
            print(f"Detected {len(faces)} faces")
            cv2.imshow('Face Detection', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        # Process video or webcam feed
        video_source = 0  # Default to webcam
        if args.video:
            video_source = args.video
        
        detect_faces_in_video(video_source, args.output, args.save)

if __name__ == "__main__":
    main()