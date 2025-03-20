import cv2
import mediapipe as mp
import argparse
import os
import logging
from datetime import datetime
from deepface import DeepFace

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        """
        Initialize the face detector with MediaPipe.
        
        Args:
            min_detection_confidence (float): Minimum confidence for face detection
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )
        
    def process_image(self, image):
        """
        Process a single image and return face detections with emotion analysis.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            tuple: (processed image, list of face detections)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)
        
        faces_info = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                
                # Convert relative coordinates to absolute
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                face_info = {
                    'bbox': (x, y, width, height),
                    'confidence': detection.score[0],
                    'landmarks': []
                }
                
                # Add facial landmarks
                for keypoint in detection.location_data.relative_keypoints:
                    px = int(keypoint.x * w)
                    py = int(keypoint.y * h)
                    face_info['landmarks'].append((px, py))
                
                # Add emotion detection
                try:
                    face_img = image[y:y+height, x:x+width]
                    if face_img.size > 0:
                        emotion = DeepFace.analyze(
                            face_img, 
                            actions=['emotion'],
                            enforce_detection=False
                        )
                        face_info['emotion'] = emotion[0]['dominant_emotion']
                except Exception as e:
                    logger.warning(f"Emotion detection failed: {str(e)}")
                    face_info['emotion'] = 'unknown'
                
                faces_info.append(face_info)
                
                # Draw bounding box
                cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
                
                # Draw confidence score
                conf_text = f"{detection.score[0]:.2f}"
                cv2.putText(image, conf_text, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw landmarks
                for px, py in face_info['landmarks']:
                    cv2.circle(image, (px, py), 2, (0, 0, 255), -1)
                
                # Draw emotion
                if 'emotion' in face_info:
                    emotion_text = face_info['emotion']
                    cv2.putText(image, emotion_text, (x, y + height + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return image, faces_info

def process_video(video_path=0, output_dir=None):
    """
    Process video stream with face detection and emotion analysis.
    
    Args:
        video_path: Video source (0 for webcam, or video file path)
        output_dir (str, optional): Directory to save frames
    """
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Could not open video source")
        return

    # Initialize face detector
    detector = FaceDetector()
    
    fps = 0
    frame_count = 0
    start_time = datetime.now()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, faces = detector.process_image(frame)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:
                time_diff = (datetime.now() - start_time).total_seconds()
                fps = frame_count / time_diff
                
            # Display FPS and face count
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Faces: {len(faces)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save frame if requested
            if output_dir and len(faces) > 0:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(
                    output_dir,
                    f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                )
                cv2.imwrite(output_path, processed_frame)
            
            # Display the frame
            cv2.imshow('Face Detection', processed_frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Face Detection System with Emotion Analysis')
    parser.add_argument('--video', help='Path to input video (omit for webcam)')
    parser.add_argument('--output', help='Directory to save output frames')
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting video processing with emotion detection...")
        video_source = args.video if args.video else 0
        process_video(video_source, args.output)
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()