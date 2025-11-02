import cv2
import mediapipe as mp
import argparse
import os
import json
import csv
import logging
import time
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from deepface import DeepFace

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FaceDetector:
    """Face detection and emotion analysis using MediaPipe and DeepFace."""
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        enable_emotion: bool = True,
        emotion_backend: str = 'opencv'
    ):
        """
        Initialize the face detector with MediaPipe.
        
        Args:
            min_detection_confidence: Minimum confidence for face detection (0.0-1.0)
            enable_emotion: Whether to enable emotion detection (slower but more features)
            emotion_backend: Backend for emotion detection ('opencv', 'tensorflow', etc.)
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )
        self.enable_emotion = enable_emotion
        self.emotion_backend = emotion_backend
        self.executor = ThreadPoolExecutor(max_workers=2) if enable_emotion else None
        
    def _detect_emotion(self, face_img: np.ndarray) -> Optional[str]:
        """
        Detect emotion in a face image.
        
        Args:
            face_img: Cropped face image
            
        Returns:
            Detected emotion or None if detection fails
        """
        if not self.enable_emotion:
            return None
            
        try:
            if face_img.size == 0:
                return None
                
            # Resize if too small for DeepFace
            if face_img.shape[0] < 48 or face_img.shape[1] < 48:
                face_img = cv2.resize(face_img, (48, 48), interpolation=cv2.INTER_CUBIC)
            
            emotion = DeepFace.analyze(
                face_img,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            return emotion[0]['dominant_emotion']
        except Exception as e:
            logger.debug(f"Emotion detection failed: {str(e)}")
            return None
    
    def process_image(
        self,
        image: np.ndarray,
        draw_results: bool = True,
        detect_emotion: Optional[bool] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process a single image and return face detections with optional emotion analysis.
        
        Args:
            image: Input image in BGR format
            draw_results: Whether to draw detection results on the image
            detect_emotion: Override emotion detection setting (None uses class default)
            
        Returns:
            Tuple of (processed image, list of face detections with info)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)
        
        faces_info = []
        if results.detections:
            h, w, _ = image.shape
            should_detect_emotion = detect_emotion if detect_emotion is not None else self.enable_emotion
            
            # Prepare face crops for parallel emotion detection
            face_futures = []
            face_info_list = []
            
            for idx, detection in enumerate(results.detections):
                bbox = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to absolute
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                width = min(w - x, int(bbox.width * w))
                height = min(h - y, int(bbox.height * h))
                
                # Ensure valid dimensions
                if width <= 0 or height <= 0:
                    continue
                
                confidence = float(detection.score[0])
                
                face_info = {
                    'bbox': (x, y, width, height),
                    'confidence': confidence,
                    'landmarks': [],
                    'timestamp': datetime.now().isoformat()
                }
                
                # Add facial landmarks
                for keypoint in detection.location_data.relative_keypoints:
                    px = int(keypoint.x * w)
                    py = int(keypoint.y * h)
                    face_info['landmarks'].append((px, py))
                
                # Start emotion detection in parallel if enabled
                if should_detect_emotion and self.executor:
                    face_img = image[y:y+height, x:x+width].copy()
                    future = self.executor.submit(self._detect_emotion, face_img)
                    face_futures.append((len(face_info_list), future))
                elif should_detect_emotion:
                    face_img = image[y:y+height, x:x+width].copy()
                    emotion = self._detect_emotion(face_img)
                    face_info['emotion'] = emotion if emotion else 'unknown'
                
                face_info_list.append(face_info)
            
            # Collect emotion results
            for face_idx, future in face_futures:
                try:
                    emotion = future.result(timeout=1.0)
                    face_info_list[face_idx]['emotion'] = emotion if emotion else 'unknown'
                except Exception as e:
                    logger.debug(f"Emotion detection timeout/failure for face {face_idx}: {str(e)}")
                    face_info_list[face_idx]['emotion'] = 'unknown'
            
            faces_info = face_info_list
            
            # Draw results if requested
            if draw_results:
                image = self._draw_detections(image, faces_info)
        
        return image, faces_info
    
    def _draw_detections(
        self,
        image: np.ndarray,
        faces_info: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Draw detection results on the image.
        
        Args:
            image: Image to draw on
            faces_info: List of face detection information
            
        Returns:
            Image with drawn detections
        """
        for face_info in faces_info:
            x, y, width, height = face_info['bbox']
            confidence = face_info['confidence']
            
            # Draw bounding box with color based on confidence
            box_color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.5 else (0, 165, 255)
            cv2.rectangle(image, (x, y), (x + width, y + height), box_color, 2)
            
            # Draw confidence score with background
            conf_text = f"{confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                image,
                (x, y - text_height - baseline - 10),
                (x + text_width, y),
                (0, 0, 0),
                -1
            )
            cv2.putText(
                image, conf_text, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            
            # Draw landmarks
            for px, py in face_info['landmarks']:
                cv2.circle(image, (px, py), 3, (0, 0, 255), -1)
            
            # Draw emotion if available
            if 'emotion' in face_info and face_info['emotion'] != 'unknown':
                emotion_text = face_info['emotion'].upper()
                (emotion_width, emotion_height), _ = cv2.getTextSize(
                    emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    image,
                    (x, y + height),
                    (x + emotion_width + 10, y + height + emotion_height + 10),
                    (255, 0, 0),
                    -1
                )
                cv2.putText(
                    image, emotion_text, (x + 5, y + height + emotion_height + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
        
        return image
    
    def __del__(self):
        """Cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=False)


class FPSCounter:
    """Simple FPS counter using exponential moving average."""
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize FPS counter.
        
        Args:
            alpha: Smoothing factor (0.0-1.0), lower = more smoothing
        """
        self.alpha = alpha
        self.fps = 0.0
        self.last_time = time.time()
        
    def update(self) -> float:
        """
        Update and return current FPS.
        
        Returns:
            Current FPS value
        """
        current_time = time.time()
        delta = current_time - self.last_time
        current_fps = 1.0 / delta if delta > 0 else 0.0
        
        # Exponential moving average
        self.fps = self.alpha * current_fps + (1 - self.alpha) * self.fps
        self.last_time = current_time
        
        return self.fps


def process_image_file(
    image_path: str,
    output_path: Optional[str] = None,
    min_confidence: float = 0.5,
    enable_emotion: bool = True,
    export_json: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a single image file.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image (None = don't save)
        min_confidence: Minimum detection confidence
        enable_emotion: Whether to detect emotions
        export_json: Path to export results as JSON (None = don't export)
        
    Returns:
        Dictionary with processing results
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    logger.info(f"Processing image: {image_path}")
    
    # Initialize detector
    detector = FaceDetector(
        min_detection_confidence=min_confidence,
        enable_emotion=enable_emotion
    )
    
    # Process image
    processed_image, faces = detector.process_image(image)
    
    results = {
        'image_path': image_path,
        'timestamp': datetime.now().isoformat(),
        'faces_detected': len(faces),
        'faces': faces
    }
    
    # Save output image
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        cv2.imwrite(output_path, processed_image)
        results['output_path'] = output_path
        logger.info(f"Saved output image: {output_path}")
    
    # Export JSON
    if export_json:
        with open(export_json, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Exported results to: {export_json}")
    
    logger.info(f"Detected {len(faces)} face(s) in image")
    return results


def process_video(
    video_path: Union[int, str] = 0,
    output_dir: Optional[str] = None,
    min_confidence: float = 0.5,
    enable_emotion: bool = True,
    frame_skip: int = 0,
    export_csv: Optional[str] = None,
    save_frames: bool = False
):
    """
    Process video stream with face detection and emotion analysis.
    
    Args:
        video_path: Video source (0 for webcam, or video file path)
        output_dir: Directory to save frames
        min_confidence: Minimum detection confidence
        enable_emotion: Whether to detect emotions (slower)
        frame_skip: Process every Nth frame (0 = process all frames, for performance)
        export_csv: Path to export detection results as CSV
        save_frames: Whether to save frames with detections
    """
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if isinstance(video_path, str):
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video properties: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")
    
    # Initialize face detector
    detector = FaceDetector(
        min_detection_confidence=min_confidence,
        enable_emotion=enable_emotion
    )
    
    # Initialize FPS counter
    fps_counter = FPSCounter()
    
    # Prepare CSV export
    csv_file = None
    csv_writer = None
    if export_csv:
        csv_file = open(export_csv, 'w', newline='')
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=['frame', 'timestamp', 'faces_detected', 'face_id', 'confidence', 'emotion', 'bbox']
        )
        csv_writer.writeheader()
    
    frame_count = 0
    detection_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(video_path, str):
                    logger.info("End of video reached")
                break
            
            # Skip frames if requested (for performance)
            if frame_skip > 0 and frame_count % (frame_skip + 1) != 0:
                frame_count += 1
                # Still display frame without processing
                if isinstance(video_path, int):
                    cv2.imshow('Face Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue
            
            # Process frame
            processed_frame, faces = detector.process_image(frame)
            
            # Update FPS
            current_fps = fps_counter.update()
            
            # Draw stats overlay on top of processed frame
            stats = [
                f"FPS: {current_fps:.1f}",
                f"Faces: {len(faces)}",
                f"Frame: {frame_count}"
            ]
            if isinstance(video_path, str):
                stats.append(f"Progress: {frame_count}/{total_frames} ({100*frame_count/total_frames:.1f}%)")
            
            # Draw semi-transparent background for stats
            overlay_height = len(stats) * 30 + 20
            overlay_width = 400
            overlay_rect = np.zeros((overlay_height, overlay_width, 3), dtype=np.uint8)
            overlay_rect.fill(0)
            
            # Blend with frame (with bounds checking)
            overlay_x, overlay_y = 10, 10
            if overlay_y + overlay_height <= height and overlay_x + overlay_width <= width:
                roi = processed_frame[overlay_y:overlay_y+overlay_height, overlay_x:overlay_x+overlay_width]
                if roi.shape[0] == overlay_height and roi.shape[1] == overlay_width:
                    blended = cv2.addWeighted(roi, 0.6, overlay_rect, 0.4, 0)
                    processed_frame[overlay_y:overlay_y+overlay_height, overlay_x:overlay_x+overlay_width] = blended
            
            # Draw text on processed frame
            y_offset = 35
            for stat in stats:
                cv2.putText(
                    processed_frame, stat, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                y_offset += 30
            
            # Export to CSV
            if csv_writer:
                timestamp = datetime.now().isoformat()
                for idx, face in enumerate(faces):
                    csv_writer.writerow({
                        'frame': frame_count,
                        'timestamp': timestamp,
                        'faces_detected': len(faces),
                        'face_id': idx,
                        'confidence': face['confidence'],
                        'emotion': face.get('emotion', 'unknown'),
                        'bbox': str(face['bbox'])
                    })
            
            # Save frame if requested
            if save_frames and output_dir and len(faces) > 0:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(
                    output_dir,
                    f"frame_{frame_count:06d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                )
                cv2.imwrite(output_path, processed_frame)
                detection_count += 1
            
            # Display the frame
            cv2.imshow('Face Detection', processed_frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User requested quit")
                break
            
            frame_count += 1
                
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    finally:
        cap.release()
        if csv_file:
            csv_file.close()
            logger.info(f"Exported detection results to: {export_csv}")
        cv2.destroyAllWindows()
        logger.info(f"Processed {frame_count} frames, saved {detection_count} frames with detections")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description='Face Detection System with Emotion Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use webcam
  python face_detector.py

  # Process video file
  python face_detector.py --video video.mp4

  # Process single image
  python face_detector.py --image photo.jpg --output output.jpg

  # Save frames and export CSV
  python face_detector.py --video video.mp4 --output frames/ --export-csv results.csv

  # Fast mode (no emotion detection, skip frames)
  python face_detector.py --no-emotion --frame-skip 2
        """
    )
    
    # Input source
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--video', type=str, help='Path to input video file (omit for webcam)')
    input_group.add_argument('--image', type=str, help='Path to input image file')
    
    # Output options
    parser.add_argument('--output', type=str, help='Output directory (video) or file path (image)')
    parser.add_argument('--export-csv', type=str, help='Export detection results to CSV file')
    parser.add_argument('--export-json', type=str, help='Export detection results to JSON file (image only)')
    parser.add_argument('--save-frames', action='store_true', help='Save frames with detections (video only)')
    
    # Processing options
    parser.add_argument('--confidence', type=float, default=0.5, help='Minimum detection confidence (0.0-1.0, default: 0.5)')
    parser.add_argument('--no-emotion', action='store_true', help='Disable emotion detection (faster)')
    parser.add_argument('--frame-skip', type=int, default=0, help='Skip N frames between processing (default: 0, for performance)')
    
    args = parser.parse_args()
    
    try:
        if args.image:
            # Process single image
            output_path = args.output if args.output else None
            results = process_image_file(
                image_path=args.image,
                output_path=output_path,
                min_confidence=args.confidence,
                enable_emotion=not args.no_emotion,
                export_json=args.export_json
            )
            logger.info(f"Successfully processed image: {len(results['faces'])} face(s) detected")
            
        else:
            # Process video or webcam
            video_source = args.video if args.video else 0
            logger.info(f"Starting video processing with {'emotion detection' if not args.no_emotion else 'face detection only'}...")
            
            process_video(
                video_path=video_source,
                output_dir=args.output,
                min_confidence=args.confidence,
                enable_emotion=not args.no_emotion,
                frame_skip=args.frame_skip,
                export_csv=args.export_csv,
                save_frames=args.save_frames
            )
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return 1
    finally:
        cv2.destroyAllWindows()
    
    return 0


if __name__ == "__main__":
    exit(main())
