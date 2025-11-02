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
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from deepface import DeepFace

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FaceDatabase:
    """Simple face database for face recognition."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize face database.
        
        Args:
            db_path: Path to JSON file storing face embeddings
        """
        self.db_path = db_path or 'face_database.json'
        self.database: Dict[str, Dict[str, Any]] = {}
        self.load_database()
    
    def load_database(self):
        """Load face database from file."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    self.database = json.load(f)
                logger.info(f"Loaded {len(self.database)} faces from database")
            except Exception as e:
                logger.warning(f"Could not load face database: {e}")
                self.database = {}
        else:
            self.database = {}
    
    def save_database(self):
        """Save face database to file."""
        try:
            with open(self.db_path, 'w') as f:
                json.dump(self.database, f, indent=2)
            logger.info(f"Saved {len(self.database)} faces to database")
        except Exception as e:
            logger.error(f"Could not save face database: {e}")
    
    def add_face(self, name: str, face_img: np.ndarray) -> bool:
        """
        Add a face to the database.
        
        Args:
            name: Name/label for the face
            face_img: Face image
            
        Returns:
            True if successful
        """
        try:
            embedding = DeepFace.represent(face_img, model_name='VGG-Face', enforce_detection=False)[0]['embedding']
            self.database[name] = {
                'embedding': embedding,
                'timestamp': datetime.now().isoformat()
            }
            self.save_database()
            return True
        except Exception as e:
            logger.error(f"Could not add face {name}: {e}")
            return False
    
    def find_match(self, face_img: np.ndarray, threshold: float = 0.6) -> Optional[Tuple[str, float]]:
        """
        Find matching face in database.
        
        Args:
            face_img: Face image to match
            threshold: Similarity threshold (lower = stricter)
            
        Returns:
            Tuple of (name, distance) if match found, None otherwise
        """
        if not self.database:
            return None
        
        try:
            embedding = DeepFace.represent(face_img, model_name='VGG-Face', enforce_detection=False)[0]['embedding']
            
            best_match = None
            best_distance = float('inf')
            
            for name, data in self.database.items():
                db_embedding = data['embedding']
                distance = np.linalg.norm(np.array(embedding) - np.array(db_embedding))
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = name
            
            if best_match and best_distance < threshold:
                similarity = 1 - (best_distance / threshold)
                return (best_match, similarity)
            
            return None
        except Exception as e:
            logger.debug(f"Face matching failed: {e}")
            return None


class FaceQualityScorer:
    """Score face quality for filtering."""
    
    @staticmethod
    def calculate_quality(face_img: np.ndarray) -> Dict[str, float]:
        """
        Calculate face quality metrics.
        
        Args:
            face_img: Cropped face image
            
        Returns:
            Dictionary with quality metrics
        """
        if face_img.size == 0:
            return {'score': 0.0, 'brightness': 0.0, 'sharpness': 0.0, 'size': 0.0}
        
        # Brightness (0-1, higher is better)
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0
        
        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = min(laplacian.var() / 1000.0, 1.0)  # Normalize
        
        # Size score (prefer larger faces)
        h, w = face_img.shape[:2]
        size_score = min((h * w) / (100 * 100), 1.0)
        
        # Overall quality score (weighted average)
        quality_score = (brightness * 0.3 + sharpness * 0.4 + size_score * 0.3)
        
        return {
            'score': quality_score,
            'brightness': brightness,
            'sharpness': sharpness,
            'size': size_score
        }


class StatisticsTracker:
    """Track statistics across video processing."""
    
    def __init__(self):
        self.total_faces = 0
        self.emotion_counts = defaultdict(int)
        self.age_sum = 0.0
        self.age_count = 0
        self.gender_counts = defaultdict(int)
        self.total_frames = 0
        self.frames_with_faces = 0
        self.recognized_faces = defaultdict(int)
        
    def update(self, faces: List[Dict[str, Any]]):
        """Update statistics with new face detections."""
        self.total_frames += 1
        if faces:
            self.frames_with_faces += 1
            self.total_faces += len(faces)
            for face in faces:
                if 'emotion' in face:
                    self.emotion_counts[face['emotion']] += 1
                if 'age' in face:
                    self.age_sum += face['age']
                    self.age_count += 1
                if 'gender' in face:
                    self.gender_counts[face['gender']] += 1
                if 'recognized_as' in face:
                    self.recognized_faces[face['recognized_as']] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        avg_age = self.age_sum / self.age_count if self.age_count > 0 else 0
        return {
            'total_frames': self.total_frames,
            'frames_with_faces': self.frames_with_faces,
            'total_faces': self.total_faces,
            'avg_faces_per_frame': self.total_faces / self.total_frames if self.total_frames > 0 else 0,
            'emotion_distribution': dict(self.emotion_counts),
            'gender_distribution': dict(self.gender_counts),
            'recognized_faces': dict(self.recognized_faces),
            'average_age': avg_age
        }


class FaceDetector:
    """Face detection and emotion analysis using MediaPipe and DeepFace."""
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        enable_emotion: bool = True,
        enable_age_gender: bool = False,
        blur_faces: bool = False,
        enable_recognition: bool = False,
        face_db: Optional[FaceDatabase] = None,
        min_quality: float = 0.0,
        export_crops: Optional[str] = None
    ):
        """
        Initialize the face detector with MediaPipe.
        
        Args:
            min_detection_confidence: Minimum confidence for face detection (0.0-1.0)
            enable_emotion: Whether to enable emotion detection (slower but more features)
            enable_age_gender: Whether to detect age and gender (slower)
            blur_faces: Whether to blur detected faces for privacy
            enable_recognition: Whether to enable face recognition
            face_db: Face database for recognition
            min_quality: Minimum face quality score to process (0.0-1.0)
            export_crops: Directory to export face crops
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )
        self.enable_emotion = enable_emotion
        self.enable_age_gender = enable_age_gender
        self.blur_faces = blur_faces
        self.enable_recognition = enable_recognition
        self.face_db = face_db
        self.min_quality = min_quality
        self.export_crops_dir = export_crops
        self.quality_scorer = FaceQualityScorer()
        self.executor = ThreadPoolExecutor(max_workers=3) if (enable_emotion or enable_age_gender or enable_recognition) else None
        
        if export_crops:
            os.makedirs(export_crops, exist_ok=True)
        
    def _analyze_face(self, face_img: np.ndarray) -> Dict[str, Any]:
        """
        Analyze face for emotion, age, and gender.
        
        Args:
            face_img: Cropped face image
            
        Returns:
            Dictionary with analysis results
        """
        results = {}
        
        if face_img.size == 0:
            return results
            
        # Resize if too small for DeepFace
        if face_img.shape[0] < 48 or face_img.shape[1] < 48:
            face_img = cv2.resize(face_img, (48, 48), interpolation=cv2.INTER_CUBIC)
        
        try:
            actions = []
            if self.enable_emotion:
                actions.append('emotion')
            if self.enable_age_gender:
                actions.extend(['age', 'gender'])
            
            if actions:
                analysis = DeepFace.analyze(
                    face_img,
                    actions=actions,
                    enforce_detection=False,
                    silent=True
                )
                
                if isinstance(analysis, list):
                    analysis = analysis[0]
                
                if 'emotion' in analysis:
                    results['emotion'] = analysis['dominant_emotion']
                if 'age' in analysis:
                    results['age'] = int(analysis['age'])
                if 'gender' in analysis:
                    results['gender'] = analysis['dominant_gender'].lower()
        except Exception as e:
            logger.debug(f"Face analysis failed: {str(e)}")
        
        return results
    
    def _recognize_face(self, face_img: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Recognize face using database.
        
        Args:
            face_img: Cropped face image
            
        Returns:
            Dictionary with recognition results or None
        """
        if not self.enable_recognition or not self.face_db:
            return None
        
        try:
            match = self.face_db.find_match(face_img)
            if match:
                name, similarity = match
                return {'recognized_as': name, 'similarity': similarity}
        except Exception as e:
            logger.debug(f"Face recognition failed: {str(e)}")
        
        return None
    
    def _blur_face(self, image: np.ndarray, x: int, y: int, width: int, height: int, blur_strength: int = 25) -> np.ndarray:
        """Blur a face region in the image."""
        face_roi = image[y:y+height, x:x+width]
        blurred = cv2.GaussianBlur(face_roi, (blur_strength, blur_strength), 0)
        image[y:y+height, x:x+width] = blurred
        return image
    
    def process_image(
        self,
        image: np.ndarray,
        draw_results: bool = True,
        detect_emotion: Optional[bool] = None,
        detect_age_gender: Optional[bool] = None,
        recognize: Optional[bool] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process a single image and return face detections with optional analysis.
        
        Args:
            image: Input image in BGR format
            draw_results: Whether to draw detection results on the image
            detect_emotion: Override emotion detection setting
            detect_age_gender: Override age/gender detection setting
            recognize: Override recognition setting
            
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
            should_detect_age_gender = detect_age_gender if detect_age_gender is not None else self.enable_age_gender
            should_recognize = recognize if recognize is not None else self.enable_recognition
            
            # Prepare face crops for parallel analysis
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
                
                # Extract face crop
                face_img = image[y:y+height, x:x+width].copy()
                
                # Calculate quality
                quality = self.quality_scorer.calculate_quality(face_img)
                
                # Skip low quality faces
                if quality['score'] < self.min_quality:
                    continue
                
                face_info = {
                    'bbox': (x, y, width, height),
                    'confidence': confidence,
                    'quality': quality,
                    'landmarks': [],
                    'timestamp': datetime.now().isoformat()
                }
                
                # Add facial landmarks
                for keypoint in detection.location_data.relative_keypoints:
                    px = int(keypoint.x * w)
                    py = int(keypoint.y * h)
                    face_info['landmarks'].append((px, py))
                
                # Export face crop if requested
                if self.export_crops_dir:
                    crop_filename = f"face_{len(face_info_list):03d}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                    crop_path = os.path.join(self.export_crops_dir, crop_filename)
                    cv2.imwrite(crop_path, face_img)
                    face_info['crop_path'] = crop_path
                
                # Blur face if requested (before analysis)
                if self.blur_faces:
                    image = self._blur_face(image, x, y, width, height)
                
                # Start face analysis in parallel if enabled
                analysis_futures = []
                if (should_detect_emotion or should_detect_age_gender) and self.executor:
                    future = self.executor.submit(self._analyze_face, face_img.copy())
                    analysis_futures.append(('analysis', future))
                elif should_detect_emotion or should_detect_age_gender:
                    analysis = self._analyze_face(face_img.copy())
                    face_info.update(analysis)
                
                if should_recognize and self.executor:
                    future = self.executor.submit(self._recognize_face, face_img.copy())
                    analysis_futures.append(('recognition', future))
                elif should_recognize:
                    recognition = self._recognize_face(face_img.copy())
                    if recognition:
                        face_info.update(recognition)
                
                if analysis_futures:
                    face_futures.append((len(face_info_list), analysis_futures))
                
                face_info_list.append(face_info)
            
            # Collect analysis results
            for face_idx, futures_list in face_futures:
                for analysis_type, future in futures_list:
                    try:
                        result = future.result(timeout=2.0)
                        if result:
                            face_info_list[face_idx].update(result)
                    except Exception as e:
                        logger.debug(f"Face {analysis_type} timeout/failure for face {face_idx}: {str(e)}")
            
            faces_info = face_info_list
            
            # Draw results if requested
            if draw_results:
                image = self._draw_detections(image, faces_info)
        
        return image, faces_info
    
    def _draw_detections(
        self,
        image: np.ndarray,
        faces_info: List[Dict[str, Any]],
        mode: str = 'detailed'
    ) -> np.ndarray:
        """
        Draw detection results on the image.
        
        Args:
            image: Image to draw on
            faces_info: List of face detection information
            mode: Visualization mode ('minimal', 'detailed', 'simple')
            
        Returns:
            Image with drawn detections
        """
        for face_info in faces_info:
            x, y, width, height = face_info['bbox']
            confidence = face_info['confidence']
            
            # Draw bounding box with color based on confidence
            box_color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.5 else (0, 165, 255)
            thickness = 3 if mode == 'detailed' else 2
            cv2.rectangle(image, (x, y), (x + width, y + height), box_color, thickness)
            
            if mode == 'minimal':
                continue
            
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
            
            if mode == 'detailed':
                # Draw landmarks
                for px, py in face_info['landmarks']:
                    cv2.circle(image, (px, py), 3, (0, 0, 255), -1)
            
            # Prepare info text
            info_lines = []
            if 'recognized_as' in face_info:
                similarity = face_info.get('similarity', 0) * 100
                info_lines.append(f"{face_info['recognized_as']} ({similarity:.0f}%)")
            if 'emotion' in face_info and face_info['emotion'] != 'unknown':
                info_lines.append(f"Emotion: {face_info['emotion'].upper()}")
            if 'age' in face_info:
                info_lines.append(f"Age: {face_info['age']}")
            if 'gender' in face_info:
                info_lines.append(f"Gender: {face_info['gender'].upper()}")
            if mode == 'detailed' and 'quality' in face_info:
                quality_score = face_info['quality']['score'] * 100
                info_lines.append(f"Quality: {quality_score:.0f}%")
            
            # Draw info text
            y_offset = y + height + 5
            for line in info_lines:
                (line_width, line_height), _ = cv2.getTextSize(
                    line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                cv2.rectangle(
                    image,
                    (x, y_offset),
                    (x + line_width + 10, y_offset + line_height + 5),
                    (255, 0, 0),
                    -1
                )
                cv2.putText(
                    image, line, (x + 5, y_offset + line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                )
                y_offset += line_height + 8
        
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


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with configuration
    """
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


def process_image_file(
    image_path: str,
    output_path: Optional[str] = None,
    min_confidence: float = 0.5,
    enable_emotion: bool = True,
    enable_age_gender: bool = False,
    blur_faces: bool = False,
    enable_recognition: bool = False,
    face_db: Optional[FaceDatabase] = None,
    min_quality: float = 0.0,
    export_crops: Optional[str] = None,
    export_json: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a single image file.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image (None = don't save)
        min_confidence: Minimum detection confidence
        enable_emotion: Whether to detect emotions
        enable_age_gender: Whether to detect age and gender
        blur_faces: Whether to blur faces for privacy
        enable_recognition: Whether to enable face recognition
        face_db: Face database for recognition
        min_quality: Minimum face quality score
        export_crops: Directory to export face crops
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
        enable_emotion=enable_emotion,
        enable_age_gender=enable_age_gender,
        blur_faces=blur_faces,
        enable_recognition=enable_recognition,
        face_db=face_db,
        min_quality=min_quality,
        export_crops=export_crops
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
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Exported results to: {export_json}")
    
    logger.info(f"Detected {len(faces)} face(s) in image")
    return results


def process_images_batch(
    image_dir: str,
    output_dir: Optional[str] = None,
    min_confidence: float = 0.5,
    enable_emotion: bool = True,
    enable_age_gender: bool = False,
    blur_faces: bool = False,
    enable_recognition: bool = False,
    face_db: Optional[FaceDatabase] = None,
    min_quality: float = 0.0,
    export_crops: Optional[str] = None,
    export_json: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process multiple images in a directory.
    
    Args:
        image_dir: Directory containing images
        output_dir: Directory to save processed images
        min_confidence: Minimum detection confidence
        enable_emotion: Whether to detect emotions
        enable_age_gender: Whether to detect age and gender
        blur_faces: Whether to blur faces
        enable_recognition: Whether to enable face recognition
        face_db: Face database for recognition
        min_quality: Minimum face quality score
        export_crops: Directory to export face crops
        export_json: Path to export results as JSON
        
    Returns:
        Dictionary with batch processing results
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [
        f for f in os.listdir(image_dir)
        if Path(f).suffix.lower() in image_extensions
    ]
    
    if not image_files:
        raise ValueError(f"No image files found in {image_dir}")
    
    logger.info(f"Processing {len(image_files)} images from {image_dir}")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'batch_timestamp': datetime.now().isoformat(),
        'source_directory': image_dir,
        'output_directory': output_dir,
        'total_images': len(image_files),
        'processed_images': [],
        'total_faces': 0
    }
    
    for idx, image_file in enumerate(image_files, 1):
        try:
            image_path = os.path.join(image_dir, image_file)
            output_path = os.path.join(output_dir, f"processed_{image_file}") if output_dir else None
            
            result = process_image_file(
                image_path=image_path,
                output_path=output_path,
                min_confidence=min_confidence,
                enable_emotion=enable_emotion,
                enable_age_gender=enable_age_gender,
                blur_faces=blur_faces,
                enable_recognition=enable_recognition,
                face_db=face_db,
                min_quality=min_quality,
                export_crops=export_crops,
                export_json=None  # Don't export individual JSONs
            )
            
            results['processed_images'].append(result)
            results['total_faces'] += result['faces_detected']
            logger.info(f"Processed {idx}/{len(image_files)}: {image_file} ({result['faces_detected']} faces)")
            
        except Exception as e:
            logger.error(f"Error processing {image_file}: {str(e)}")
            results['processed_images'].append({
                'image_path': image_path,
                'error': str(e)
            })
    
    # Export batch results
    if export_json:
        with open(export_json, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Exported batch results to: {export_json}")
    
    logger.info(f"Batch processing complete: {results['total_faces']} total faces detected")
    return results


def process_video(
    video_path: Union[int, str] = 0,
    output_dir: Optional[str] = None,
    min_confidence: float = 0.5,
    enable_emotion: bool = True,
    enable_age_gender: bool = False,
    blur_faces: bool = False,
    enable_recognition: bool = False,
    face_db: Optional[FaceDatabase] = None,
    min_quality: float = 0.0,
    export_crops: Optional[str] = None,
    frame_skip: int = 0,
    export_csv: Optional[str] = None,
    export_video: Optional[str] = None,
    save_frames: bool = False
):
    """
    Process video stream with face detection and emotion analysis.
    
    Args:
        video_path: Video source (0 for webcam, or video file path)
        output_dir: Directory to save frames
        min_confidence: Minimum detection confidence
        enable_emotion: Whether to detect emotions (slower)
        enable_age_gender: Whether to detect age and gender (slower)
        blur_faces: Whether to blur faces for privacy
        enable_recognition: Whether to enable face recognition
        face_db: Face database for recognition
        min_quality: Minimum face quality score
        export_crops: Directory to export face crops
        frame_skip: Process every Nth frame (0 = process all frames, for performance)
        export_csv: Path to export detection results as CSV
        export_video: Path to export processed video (None = don't export)
        save_frames: Whether to save frames with detections
    """
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    is_file = isinstance(video_path, str)
    if is_file:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video properties: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")
    
    # Initialize video writer if exporting
    video_writer = None
    if export_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(export_video, fourcc, fps, (width, height))
        logger.info(f"Exporting processed video to: {export_video}")
    
    # Initialize face detector
    detector = FaceDetector(
        min_detection_confidence=min_confidence,
        enable_emotion=enable_emotion,
        enable_age_gender=enable_age_gender,
        blur_faces=blur_faces,
        enable_recognition=enable_recognition,
        face_db=face_db,
        min_quality=min_quality,
        export_crops=export_crops
    )
    
    # Initialize FPS counter and statistics
    fps_counter = FPSCounter()
    stats = StatisticsTracker()
    
    # Prepare CSV export
    csv_file = None
    csv_writer = None
    if export_csv:
        csv_file = open(export_csv, 'w', newline='')
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=['frame', 'timestamp', 'faces_detected', 'face_id', 'confidence', 'emotion', 'age', 'gender', 'recognized_as', 'quality', 'bbox']
        )
        csv_writer.writeheader()
    
    frame_count = 0
    detection_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_file:
                    logger.info("End of video reached")
                break
            
            # Skip frames if requested (for performance)
            if frame_skip > 0 and frame_count % (frame_skip + 1) != 0:
                frame_count += 1
                # Still display/write frame without processing
                if is_file and video_writer:
                    video_writer.write(frame)
                if not is_file:
                    cv2.imshow('Face Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue
            
            # Process frame
            processed_frame, faces = detector.process_image(frame)
            
            # Update statistics
            stats.update(faces)
            
            # Update FPS
            current_fps = fps_counter.update()
            
            # Draw stats overlay on top of processed frame
            stats_lines = [
                f"FPS: {current_fps:.1f}",
                f"Faces: {len(faces)}",
                f"Frame: {frame_count}",
                f"Total: {stats.total_faces}",
            ]
            if is_file:
                stats_lines.append(f"Progress: {frame_count}/{total_frames} ({100*frame_count/total_frames:.1f}%)")
            
            # Draw semi-transparent background for stats
            overlay_height = len(stats_lines) * 30 + 20
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
            for stat in stats_lines:
                cv2.putText(
                    processed_frame, stat, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                y_offset += 30
            
            # Write to video file if exporting
            if video_writer:
                video_writer.write(processed_frame)
            
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
                        'age': face.get('age', ''),
                        'gender': face.get('gender', ''),
                        'recognized_as': face.get('recognized_as', ''),
                        'quality': face.get('quality', {}).get('score', 0),
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
            
            # Display the frame (only for webcam or preview)
            if not is_file or not video_writer:
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
        if video_writer:
            video_writer.release()
            logger.info(f"Exported processed video to: {export_video}")
        if csv_file:
            csv_file.close()
            logger.info(f"Exported detection results to: {export_csv}")
        
        # Print statistics summary
        summary = stats.get_summary()
        logger.info("=" * 50)
        logger.info("Processing Statistics:")
        logger.info(f"  Total frames: {summary['total_frames']}")
        logger.info(f"  Frames with faces: {summary['frames_with_faces']}")
        logger.info(f"  Total faces detected: {summary['total_faces']}")
        logger.info(f"  Average faces per frame: {summary['avg_faces_per_frame']:.2f}")
        if summary['emotion_distribution']:
            logger.info(f"  Emotion distribution: {summary['emotion_distribution']}")
        if summary['gender_distribution']:
            logger.info(f"  Gender distribution: {summary['gender_distribution']}")
        if summary['recognized_faces']:
            logger.info(f"  Recognized faces: {summary['recognized_faces']}")
        if summary['average_age'] > 0:
            logger.info(f"  Average age: {summary['average_age']:.1f}")
        logger.info("=" * 50)
        
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

  # Process batch of images
  python face_detector.py --batch images/ --output processed/

  # Face recognition (add face first)
  python face_detector.py --add-face "John Doe" --image photo.jpg
  python face_detector.py --video video.mp4 --recognize

  # Export face crops
  python face_detector.py --image photo.jpg --export-crops crops/

  # Quality filtering
  python face_detector.py --video video.mp4 --min-quality 0.5

  # Save frames and export CSV
  python face_detector.py --video video.mp4 --output frames/ --export-csv results.csv

  # Export processed video
  python face_detector.py --video input.mp4 --export-video output.mp4

  # Age/gender detection with emotion
  python face_detector.py --image photo.jpg --age-gender

  # Blur faces for privacy
  python face_detector.py --video video.mp4 --blur-faces

  # Configuration file
  python face_detector.py --config config.json

  # Fast mode (no emotion detection, skip frames)
  python face_detector.py --no-emotion --frame-skip 2
        """
    )
    
    # Input source
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--video', type=str, help='Path to input video file (omit for webcam)')
    input_group.add_argument('--image', type=str, help='Path to input image file')
    input_group.add_argument('--batch', type=str, help='Directory containing images for batch processing')
    
    # Output options
    parser.add_argument('--output', type=str, help='Output directory (video/batch) or file path (image)')
    parser.add_argument('--export-csv', type=str, help='Export detection results to CSV file')
    parser.add_argument('--export-json', type=str, help='Export detection results to JSON file')
    parser.add_argument('--export-video', type=str, help='Export processed video to file')
    parser.add_argument('--export-crops', type=str, help='Directory to export individual face crops')
    parser.add_argument('--save-frames', action='store_true', help='Save frames with detections (video only)')
    
    # Processing options
    parser.add_argument('--confidence', type=float, default=0.5, help='Minimum detection confidence (0.0-1.0, default: 0.5)')
    parser.add_argument('--no-emotion', action='store_true', help='Disable emotion detection (faster)')
    parser.add_argument('--age-gender', action='store_true', help='Enable age and gender detection')
    parser.add_argument('--blur-faces', action='store_true', help='Blur detected faces for privacy')
    parser.add_argument('--frame-skip', type=int, default=0, help='Skip N frames between processing (default: 0, for performance)')
    parser.add_argument('--min-quality', type=float, default=0.0, help='Minimum face quality score (0.0-1.0, default: 0.0)')
    
    # Face recognition options
    parser.add_argument('--recognize', action='store_true', help='Enable face recognition')
    parser.add_argument('--add-face', type=str, help='Add face to database (requires --image and name)')
    parser.add_argument('--face-db', type=str, default='face_database.json', help='Path to face database file')
    
    # Configuration
    parser.add_argument('--config', type=str, help='Path to JSON configuration file')
    
    args = parser.parse_args()
    
    # Load configuration file if provided
    config = {}
    if args.config:
        config = load_config(args.config)
        # Override with config values (command line args take precedence)
        min_confidence = config.get('confidence', args.confidence)
        enable_emotion = config.get('enable_emotion', not args.no_emotion)
        enable_age_gender = config.get('enable_age_gender', args.age_gender)
        enable_recognition = config.get('enable_recognition', args.recognize)
        min_quality = config.get('min_quality', args.min_quality)
        blur_faces = config.get('blur_faces', args.blur_faces)
    else:
        min_confidence = args.confidence
        enable_emotion = not args.no_emotion
        enable_age_gender = args.age_gender
        enable_recognition = args.recognize
        min_quality = args.min_quality
        blur_faces = args.blur_faces
    
    # Initialize face database if recognition is enabled
    face_db = None
    if enable_recognition or args.add_face:
        face_db = FaceDatabase(args.face_db)
    
    # Handle adding face to database
    if args.add_face:
        if not args.image:
            logger.error("--add-face requires --image")
            return 1
        
        image = cv2.imread(args.image)
        if image is None:
            logger.error(f"Could not read image: {args.image}")
            return 1
        
        # Detect face first
        detector = FaceDetector(min_detection_confidence=0.5)
        _, faces = detector.process_image(image, draw_results=False)
        
        if not faces:
            logger.error("No faces detected in image")
            return 1
        
        # Use the first detected face
        x, y, w, h = faces[0]['bbox']
        face_img = image[y:y+h, x:x+w]
        
        if face_db.add_face(args.add_face, face_img):
            logger.info(f"Successfully added '{args.add_face}' to face database")
            return 0
        else:
            logger.error("Failed to add face to database")
            return 1
    
    try:
        if args.batch:
            # Batch process images
            results = process_images_batch(
                image_dir=args.batch,
                output_dir=args.output,
                min_confidence=min_confidence,
                enable_emotion=enable_emotion,
                enable_age_gender=enable_age_gender,
                blur_faces=blur_faces,
                enable_recognition=enable_recognition,
                face_db=face_db,
                min_quality=min_quality,
                export_crops=args.export_crops,
                export_json=args.export_json
            )
            logger.info(f"Batch processing complete: {results['total_faces']} total faces in {results['total_images']} images")
            
        elif args.image:
            # Process single image
            output_path = args.output if args.output else None
            results = process_image_file(
                image_path=args.image,
                output_path=output_path,
                min_confidence=min_confidence,
                enable_emotion=enable_emotion,
                enable_age_gender=enable_age_gender,
                blur_faces=blur_faces,
                enable_recognition=enable_recognition,
                face_db=face_db,
                min_quality=min_quality,
                export_crops=args.export_crops,
                export_json=args.export_json
            )
            logger.info(f"Successfully processed image: {len(results['faces'])} face(s) detected")
            
        else:
            # Process video or webcam
            video_source = args.video if args.video else 0
            features = []
            if enable_emotion:
                features.append('emotion')
            if enable_age_gender:
                features.append('age/gender')
            if enable_recognition:
                features.append('recognition')
            feature_str = ' and '.join(features) if features else 'face detection only'
            
            logger.info(f"Starting video processing with {feature_str}...")
            
            process_video(
                video_path=video_source,
                output_dir=args.output,
                min_confidence=min_confidence,
                enable_emotion=enable_emotion,
                enable_age_gender=enable_age_gender,
                blur_faces=blur_faces,
                enable_recognition=enable_recognition,
                face_db=face_db,
                min_quality=min_quality,
                export_crops=args.export_crops,
                frame_skip=args.frame_skip,
                export_csv=args.export_csv,
                export_video=args.export_video,
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
