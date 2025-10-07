import json
import os
import cv2
import numpy as np
import ast
import torch
import glob
from tqdm import tqdm
from insightface.app import FaceAnalysis
from datetime import datetime
from PIL import Image
import sys
from torchvision import transforms

# Add project root to path to allow direct script execution
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rg_fiqa.models.rg_fiqa import RGFIQA

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'face_fusion', 'ProxyFusion-NeurIPS-main'))
from models.fusion_models import ProxyFusion
PROXY_FUSION_AVAILABLE = True
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'face_fusion', 'ProxyFusion-NeurIPS-main', 'checkpoints', 'ProxyFusion_ckpt_Adaface_RetinaFace_Proxy4x4_TAR@FARe-2_0.6910.pth.tar')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_thresh=0.4, det_size=(320, 320))

# --- Load RG-FIQA model ---
RGFIQA_CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), '..', 'rg_fiqa', 'checkpoints', 'rg_fiqa_best.pth')
tfq_device = DEVICE
rgfiqa_model = RGFIQA().to(tfq_device)
try:
    rgfiqa_model.load_state_dict(torch.load(RGFIQA_CHECKPOINT_PATH, map_location=tfq_device))
    rgfiqa_model.eval()
    print("RG-FIQA model loaded successfully!")
except FileNotFoundError:
    print(f"Warning: RG-FIQA model checkpoint not found at {RGFIQA_CHECKPOINT_PATH}")
    rgfiqa_model = None

tfq_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])

@torch.no_grad()
def rgfiqa_predict(image_bgr):
    """Predicts image quality score using RG-FIQA."""
    if rgfiqa_model is None or image_bgr is None:
        return 0.0
    
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    x = tfq_transform(rgb).unsqueeze(0).to(tfq_device)
    score = rgfiqa_model(x).item()
    return float(score)
# -------------------------


# Global variable for caching the model instance
fusion_model = None

def get_fusion_model():
    """
    Lazily loads and caches the ProxyFusion model (Singleton pattern).
    The model is loaded only on the first call.
    """
    global fusion_model
    if fusion_model is not None:
        return fusion_model

    if PROXY_FUSION_AVAILABLE and os.path.exists(MODEL_PATH):
        try:
            model = ProxyFusion(DIM=512).to(DEVICE)
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            if "state_dict" in checkpoint and "model_weights" in checkpoint["state_dict"]:
                model.load_state_dict(checkpoint["state_dict"]["model_weights"], strict=False)
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"], strict=False)
            elif "model" in checkpoint:
                model.load_state_dict(checkpoint["model"], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            model.eval()
            print("ProxyFusion model loaded successfully!")
            fusion_model = model  # Cache the model
            return fusion_model
        except Exception as e:
            print(f"Failed to load ProxyFusion model: {e}. Will use simple average fusion.")
            fusion_model = None # Mark as failed
            return None
    else:
        print("ProxyFusion model file not found or unavailable. Will use simple average fusion.")
        fusion_model = None # Mark as unavailable
        return None



def load_face_features(file_path):
    """Loads student numbers and face features from a faces.json file, handling string or null faces."""
    try:
        with open(file_path, 'r') as f:
            faces_data = json.load(f)
        if faces_data is None:
            return None
        if not faces_data or not isinstance(faces_data, list):
            return None
        image_feats = {}
        for face in faces_data:
            if face.get("face") is not None:
                try:
                    if isinstance(face["face"], str):
                        if not face["face"].strip():
                            print(f"Empty faces string for student {face.get('xh', 'unknown')}, skipping")
                            continue
                        try:
                            face_list = ast.literal_eval(face["face"])
                            face_array = np.array(face_list)
                            if face_array.size > 0:
                                image_feats[face["xh"]] = face_array / np.linalg.norm(face_array)
                        except (ValueError, SyntaxError) as e:
                            print(f"Error parsing faces feature for student {face.get('xh', 'unknown')}: {e}, skipping")
                            continue
                    elif isinstance(face["face"], list):
                        face_array = np.array(face["face"])
                        if face_array.size > 0:
                            image_feats[face["xh"]] = face_array / np.linalg.norm(face_array)
                except Exception as e:
                    print(f"Unexpected error processing student {face.get('xh', 'unknown')}: {e}, skipping")
                    continue
        if not image_feats:
            return None
        return image_feats
    except Exception as e:
        print(f"Error loading face features from {file_path}: {e}")
        return None


def compute_quality(image):
    """
    Computes the quality score for a face image.
    This version uses the pre-trained RG-FIQA model for prediction.
    """
    return rgfiqa_predict(image)


def get_face_quality_from_insightface(image):
    """Gets face detection confidence from insightface as a quality score."""
    if image is None:
        return 0, None
    
    faces = app.get(image)
    if not faces:
        return 0, None
    
    best_face = max(faces, key=lambda f: f.det_score)
    return float(best_face.det_score), best_face


def compute_pose_quality(yaw, pitch, roll):
    """
    Computes a quality score based on face pose angles.
    """
    yaw_rad = abs(np.radians(yaw))
    pitch_rad = abs(np.radians(pitch)) 
    roll_rad = abs(np.radians(roll))
    
    yaw_score = np.exp(-(yaw_rad**2) / (2 * (np.radians(30))**2))
    pitch_score = np.exp(-(pitch_rad**2) / (2 * (np.radians(20))**2))
    roll_score = np.exp(-(roll_rad**2) / (2 * (np.radians(15))**2))
    
    pose_score = 0.4 * yaw_score + 0.3 * pitch_score + 0.3 * roll_score
    
    return pose_score


def get_face_pose_angles(face):
    """
    Extracts pose angles from an insightface face object.
    """
    try:
        if hasattr(face, 'pose') and face.pose is not None:
            if len(face.pose) >= 3:
                yaw, pitch, roll = face.pose[:3]
                return float(yaw), float(pitch), float(roll)
        
        if hasattr(face, 'kps') and face.kps is not None:
            kps = face.kps.reshape(-1, 2)
            if len(kps) >= 5:
                left_eye = kps[0]
                right_eye = kps[1]
                nose = kps[2]
                left_mouth = kps[3]
                right_mouth = kps[4]
                
                eye_center = (left_eye + right_eye) / 2
                eye_distance = np.linalg.norm(right_eye - left_eye)
                
                yaw = np.degrees(np.arctan2(nose[0] - eye_center[0], eye_distance)) * 2
                
                mouth_center = (left_mouth + right_mouth) / 2
                eye_mouth_distance = mouth_center[1] - eye_center[1]
                expected_distance = eye_distance * 1.2
                pitch = np.degrees(np.arctan2(eye_mouth_distance - expected_distance, expected_distance)) * 1.5
                
                roll = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
                
                return float(yaw), float(pitch), float(roll)
        
        return 0.0, 0.0, 0.0
        
    except Exception as e:
        print(f"Error extracting pose angles: {e}")
        return 0.0, 0.0, 0.0


def get_comprehensive_quality_score(image, face):
    """Comprehensive quality assessment: combines image quality, detection confidence, and face pose."""
    if image is None or face is None:
        return 0
    
    # 1. Image Quality Score
    image_quality = compute_quality(image)
    
    # 2. Face Detection Confidence
    detection_confidence = float(face.det_score)
    
    # 3. Face Pose Quality Score
    yaw, pitch, roll = get_face_pose_angles(face)
    pose_quality = compute_pose_quality(yaw, pitch, roll)
    
    # 4. Linearly combine scores (weights are adjustable)
    w_image = 0.4
    w_detection = 0.3
    w_pose = 0.3
    
    comprehensive_score = (w_image * image_quality + 
                          w_detection * detection_confidence + 
                          w_pose * pose_quality)
    
    return comprehensive_score


def detect_faces_and_landmarks(image):
    """Detects faces in an image and extracts features."""
    if image is None:
        print(f"Failed to load image")
        return []
    results = []
    faces = app.get(image)
    for face in faces:
        bbox = face.bbox
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
        results.append({
            'box': (x, y, w, h),
            'embedding': face.embedding,
            'det_score': float(face.det_score),
        })
    return results


def apply_quality_weights(features_list, quality_scores):
    """
    Applies quality weights to a list of features while maintaining their norm.
    """
    if len(quality_scores) == 0:
        return features_list
    
    quality_weights = np.array(quality_scores)
    quality_weights = quality_weights / (np.sum(quality_weights) + 1e-8)
    
    weighted_features = []
    for i, feat in enumerate(features_list):
        feat_norm = feat / np.linalg.norm(feat)
        weighted_feat = feat_norm * quality_weights[i]
        if np.linalg.norm(weighted_feat) > 1e-8:
            weighted_feat = weighted_feat / np.linalg.norm(weighted_feat)
        weighted_features.append(weighted_feat)
    
    return np.array(weighted_features)


def fuse_person_features(person_path):
    """Fuses features from all images in a person's folder using quality weighting."""
    image_files = [f for f in os.listdir(person_path) if f.endswith('.jpg')]
    
    if not image_files:
        return None, 0, []
    
    features = []
    quality_scores = []
    valid_images = []
    skipped_multi_face = []
    images_for_quality = []
    
    for image_file in image_files:
        image_path = os.path.join(person_path, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            continue
        
        faces = app.get(image)
        
        if len(faces) > 1:
            print(f"Skipping multi-face image: {image_file} (detected {len(faces)} faces)")
            skipped_multi_face.append(image_file)
            continue
        elif len(faces) == 1:
            best_face = faces[0]
            
            comprehensive_quality = get_comprehensive_quality_score(image, best_face)
            
            image_quality = compute_quality(image)
            detection_confidence = float(best_face.det_score)
            yaw, pitch, roll = get_face_pose_angles(best_face)
            pose_quality = compute_pose_quality(yaw, pitch, roll)
            
            features.append(best_face.embedding)
            quality_scores.append(comprehensive_quality)
            valid_images.append(image_file)
            images_for_quality.append(image)
    
    if not features:
        print(f"No valid single-face images found in this person's folder (Total: {len(image_files)}, Skipped multi-face: {len(skipped_multi_face)})")
        return None, 0, [], []
    
    features = np.array(features)
    quality_scores = np.array(quality_scores)
    avg_quality = np.mean(quality_scores)
    
    print(f"Person feature extraction complete: Total images {len(image_files)}, Valid single-face {len(valid_images)}, Skipped multi-face {len(skipped_multi_face)}")
    print(f"Quality score range: {np.min(quality_scores):.3f} - {np.max(quality_scores):.3f}, Average: {avg_quality:.3f}")
    
    fusion_model_instance = get_fusion_model()
    if fusion_model_instance is not None and len(features) > 1:
        try:
            with torch.no_grad():
                print("Using hybrid fusion method: quality selection then ProxyFusion")
                
                quality_threshold = np.percentile(quality_scores, 20)
                high_quality_indices = quality_scores >= quality_threshold
                
                if np.sum(high_quality_indices) >= 2:
                    selected_features = features[high_quality_indices]
                    print(f"Selected {len(selected_features)}/{len(features)} high-quality features for ProxyFusion")
                else:
                    selected_features = features
                    print(f"Using all {len(features)} features for ProxyFusion")
                
                normalized_features = []
                for feat in selected_features:
                    normalized_features.append(feat / np.linalg.norm(feat))
                normalized_features = np.array(normalized_features)
                
                input_tensor = torch.tensor(normalized_features, dtype=torch.float32, device=DEVICE)
                fused = fusion_model_instance.eval_fuse_probe(input_tensor)
                fused_feature = fused.mean(dim=0).cpu().numpy()
                fused_feature = fused_feature / np.linalg.norm(fused_feature)
                print(f"✓ Hybrid fusion complete")
                return fused_feature, avg_quality, valid_images, skipped_multi_face
        except Exception as e:
            print(f"ProxyFusion failed: {e}. Falling back to simple weighted average.")
    
    if len(features) > 1:
        normalized_features = []
        for feat in features:
            normalized_features.append(feat / np.linalg.norm(feat))
        normalized_features = np.array(normalized_features)
        
        weighted_features = apply_quality_weights(normalized_features, quality_scores)
        
        fused_feature = np.sum(weighted_features, axis=0)
        fused_feature = fused_feature / np.linalg.norm(fused_feature)
        print(f"Fused {len(features)} features using quality weighted average.")
        return fused_feature, avg_quality, valid_images, skipped_multi_face
    else:
        single_feature = features[0] / np.linalg.norm(features[0])
        return single_feature, avg_quality, valid_images, skipped_multi_face


def select_best_face_per_person_with_fusion(class_folder):
    """Generates a final feature for each person using feature fusion."""
    person_features = {}
    person_folders = [f for f in os.listdir(class_folder) if f.startswith('person_')]
    
    print(f"Found {len(person_folders)} person folders in {class_folder}")
    
    for person_folder in person_folders:
        person_path = os.path.join(class_folder, person_folder)
        if not os.path.isdir(person_path):
            continue
        
        fused_feature, avg_quality, valid_images, skipped_multi_face = fuse_person_features(person_path)
        
        if fused_feature is not None:
            person_features[person_folder] = {
                'embedding': fused_feature,
                'person_path': person_path,
                'quality_score': avg_quality,
                'image_count': len(valid_images),
                'valid_images': valid_images,
                'skipped_multi_face': skipped_multi_face,
                'total_images': len(valid_images) + len(skipped_multi_face)
            }
    
    return person_features


def process_person_matching(image_feats, person_features, similarity_threshold=0):
    """Handles person-to-student matching."""
    matched_students = {}
    person_match_details = {}
    successful_students = []
    
    print(f"Processing {len(person_features)} persons against {len(image_feats)} known students")
    
    for person_id, person_data in person_features.items():
        face_feat = person_data['embedding']
        
        best_match = None
        best_score = float('-inf')
        
        for xh, img_feat in image_feats.items():
            score = np.dot(img_feat, face_feat.T)
            if score > best_score:
                best_score = score
                best_match = xh
        
        person_match_details[person_id] = {
            'best_match': best_match,
            'similarity_score': float(best_score),
            'person_path': person_data['person_path'],
            'quality_score': float(person_data['quality_score']),
            'image_count': person_data['image_count'],
            'valid_images': person_data['valid_images'],
            'skipped_multi_face': person_data['skipped_multi_face'],
            'total_images': person_data['total_images'],
            'matched': bool(best_score >= similarity_threshold)
        }
        
        if best_score >= similarity_threshold:
            if best_match not in matched_students:
                matched_students[best_match] = []
                successful_students.append(best_match)
            matched_students[best_match].append({
                'person_id': person_id,
                'similarity_score': float(best_score),
                'quality_score': float(person_data['quality_score']),
                'image_count': person_data['image_count']
            })
    
    matched_persons = sum(1 for details in person_match_details.values() if details['matched'])
    total_persons = len(person_features)
    unique_matched_students = len(matched_students)
    
    all_similarity_scores = [details['similarity_score'] for details in person_match_details.values() if details['matched']]
    average_similarity = float(np.mean(all_similarity_scores)) if all_similarity_scores else 0.0
    
    all_quality_scores = [person_data['quality_score'] for person_data in person_features.values()]
    average_quality = float(np.mean(all_quality_scores)) if all_quality_scores else 0.0
    
    all_image_counts = [person_data['image_count'] for person_data in person_features.values()]
    average_image_count = float(np.mean(all_image_counts)) if all_image_counts else 0.0
    
    all_total_images = [person_data['total_images'] for person_data in person_features.values()]
    all_skipped_counts = [len(person_data['skipped_multi_face']) for person_data in person_features.values()]
    total_images_processed = sum(all_total_images)
    total_skipped_multi_face = sum(all_skipped_counts)
    
    return matched_students, person_match_details, successful_students, {
        'total_persons': int(total_persons),
        'matched_persons': int(matched_persons),
        'unique_matched_students': int(unique_matched_students),
        'person_match_rate': float(round((matched_persons / total_persons * 100), 2)) if total_persons > 0 else 0.0,
        'student_recognition_rate': float(round((unique_matched_students / len(image_feats) * 100), 2)) if len(image_feats) > 0 else 0.0,
        'average_similarity': float(round(average_similarity, 4)),
        'average_quality_score': float(round(average_quality, 2)),
        'average_image_count': float(round(average_image_count, 1)),
        'total_images_processed': int(total_images_processed),
        'total_skipped_multi_face': int(total_skipped_multi_face),
        'single_face_ratio': float(round(((total_images_processed - total_skipped_multi_face) / total_images_processed * 100), 2)) if total_images_processed > 0 else 0.0
    }


def process_class_folder(class_folder):
    """Processes a single class folder."""
    attendance_file = os.path.join(class_folder, 'person_attendance_proxyfusion_quality.json')
    
    nb116_class_folder = class_folder.replace('NB116_person_spatial', 'NB116')
    faces_file = os.path.join(nb116_class_folder, 'faces.json')
    
    print(f"Processing class folder: {class_folder}")
    print(f"Looking for faces.json in: {faces_file}")
    
    if not os.path.exists(faces_file):
        print(f"faces.json not found in {faces_file}")
        return False
    
    image_feats = load_face_features(faces_file)
    if image_feats is None:
        print(f"Invalid or empty faces.json in {faces_file}")
        return False

    person_features = select_best_face_per_person_with_fusion(class_folder)
    
    if not person_features:
        print(f"No valid person features found in {class_folder}")
        return False
    
    matched_students, person_match_details, successful_students, stats = process_person_matching(image_feats, person_features)
    
    print(f"\n=== Recognition Stats ===")
    print(f"Total persons: {stats['total_persons']}")
    print(f"Matched persons: {stats['matched_persons']}")
    print(f"Unique matched students: {stats['unique_matched_students']}")
    print(f"Person match rate: {stats['person_match_rate']}%")
    print(f"Student recognition rate: {stats['student_recognition_rate']}%")
    print(f"Average similarity score: {stats['average_similarity']}")
    print(f"Average quality score: {stats['average_quality_score']}")
    print(f"Average image count per person: {stats['average_image_count']}")
    
    print(f"\n=== Image Processing Stats ===")
    print(f"Total images processed: {stats['total_images_processed']}")
    print(f"Single face images used: {stats['total_images_processed'] - stats['total_skipped_multi_face']}")
    print(f"Multi-face images skipped: {stats['total_skipped_multi_face']}")
    print(f"Single face ratio: {stats['single_face_ratio']}%")
    
    print(f"\n=== Successfully Recognized Students ({len(successful_students)}) ===")
    if successful_students:
        successful_students.sort()
        for i, student_id in enumerate(successful_students, 1):
            similarity_scores = [match['similarity_score'] for match in matched_students[student_id]]
            max_similarity = max(similarity_scores)
            print(f"{i:2d}. {student_id} (Max Similarity: {max_similarity:.4f})")
    else:
        print("No students were successfully recognized.")
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Processing time: {current_time}")
    
    attendance = {
        'class_folder': class_folder,
        'faces_json_source': faces_file,
        'matched_students': {k: v for k, v in matched_students.items()},
        'successful_students': successful_students,
        'person_match_details': person_match_details,
        'fusion_method': 'ProxyFusion' if get_fusion_model() is not None else 'Simple Average',
        'stats': {
            **stats,
            'total_faces_in_json': len(image_feats),
            'processing_time': current_time
        }
    }
    
    with open(attendance_file, 'w') as f:
        json.dump(attendance, f, indent=4, ensure_ascii=False)
    
    print(f"Saved attendance data to {attendance_file}")
    return True


def process_nb116_person_folder(base_folder="NB116_person"):
    """Processes all courses in the NB116_person folder."""
    if not os.path.exists(base_folder):
        print(f"Base folder {base_folder} does not exist")
        return
    
    processed_count = 0
    success_count = 0
    
    print(f"Starting to process NB116_person courses in {base_folder}")
    
    # Traverse all date folders
    for date_folder in os.listdir(base_folder):
        date_path = os.path.join(base_folder, date_folder)
        if not os.path.isdir(date_path):
            continue
        
        print(f"\nProcessing date folder: {date_folder}")
        
        # Traverse all class folders
        for class_folder in os.listdir(date_path):
            if class_folder.startswith("Class_"):
                class_path = os.path.join(date_path, class_folder)
                if os.path.isdir(class_path):
                    processed_count += 1
                    if process_class_folder(class_path):
                        success_count += 1
    
    print(f"\n=== Processing Summary ===")
    print(f"Total classes processed: {processed_count}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {processed_count - success_count}")
    if processed_count > 0:
        success_rate = (success_count / processed_count) * 100
        print(f"Success rate: {success_rate:.2f}%")


if __name__ == "__main__":
    process_nb116_person_folder(r"dataset\NB116_person_spatial")
