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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'face_fusion', 'ProxyFusion-NeurIPS-main'))
from models.fusion_models import ProxyFusion
PROXY_FUSION_AVAILABLE = True
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'face_fusion', 'ProxyFusion-NeurIPS-main', 'checkpoints', 'ProxyFusion_ckpt_Adaface_RetinaFace_Proxy4x4_TAR@FARe-2_0.6910.pth.tar')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_thresh=0.4, det_size=(320, 320))

# Initialize ProxyFusion model (if available)
fusion_model = None
if PROXY_FUSION_AVAILABLE and os.path.exists(MODEL_PATH):
    try:
        fusion_model = ProxyFusion(DIM=512).to(DEVICE)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        if "state_dict" in checkpoint and "model_weights" in checkpoint["state_dict"]:
            fusion_model.load_state_dict(checkpoint["state_dict"]["model_weights"], strict=False)
        elif "state_dict" in checkpoint:
            fusion_model.load_state_dict(checkpoint["state_dict"], strict=False)
        elif "model" in checkpoint:
            fusion_model.load_state_dict(checkpoint["model"], strict=False)
        else:
            fusion_model.load_state_dict(checkpoint, strict=False)
        fusion_model.eval()
        print("ProxyFusion model loaded successfully!")
    except Exception as e:
        print(f"Failed to load ProxyFusion model: {e}, falling back to simple average")
        fusion_model = None


def load_face_features(file_path):
    """Load faces.json for each class to get student IDs and face features.
    Handles the case where the "face" field is a string or null."""
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
                        # Check if the string is empty or whitespace only
                        if not face["face"].strip():
                            print(f"Empty faces string for student {face.get('xh', 'unknown')}, skipping")
                            continue
                        try:
                            face_list = ast.literal_eval(face["face"])
                            # Normalize the feature vector
                            face_array = np.array(face_list)
                            if face_array.size > 0:  # Ensure array is not empty
                                image_feats[face["xh"]] = face_array / np.linalg.norm(face_array)
                        except (ValueError, SyntaxError) as e:
                            print(f"Error parsing faces feature for student {face.get('xh', 'unknown')}: {e}, skipping")
                            continue
                    elif isinstance(face["face"], list):
                        # Normalize the feature vector
                        face_array = np.array(face["face"])
                        if face_array.size > 0:  # Ensure array is not empty
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


def get_face_quality_from_insightface(image):
    """Use insightface to get face detection confidence as a quality score."""
    if image is None:
        return 0, None
    
    faces = app.get(image)
    if not faces:
        return 0, None
    
    # Select the face with the highest confidence
    best_face = max(faces, key=lambda f: f.det_score)
    return float(best_face.det_score), best_face


def detect_faces_and_landmarks(image):
    """Detect faces in the image and extract features."""
    if image is None:
        print(f"加载图像失败")
        return []
    results = []
    faces = app.get(image)
    for face in faces:
        bbox = face.bbox
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
        # if face.pose[0] > -20:  
        results.append({
            'box': (x, y, w, h),
            'embedding': face.embedding,
            'det_score': float(face.det_score),  # 添加置信度
        })
    return results


def fuse_person_features(person_path):
    """融合一个person文件夹中所有图片的特征"""
    image_files = [f for f in os.listdir(person_path) if f.endswith('.jpg')]
    
    if not image_files:
        return None, 0, []
    
    # 提取所有图片的特征
    features = []
    quality_scores = []
    valid_images = []
    skipped_multi_face = []
    
    for image_file in image_files:
        image_path = os.path.join(person_path, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            continue
        
        # 使用insightface提取特征
        faces = app.get(image)
        
        # Check number of faces; skip images with multiple faces
        if len(faces) > 1:
            print(f"跳过多人脸图片: {image_file} (检测到 {len(faces)} 张人脸)")
            skipped_multi_face.append(image_file)
            continue
        elif len(faces) == 1:
            best_face = faces[0]  # Only one face, use directly
            features.append(best_face.embedding)
            quality_scores.append(float(best_face.det_score))
            valid_images.append(image_file)
    
    if not features:
        print(f"该person文件夹中没有有效的单人脸图片 (总图片: {len(image_files)}, 跳过多人脸: {len(skipped_multi_face)})")
        return None, 0, [], []
    
    features = np.array(features)
    avg_quality = np.mean(quality_scores)
    
    # 打印处理统计信息
    print(f"Person特征提取完成: 总图片 {len(image_files)}, 有效单人脸 {len(valid_images)}, 跳过多人脸 {len(skipped_multi_face)}")
    # Fuse features using ProxyFusion (if available)
    if fusion_model is not None and len(features) > 1:
        try:
            with torch.no_grad():
                input_tensor = torch.tensor(features, dtype=torch.float32, device=DEVICE)
                fused = fusion_model.eval_fuse_probe(input_tensor)
                fused_feature = fused.mean(dim=0).cpu().numpy()
                # Normalize fused feature
                fused_feature = fused_feature / np.linalg.norm(fused_feature)
                print(f"Fused {len(features)} features using ProxyFusion")
                return fused_feature, avg_quality, valid_images, skipped_multi_face
        except Exception as e:
            print(f"ProxyFusion fusion failed: {e}, using simple average instead")
    
    # Simple average fusion (fallback)
    if len(features) > 1:
        # Normalize each feature
        normalized_features = []
        for feat in features:
            normalized_features.append(feat / np.linalg.norm(feat))
        # Average fusion
        fused_feature = np.mean(normalized_features, axis=0)
        fused_feature = fused_feature / np.linalg.norm(fused_feature)
        print(f"Fused {len(features)} features using simple average")
        return fused_feature, avg_quality, valid_images, skipped_multi_face
    else:
        # Only one feature, normalize and return
        single_feature = features[0] / np.linalg.norm(features[0])
        return single_feature, avg_quality, valid_images, skipped_multi_face


def select_best_face_per_person_with_fusion(class_folder):
    """Generate final features per person using feature fusion."""
    person_features = {}
    person_folders = [f for f in os.listdir(class_folder) if f.startswith('person_')]
    
    print(f"Found {len(person_folders)} person folders in {class_folder}")
    
    for person_folder in person_folders:
        person_path = os.path.join(class_folder, person_folder)
        if not os.path.isdir(person_path):
            continue
        
        # Use feature fusion
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


def process_person_matching(image_feats, person_features, similarity_threshold=0.1):
    """Process person-to-student face matching."""
    matched_students = {}  # Store matched students and their similarity
    person_match_details = {}  # Store per-person matching details
    successful_students = []  # Store IDs of successfully recognized students
    
    print(f"Processing {len(person_features)} persons against {len(image_feats)} known students")
    
    for person_id, person_data in person_features.items():
        face_feat = person_data['embedding']
        
        best_match = None
        best_score = float('-inf')
        
        # Compare with all known student features
        for xh, img_feat in image_feats.items():
            score = np.dot(img_feat, face_feat.T)
            if score > best_score:
                best_score = score
                best_match = xh
        
        # Record per-person matching details
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
        
        # If similarity exceeds threshold, record the match
        if best_score >= similarity_threshold:
            if best_match not in matched_students:
                matched_students[best_match] = []
                successful_students.append(best_match)  # Add to success list
            matched_students[best_match].append({
                'person_id': person_id,
                'similarity_score': float(best_score),
                'quality_score': float(person_data['quality_score']),
                'image_count': person_data['image_count']
            })
    
    # Compute statistics
    matched_persons = sum(1 for details in person_match_details.values() if details['matched'])
    total_persons = len(person_features)
    unique_matched_students = len(matched_students)
    
    # Compute average similarity
    all_similarity_scores = [details['similarity_score'] for details in person_match_details.values() if details['matched']]
    average_similarity = float(np.mean(all_similarity_scores)) if all_similarity_scores else 0.0
    
    # Compute average quality score
    all_quality_scores = [person_data['quality_score'] for person_data in person_features.values()]
    average_quality = float(np.mean(all_quality_scores)) if all_quality_scores else 0.0
    
    # Compute average number of images
    all_image_counts = [person_data['image_count'] for person_data in person_features.values()]
    average_image_count = float(np.mean(all_image_counts)) if all_image_counts else 0.0
    
    # Compute multi-face image stats
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
    """Process a single class folder."""
    attendance_file = os.path.join(class_folder, 'person_attendance_proxyfusion.json')
    
    # Locate the corresponding faces.json file (in the NB116 directory)
    nb116_class_folder = class_folder.replace('NB116_person_spatial', 'NB116')
    faces_file = os.path.join(nb116_class_folder, 'faces.json')
    
    print(f"Processing class folder: {class_folder}")
    print(f"Looking for faces.json in: {faces_file}")
    
    # Check if faces.json exists
    if not os.path.exists(faces_file):
        print(f"faces.json not found in {faces_file}")
        return False
    
    # Load face features
    image_feats = load_face_features(faces_file)
    if image_feats is None:
        print(f"Invalid or empty faces.json in {faces_file}")
        return False

    # Perform feature fusion for each person
    person_features = select_best_face_per_person_with_fusion(class_folder)
    
    if not person_features:
        print(f"No valid person features found in {class_folder}")
        return False
    
    # Perform person matching
    matched_students, person_match_details, successful_students, stats = process_person_matching(image_feats, person_features)
    
    # Output results
    print(f"\n=== 识别统计结果 ===")
    print(f"Total persons: {stats['total_persons']}")
    print(f"Matched persons: {stats['matched_persons']}")
    print(f"Unique matched students: {stats['unique_matched_students']}")
    print(f"Person match rate: {stats['person_match_rate']}%")
    print(f"Student recognition rate: {stats['student_recognition_rate']}%")
    print(f"Average similarity score: {stats['average_similarity']}")
    print(f"Average quality score: {stats['average_quality_score']}")
    print(f"Average image count per person: {stats['average_image_count']}")
    
    print(f"\n=== 图片处理统计 ===")
    print(f"Total images processed: {stats['total_images_processed']}")
    print(f"Single face images used: {stats['total_images_processed'] - stats['total_skipped_multi_face']}")
    print(f"Multi-face images skipped: {stats['total_skipped_multi_face']}")
    print(f"Single face ratio: {stats['single_face_ratio']}%")
    
    # 输出成功识别的学号
    print(f"\n=== 成功识别的学号 ({len(successful_students)}个) ===")
    if successful_students:
        successful_students.sort()  # 排序显示
        for i, student_id in enumerate(successful_students, 1):
            similarity_scores = [match['similarity_score'] for match in matched_students[student_id]]
            max_similarity = max(similarity_scores)
            print(f"{i:2d}. {student_id} (最高相似度: {max_similarity:.4f})")
    else:
        print("无成功识别的学号")
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Processing time: {current_time}")
    
    # 保存考勤结果
    attendance = {
        'class_folder': class_folder,
        'faces_json_source': faces_file,
        'matched_students': {k: v for k, v in matched_students.items()},
        'successful_students': successful_students,  # 添加成功识别的学号列表
        'person_match_details': person_match_details,
        'fusion_method': 'ProxyFusion' if fusion_model is not None else 'Simple Average',
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
    """处理 NB116_person 文件夹中的所有课程"""
    if not os.path.exists(base_folder):
        print(f"Base folder {base_folder} does not exist")
        return
    
    processed_count = 0
    success_count = 0
    
    print(f"Starting to process NB116_person courses in {base_folder}")
    
    # 遍历所有日期文件夹
    for date_folder in os.listdir(base_folder):
        date_path = os.path.join(base_folder, date_folder)
        if not os.path.isdir(date_path):
            continue
        
        print(f"\nProcessing date folder: {date_folder}")
        
        # 遍历所有课程文件夹
        for class_folder in os.listdir(date_path):
            if class_folder.startswith("Class_"):
                class_path = os.path.join(date_path, class_folder)
                if os.path.isdir(class_path):
                    processed_count += 1
                    if process_class_folder(class_path):
                        success_count += 1
    
    # 输出总体统计
    print(f"\n=== Processing Summary ===")
    print(f"Total classes processed: {processed_count}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {processed_count - success_count}")
    if processed_count > 0:
        success_rate = (success_count / processed_count) * 100
        print(f"Success rate: {success_rate:.2f}%")


if __name__ == "__main__":
    # 处理 NB116_person 文件夹
    process_nb116_person_folder(r"dataset\NB116_person_spatial")