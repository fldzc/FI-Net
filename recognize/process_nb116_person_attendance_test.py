import json
import os
import cv2
import numpy as np
import ast
import time
import psutil
import threading
from tqdm import tqdm
from insightface.app import FaceAnalysis
from datetime import datetime

app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_thresh=0.35, det_size=(320, 320))

# Performance monitoring class
class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.peak_memory = 0
        self.cpu_usage_samples = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.peak_memory = 0
        self.cpu_usage_samples = []
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.end_time = time.time()
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
    
    def _monitor_resources(self):
        """Monitor system resource usage."""
        process = psutil.Process()
        while self.monitoring:
            try:
                # Monitor memory usage
                memory_info = process.memory_info()
                current_memory = memory_info.rss / 1024 / 1024  # MB
                self.peak_memory = max(self.peak_memory, current_memory)
                
                # Monitor CPU usage
                cpu_percent = process.cpu_percent()
                self.cpu_usage_samples.append(cpu_percent)
                
                time.sleep(0.5)  # sample every 0.5 seconds
            except:
                break
    
    def get_performance_stats(self):
        """Get performance statistics."""
        if self.start_time is None or self.end_time is None:
            return None
        
        processing_time = self.end_time - self.start_time
        avg_cpu_usage = np.mean(self.cpu_usage_samples) if self.cpu_usage_samples else 0
        max_cpu_usage = max(self.cpu_usage_samples) if self.cpu_usage_samples else 0
        
        return {
            'processing_time_seconds': round(processing_time, 2),
            'processing_time_formatted': f"{int(processing_time // 60)}m {int(processing_time % 60)}s",
            'peak_memory_mb': round(self.peak_memory, 2),
            'average_cpu_usage_percent': round(avg_cpu_usage, 2),
            'max_cpu_usage_percent': round(max_cpu_usage, 2),
            'cpu_samples_count': len(self.cpu_usage_samples)
        }

# Global performance monitor
performance_monitor = PerformanceMonitor()


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
                        # 检查字符串是否为空或只包含空白字符
                        if not face["face"].strip():
                            print(f"Empty faces string for student {face.get('xh', 'unknown')}, skipping")
                            continue
                        try:
                            face_list = ast.literal_eval(face["face"])
                            # 归一化特征向量
                            face_array = np.array(face_list)
                            if face_array.size > 0:  # 确保数组不为空
                                image_feats[face["xh"]] = face_array / np.linalg.norm(face_array)
                        except (ValueError, SyntaxError) as e:
                            print(f"Error parsing faces feature for student {face.get('xh', 'unknown')}: {e}, skipping")
                            continue
                    elif isinstance(face["face"], list):
                        # 归一化特征向量
                        face_array = np.array(face["face"])
                        if face_array.size > 0:  # 确保数组不为空
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
    """Use insightface to compute a quality score combining confidence and face size."""
    if image is None:
        return 0, None
    
    faces = app.get(image)
    if not faces:
        return 0, None
    
    # Compute combined quality score: confidence * face-size weight
    def calculate_quality_score(face):
        bbox = face.bbox
        face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        # Normalize face area (assume max image size 1280x1280)
        normalized_area = min(face_area / (1280 * 1280), 1.0)
        # Combined score: 70% confidence + 30% area weight
        return face.det_score * 0.7 + normalized_area * 0.3
    
    # Select the face with the highest combined quality score
    best_face = max(faces, key=calculate_quality_score)
    quality_score = calculate_quality_score(best_face)
    return float(quality_score), best_face


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
        if face.pose[0] > -30:  # Relax side-face filtering to include more angles
            results.append({
                'box': (x, y, w, h),
                'embedding': face.embedding,
                'det_score': float(face.det_score),  # Add confidence score
            })
    return results


def select_best_face_per_person(class_folder):
    """Select the highest-quality face image for each person."""
    best_faces = {}
    person_folders = [f for f in os.listdir(class_folder) if f.startswith('person_')]
    
    print(f"Found {len(person_folders)} person folders in {class_folder}")
    
    for person_folder in person_folders:
        person_path = os.path.join(class_folder, person_folder)
        if not os.path.isdir(person_path):
            continue
            
        # Get all images in this person's folder
        image_files = [f for f in os.listdir(person_path) if f.endswith('.jpg')]
        
        if not image_files:
            continue
            
        best_quality = -1
        best_image_path = None
        best_embedding = None
        
        # Iterate all images of this person and select the highest quality one
        for image_file in image_files:
            image_path = os.path.join(person_path, image_file)
            image = cv2.imread(image_path)
            
            if image is None:
                continue
                
            # Use insightface to obtain face quality and features
            quality, best_face = get_face_quality_from_insightface(image)
            
            if best_face is not None and quality > best_quality:
                best_quality = quality
                best_image_path = image_path
                best_embedding = best_face.embedding
        
        if best_embedding is not None:
            # Normalize the feature vector
            best_embedding = best_embedding / np.linalg.norm(best_embedding)
            best_faces[person_folder] = {
                'embedding': best_embedding,
                'image_path': best_image_path,
                'quality_score': best_quality
            }
    
    return best_faces


def process_person_matching(image_feats, best_faces, similarity_threshold=0.25):
    """Process person-to-student face matching."""
    matched_students = {}  # Store matched students and their similarity
    person_match_details = {}  # Store per-person matching details
    
    print(f"Processing {len(best_faces)} persons against {len(image_feats)} known students")
    
    for person_id, person_data in best_faces.items():
        face_feat = person_data['embedding']
        quality_score = person_data['quality_score']
        
        # Compute all similarity scores
        similarity_scores = []
        for xh, img_feat in image_feats.items():
            score = np.dot(img_feat, face_feat.T)
            # Weighted similarity combining quality score
            weighted_score = score * (0.8 + 0.2 * min(quality_score, 1.0))
            similarity_scores.append((xh, score, weighted_score))
        
        # Sort by weighted score
        similarity_scores.sort(key=lambda x: x[2], reverse=True)
        
        best_match = similarity_scores[0][0] if similarity_scores else None
        best_score = similarity_scores[0][1] if similarity_scores else float('-inf')
        weighted_best_score = similarity_scores[0][2] if similarity_scores else float('-inf')
        
        # Dynamic threshold: stricter for higher-quality faces
        dynamic_threshold = similarity_threshold + (quality_score - 0.5) * 0.1
        dynamic_threshold = max(0.15, min(dynamic_threshold, 0.4))  # 限制在合理范围内
        
        # Record per-person matching details
        person_match_details[person_id] = {
            'best_match': best_match,
            'similarity_score': float(best_score),
            'weighted_score': float(weighted_best_score),
            'dynamic_threshold': float(dynamic_threshold),
            'image_path': person_data['image_path'],
            'quality_score': float(quality_score),
            'matched': bool(weighted_best_score >= dynamic_threshold)
        }
        
        # If weighted similarity exceeds dynamic threshold, record match
        if weighted_best_score >= dynamic_threshold:
            if best_match not in matched_students:
                matched_students[best_match] = []
            matched_students[best_match].append({
                'person_id': person_id,
                'similarity_score': float(best_score),
                'weighted_score': float(weighted_best_score),
                'quality_score': float(quality_score)
            })
    
    # Compute statistics
    matched_persons = sum(1 for details in person_match_details.values() if details['matched'])
    total_persons = len(best_faces)
    unique_matched_students = len(matched_students)
    
    # Compute average similarity
    all_similarity_scores = [details['similarity_score'] for details in person_match_details.values() if details['matched']]
    average_similarity = float(np.mean(all_similarity_scores)) if all_similarity_scores else 0.0
    
    # Compute average quality score
    all_quality_scores = [person_data['quality_score'] for person_data in best_faces.values()]
    average_quality = float(np.mean(all_quality_scores)) if all_quality_scores else 0.0
    
    return matched_students, person_match_details, {
        'total_persons': int(total_persons),
        'matched_persons': int(matched_persons),
        'unique_matched_students': int(unique_matched_students),
        'person_match_rate': float(round((matched_persons / total_persons * 100), 2)) if total_persons > 0 else 0.0,
        'student_recognition_rate': float(round((unique_matched_students / len(image_feats) * 100), 2)) if len(image_feats) > 0 else 0.0,
        'average_similarity': float(round(average_similarity, 4)),
        'average_quality_score': float(round(average_quality, 2))
    }


def process_class_folder(class_folder):
    """处理单个课程文件夹"""
    # 开始单个课程处理性能监控
    class_monitor = PerformanceMonitor()
    class_monitor.start_monitoring()
    
    attendance_file = os.path.join(class_folder, 'person_attendance.json')
    
    # 查找对应的faces.json文件（在NB116目录中）
    nb116_class_folder = class_folder.replace('NB116_person', 'NB116')
    faces_file = os.path.join(nb116_class_folder, 'faces.json')
    
    print(f"Processing class folder: {class_folder}")
    print(f"Looking for faces.json in: {faces_file}")
    
    # 检查 faces.json 是否存在
    if not os.path.exists(faces_file):
        print(f"faces.json not found in {faces_file}")
        class_monitor.stop_monitoring()
        return False
    
    # 加载人脸特征
    image_feats = load_face_features(faces_file)
    if image_feats is None:
        print(f"Invalid or empty faces.json in {faces_file}")
        class_monitor.stop_monitoring()
        return False

    # 为每个person选择最优人脸
    best_faces = select_best_face_per_person(class_folder)
    
    if not best_faces:
        print(f"No valid person faces found in {class_folder}")
        class_monitor.stop_monitoring()
        return False
    
    # 处理person匹配（使用优化后的阈值）
    matched_students, person_match_details, stats = process_person_matching(image_feats, best_faces, similarity_threshold=0.25)
    
    # 停止单个课程处理性能监控
    class_monitor.stop_monitoring()
    class_performance = class_monitor.get_performance_stats()
    
    # 输出结果
    print(f"\nTotal persons: {stats['total_persons']}")
    print(f"Matched persons: {stats['matched_persons']}")
    print(f"Unique matched students: {stats['unique_matched_students']}")
    print(f"Person match rate: {stats['person_match_rate']}%")
    print(f"Student recognition rate: {stats['student_recognition_rate']}%")
    print(f"Average similarity score: {stats['average_similarity']}")
    print(f"Average quality score: {stats['average_quality_score']}")
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Processing time: {current_time}")
    
    # 保存考勤结果
    attendance = {
        'class_folder': class_folder,
        'faces_json_source': faces_file,
        'matched_students': {k: v for k, v in matched_students.items()},
        'person_match_details': person_match_details,
        'stats': {
            **stats,
            'total_faces_in_json': len(image_feats),
            'processing_time': current_time
        },
        'class_processing_performance': class_performance
    }
    
    with open(attendance_file, 'w') as f:
        json.dump(attendance, f, indent=4, ensure_ascii=False)
    
    print(f"Saved attendance data to {attendance_file}")
    return True


def save_performance_log(performance_data, log_file="person_performance_log.json"):
    """保存性能日志到文件"""
    try:
        # 如果文件存在，读取现有数据
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        else:
            existing_data = []
        
        # 添加新的性能数据
        existing_data.append(performance_data)
        
        # 保存到文件
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)
        
        print(f"Performance log saved to {log_file}")
    except Exception as e:
        print(f"Error saving performance log: {e}")


def process_nb116_person_folder(base_folder="NB116_person"):
    """处理 NB116_person 文件夹中的所有课程"""
    if not os.path.exists(base_folder):
        print(f"Base folder {base_folder} does not exist")
        return
    
    # 开始总体处理性能监控
    performance_monitor.start_monitoring()
    
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
    
    # 停止总体处理性能监控
    performance_monitor.stop_monitoring()
    overall_performance = performance_monitor.get_performance_stats()
    
    # 输出总体统计
    print(f"\n=== Processing Summary ===")
    print(f"Total classes processed: {processed_count}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {processed_count - success_count}")
    if processed_count > 0:
        success_rate = (success_count / processed_count) * 100
        print(f"Success rate: {success_rate:.2f}%")
    
    # 输出性能统计
    if overall_performance:
        print(f"\n=== Performance Statistics ===")
        print(f"Total processing time: {overall_performance['processing_time_formatted']}")
        print(f"Peak memory usage: {overall_performance['peak_memory_mb']} MB")
        print(f"Average CPU usage: {overall_performance['average_cpu_usage_percent']}%")
        print(f"Max CPU usage: {overall_performance['max_cpu_usage_percent']}%")
    
    # 保存性能日志
    performance_log_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'base_folder': base_folder,
        'processing_summary': {
            'total_classes': processed_count,
            'successful_classes': success_count,
            'failed_classes': processed_count - success_count,
            'success_rate': round((success_count / processed_count) * 100, 2) if processed_count > 0 else 0
        },
        'overall_performance': overall_performance
    }
    
    save_performance_log(performance_log_data)


if __name__ == "__main__":
    # 处理 NB116_person 文件夹
    process_nb116_person_folder("NB116_person")