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
from collections import defaultdict, deque
import math

app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_thresh=0.4, det_size=(320, 320))

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


class FaceFeature:
    """Face feature entity with embedding, quality score, pose angles, and timestamp."""
    def __init__(self, embedding, quality_score, pose_angles, timestamp, image_path=None):
        self.embedding = embedding / np.linalg.norm(embedding)  # normalize
        self.quality_score = quality_score
        self.pose_angles = pose_angles  # [yaw, pitch, roll]
        self.timestamp = timestamp
        self.image_path = image_path
    
    def get_pose_category(self):
        """Categorize by pose angles."""
        yaw, pitch, roll = self.pose_angles
        
        # Categorize by yaw angle
        if abs(yaw) < 15:
            return 'frontal'
        elif abs(yaw) < 45:
            return 'semi_profile'
        else:
            return 'profile'
    
    def pose_similarity(self, other_feature):
        """Compute pose similarity."""
        yaw_diff = abs(self.pose_angles[0] - other_feature.pose_angles[0])
        pitch_diff = abs(self.pose_angles[1] - other_feature.pose_angles[1])
        roll_diff = abs(self.pose_angles[2] - other_feature.pose_angles[2])
        
        # Pose similarity score (smaller angle difference => higher similarity)
        pose_sim = 1.0 / (1.0 + (yaw_diff + pitch_diff + roll_diff) / 90.0)
        return pose_sim


class DynamicFeaturePool:
    """Dynamic feature pool manager."""
    def __init__(self, max_features=5, replacement_strategy='quality'):
        self.max_features = max_features
        self.replacement_strategy = replacement_strategy  # 'fifo' or 'quality'
        self.features = deque(maxlen=max_features if replacement_strategy == 'fifo' else None)
    
    def add_feature(self, feature):
        """Add a new feature."""
        if self.replacement_strategy == 'fifo':
            self.features.append(feature)
        else:  # quality-based replacement
            if len(self.features) < self.max_features:
                self.features.append(feature)
            else:
                # Replace the feature with the lowest quality
                min_quality_idx = min(range(len(self.features)), 
                                     key=lambda i: self.features[i].quality_score)
                if feature.quality_score > self.features[min_quality_idx].quality_score:
                    self.features[min_quality_idx] = feature
    
    def get_features(self):
        """Get all features."""
        return list(self.features)
    
    def get_best_features_by_pose(self, target_pose_category, top_k=3):
        """Get best features by pose category."""
        matching_features = [f for f in self.features if f.get_pose_category() == target_pose_category]
        if not matching_features:
            # If no matching pose, return highest-quality features
            matching_features = list(self.features)
        
        # Sort by quality score
        matching_features.sort(key=lambda f: f.quality_score, reverse=True)
        return matching_features[:top_k]


class PersonTracker:
    """Person tracker used to associate the same individual across frames."""
    def __init__(self):
        self.person_pools = {}  # person_id -> DynamicFeaturePool
        self.person_reid_features = {}  # person_id -> ReID feature (for cross-frame association)
    
    def add_person_feature(self, person_id, face_feature, reid_feature=None):
        """Add a face feature for the given person."""
        if person_id not in self.person_pools:
            self.person_pools[person_id] = DynamicFeaturePool()
        
        self.person_pools[person_id].add_feature(face_feature)
        
        if reid_feature is not None:
            self.person_reid_features[person_id] = reid_feature
    
    def get_person_features(self, person_id):
        """Get all features for the specified person."""
        if person_id in self.person_pools:
            return self.person_pools[person_id].get_features()
        return []
    
    def associate_across_frames(self, new_reid_feature, similarity_threshold=0.7):
        """Associate across frames to find the most similar known person."""
        best_match = None
        best_score = -1
        
        for person_id, reid_feat in self.person_reid_features.items():
            score = np.dot(new_reid_feature, reid_feat)
            if score > best_score and score > similarity_threshold:
                best_score = score
                best_match = person_id
        
        return best_match, best_score


def load_face_features(file_path):
    """Load faces.json for each class to get student IDs and face features."""
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


def get_face_analysis_from_insightface(image):
    """Use insightface to get face analysis results."""
    if image is None:
        return None
    
    faces = app.get(image)
    if not faces:
        return None
    
    # 选择置信度最高的人脸
    best_face = max(faces, key=lambda f: f.det_score)
    return best_face


def extract_multi_frame_features(class_folder):
    """Extract multi-frame features and build a dynamic feature pool."""
    person_tracker = PersonTracker()
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
        
        # Extract features for each image
        for image_file in image_files:
            image_path = os.path.join(person_path, image_file)
            image = cv2.imread(image_path)
            
            if image is None:
                continue
            
            # Use insightface to get face analysis results
            face_analysis = get_face_analysis_from_insightface(image)
            
            if face_analysis is not None:
                # Create face feature object
                face_feature = FaceFeature(
                    embedding=face_analysis.embedding,
                    quality_score=float(face_analysis.det_score),
                    pose_angles=[float(face_analysis.pose[0]), float(face_analysis.pose[1]), float(face_analysis.pose[2])],
                    timestamp=time.time(),
                    image_path=image_path
                )
                
                # Add to person tracker
                person_tracker.add_person_feature(person_folder, face_feature)
    
    return person_tracker


def multi_frame_fusion_matching(student_features, person_tracker, similarity_threshold=0.3):
    """Multi-frame fusion matching strategy."""
    matched_students = {}
    person_match_details = {}
    
    print(f"Processing {len(person_tracker.person_pools)} persons against {len(student_features)} known students")
    
    for person_id, feature_pool in person_tracker.person_pools.items():
        person_features = feature_pool.get_features()
        
        if not person_features:
            continue
        
        best_match = None
        best_score = float('-inf')
        best_feature_info = None
        
        # Match against each student
        for xh, student_feat in student_features.items():
            # Multi-feature fusion matching strategy
            feature_scores = []
            
            for person_feat in person_features:
                # Compute feature similarity
                feat_similarity = np.dot(student_feat, person_feat.embedding)
                
                # Compute pose bonus (higher for frontal)
                pose_bonus = 1.0
                if person_feat.get_pose_category() == 'frontal':
                    pose_bonus = 1.2
                elif person_feat.get_pose_category() == 'semi_profile':
                    pose_bonus = 1.0
                else:  # profile
                    pose_bonus = 0.8
                
                # Quality weight
                quality_weight = min(1.2, person_feat.quality_score + 0.2)
                
                # Final combined score
                final_score = feat_similarity * pose_bonus * quality_weight
                feature_scores.append({
                    'score': final_score,
                    'feature': person_feat,
                    'raw_similarity': feat_similarity
                })
            
            # Fusion strategy: max score + weighted average
            if feature_scores:
                # Method 1: take max score
                max_score_info = max(feature_scores, key=lambda x: x['score'])
                max_score = max_score_info['score']
                
                # Method 2: weighted average (higher quality => larger weight)
                total_weight = sum(info['feature'].quality_score for info in feature_scores)
                if total_weight > 0:
                    weighted_avg = sum(info['score'] * info['feature'].quality_score for info in feature_scores) / total_weight
                else:
                    weighted_avg = 0
                
                # Method 3: top-k average (average of top 3 scores)
                top_k_scores = sorted([info['score'] for info in feature_scores], reverse=True)[:3]
                top_k_avg = np.mean(top_k_scores) if top_k_scores else 0
                
                # Combine three methods
                final_score = 0.5 * max_score + 0.3 * weighted_avg + 0.2 * top_k_avg
                
                if final_score > best_score:
                    best_score = final_score
                    best_match = xh
                    best_feature_info = max_score_info
        
        # 记录匹配详情
        person_match_details[person_id] = {
            'best_match': best_match,
            'similarity_score': float(best_score),
            'raw_similarity': float(best_feature_info['raw_similarity']) if best_feature_info else 0.0,
            'best_feature_path': best_feature_info['feature'].image_path if best_feature_info else None,
            'best_feature_quality': float(best_feature_info['feature'].quality_score) if best_feature_info else 0.0,
            'best_feature_pose': best_feature_info['feature'].get_pose_category() if best_feature_info else 'unknown',
            'total_features_used': len(person_features),
            'matched': bool(best_score >= similarity_threshold)
        }
        
        # 如果相似度超过阈值，记录匹配
        if best_score >= similarity_threshold:
            if best_match not in matched_students:
                matched_students[best_match] = []
            matched_students[best_match].append({
                'person_id': person_id,
                'similarity_score': float(best_score),
                'raw_similarity': float(best_feature_info['raw_similarity']) if best_feature_info else 0.0,
                'quality_score': float(best_feature_info['feature'].quality_score) if best_feature_info else 0.0,
                'pose_category': best_feature_info['feature'].get_pose_category() if best_feature_info else 'unknown'
            })
    
    # 计算统计数据
    matched_persons = sum(1 for details in person_match_details.values() if details['matched'])
    total_persons = len(person_tracker.person_pools)
    unique_matched_students = len(matched_students)
    
    # 计算平均相似度
    all_similarity_scores = [details['similarity_score'] for details in person_match_details.values() if details['matched']]
    average_similarity = float(np.mean(all_similarity_scores)) if all_similarity_scores else 0.0
    
    # 计算平均质量分数
    all_quality_scores = [details['best_feature_quality'] for details in person_match_details.values() if details['best_feature_quality'] > 0]
    average_quality = float(np.mean(all_quality_scores)) if all_quality_scores else 0.0
    
    return matched_students, person_match_details, {
        'total_persons': int(total_persons),
        'matched_persons': int(matched_persons),
        'unique_matched_students': int(unique_matched_students),
        'person_match_rate': float(round((matched_persons / total_persons * 100), 2)) if total_persons > 0 else 0.0,
        'student_recognition_rate': float(round((unique_matched_students / len(student_features) * 100), 2)) if len(student_features) > 0 else 0.0,
        'average_similarity': float(round(average_similarity, 4)),
        'average_quality_score': float(round(average_quality, 2))
    }


def process_class_folder_multiframe(class_folder):
    """使用多帧融合策略处理单个课程文件夹"""
    # 开始单个课程处理性能监控
    class_monitor = PerformanceMonitor()
    class_monitor.start_monitoring()
    
    attendance_file = os.path.join(class_folder, 'multiframe_person_attendance.json')
    
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

    # 提取多帧特征
    person_tracker = extract_multi_frame_features(class_folder)
    
    if not person_tracker.person_pools:
        print(f"No valid person features found in {class_folder}")
        class_monitor.stop_monitoring()
        return False
    
    # 使用多帧融合匹配
    matched_students, person_match_details, stats = multi_frame_fusion_matching(
        image_feats, person_tracker, similarity_threshold=0.3
    )
    
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
        'processing_method': 'multi_frame_fusion',
        'matched_students': {k: v for k, v in matched_students.items()},
        'person_match_details': person_match_details,
        'stats': {
            **stats,
            'total_faces_in_json': len(image_feats),
            'processing_time': current_time
        },
        'class_processing_performance': class_performance
    }
    
    with open(attendance_file, 'w', encoding='utf-8') as f:
        json.dump(attendance, f, indent=4, ensure_ascii=False)
    
    print(f"Saved attendance data to {attendance_file}")
    return True


def save_performance_log(performance_data, log_file="multiframe_performance_log.json"):
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


def process_nb116_person_folder_multiframe(base_folder="NB116_person"):
    """使用多帧融合策略处理 NB116_person 文件夹中的所有课程"""
    if not os.path.exists(base_folder):
        print(f"Base folder {base_folder} does not exist")
        return
    
    # 开始总体处理性能监控
    performance_monitor.start_monitoring()
    
    processed_count = 0
    success_count = 0
    
    print(f"Starting multi-frame fusion processing for NB116_person courses in {base_folder}")
    
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
                    if process_class_folder_multiframe(class_path):
                        success_count += 1
    
    # 停止总体处理性能监控
    performance_monitor.stop_monitoring()
    overall_performance = performance_monitor.get_performance_stats()
    
    # 输出总体统计
    print(f"\n=== Multi-Frame Fusion Processing Summary ===")
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
        'processing_method': 'multi_frame_fusion',
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
    # 使用多帧融合策略处理 NB116_person 文件夹
    process_nb116_person_folder_multiframe("NB116_person")