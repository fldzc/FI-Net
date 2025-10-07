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
        pose_sim = 1.0 / (1.0 + (yaw_diff + pitch_diff + roll_diff) / 90.0 )
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


def pose_aware_matching(student_features, person_tracker, similarity_threshold=0.3, pose_weight=0.2):
    """Pose-aware matching strategy."""
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
                
                # Compute pose bonus (assume student features are frontal; give higher weight to frontal)
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
            
            # Choose best strategy: max score or weighted average
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
                
                # Combine the two methods
                final_score = 0.7 * max_score + 0.3 * weighted_avg
                
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


def build_enhanced_feature_library(base_folder="NB116_person", training_dates=["508NB116", "515NB116", "522NB116"]):
    """构建增强特征库，使用508、515、522的数据"""
    enhanced_features = defaultdict(list)  # student_id -> list of FaceFeature
    
    print(f"Building enhanced feature library from training dates: {training_dates}")
    
    for date_folder in training_dates:
        date_path = os.path.join(base_folder, date_folder)
        if not os.path.exists(date_path):
            print(f"Training date folder {date_path} does not exist, skipping")
            continue
        
        print(f"Processing training date: {date_folder}")
        
        # 遍历该日期下的所有课程
        for class_folder in os.listdir(date_path):
            if class_folder.startswith("Class_"):
                class_path = os.path.join(date_path, class_folder)
                if not os.path.isdir(class_path):
                    continue
                
                print(f"  Processing class: {class_folder}")
                
                # 加载对应的faces.json
                nb116_class_folder = class_path.replace('NB116_person', 'NB116')
                faces_file = os.path.join(nb116_class_folder, 'faces.json')
                
                if not os.path.exists(faces_file):
                    print(f"    faces.json not found: {faces_file}")
                    continue
                
                # 加载学生特征
                student_features = load_face_features(faces_file)
                if student_features is None:
                    print(f"    Invalid faces.json: {faces_file}")
                    continue
                
                # 提取person特征
                person_tracker = extract_multi_frame_features(class_path)
                
                # 进行匹配，将匹配成功的特征添加到增强库
                matched_students, person_match_details, stats = pose_aware_matching(
                    student_features, person_tracker, similarity_threshold=0.3
                )
                
                print(f"    Matched {stats['unique_matched_students']} students from {stats['total_persons']} persons")
                
                # 将匹配成功的特征添加到增强库
                for student_id, matches in matched_students.items():
                    for match in matches:
                        person_id = match['person_id']
                        person_features = person_tracker.get_person_features(person_id)
                        
                        # 添加该person的所有高质量特征
                        for feature in person_features:
                            if feature.quality_score > 0.5:  # 只保留高质量特征
                                enhanced_features[student_id].append(feature)
    
    # 为每个学生创建动态特征池
    student_feature_pools = {}
    for student_id, features in enhanced_features.items():
        pool = DynamicFeaturePool(max_features=8, replacement_strategy='quality')
        for feature in features:
            pool.add_feature(feature)
        student_feature_pools[student_id] = pool
        print(f"Student {student_id}: {len(pool.get_features())} features in enhanced library")
    
    return student_feature_pools


def process_target_class_with_enhanced_library(class_folder, enhanced_library):
    """使用增强特征库处理目标课程"""
    print(f"Processing target class with enhanced library: {class_folder}")
    
    # 提取目标课程的person特征
    person_tracker = extract_multi_frame_features(class_folder)
    
    if not person_tracker.person_pools:
        print(f"No person features found in {class_folder}")
        return None, None, None
    
    # 使用增强特征库进行匹配
    matched_students = {}
    person_match_details = {}
    
    print(f"Matching {len(person_tracker.person_pools)} persons against enhanced library with {len(enhanced_library)} students")
    
    for person_id, person_pool in person_tracker.person_pools.items():
        person_features = person_pool.get_features()
        
        if not person_features:
            continue
        
        best_match = None
        best_score = float('-inf')
        best_match_details = None
        
        # 与增强库中的每个学生进行匹配
        for student_id, student_pool in enhanced_library.items():
            student_features = student_pool.get_features()
            
            if not student_features:
                continue
            
            # 多对多特征匹配
            max_similarity = 0
            best_person_feat = None
            best_student_feat = None
            
            for person_feat in person_features:
                for student_feat in student_features:
                    # 特征相似度
                    feat_similarity = np.dot(person_feat.embedding, student_feat.embedding)
                    
                    # 姿态相似度加权
                    pose_similarity = person_feat.pose_similarity(student_feat)
                    
                    # 质量加权
                    quality_weight = (person_feat.quality_score + student_feat.quality_score) / 2
                    
                    # 综合分数
                    combined_score = feat_similarity * (1 + 0.2 * pose_similarity) * (0.8 + 0.2 * quality_weight)
                    
                    if combined_score > max_similarity:
                        max_similarity = combined_score
                        best_person_feat = person_feat
                        best_student_feat = student_feat
            
            if max_similarity > best_score:
                best_score = max_similarity
                best_match = student_id
                best_match_details = {
                    'person_feature': best_person_feat,
                    'student_feature': best_student_feat,
                    'similarity': max_similarity
                }
        
        # 记录匹配详情
        person_match_details[person_id] = {
            'best_match': best_match,
            'similarity_score': float(best_score),
            'person_feature_count': len(person_features),
            'best_person_quality': float(best_match_details['person_feature'].quality_score) if best_match_details else 0.0,
            'best_person_pose': best_match_details['person_feature'].get_pose_category() if best_match_details else 'unknown',
            'best_student_quality': float(best_match_details['student_feature'].quality_score) if best_match_details else 0.0,
            'best_student_pose': best_match_details['student_feature'].get_pose_category() if best_match_details else 'unknown',
            'matched': bool(best_score >= 0.3)  # 使用较高的阈值
        }
        
        # 如果匹配成功，记录结果
        if best_score >= 0.3:
            if best_match not in matched_students:
                matched_students[best_match] = []
            matched_students[best_match].append({
                'person_id': person_id,
                'similarity_score': float(best_score),
                'person_quality': float(best_match_details['person_feature'].quality_score),
                'person_pose': best_match_details['person_feature'].get_pose_category()
            })
    
    # 计算统计数据
    matched_persons = sum(1 for details in person_match_details.values() if details['matched'])
    total_persons = len(person_tracker.person_pools)
    unique_matched_students = len(matched_students)
    
    all_similarity_scores = [details['similarity_score'] for details in person_match_details.values() if details['matched']]
    average_similarity = float(np.mean(all_similarity_scores)) if all_similarity_scores else 0.0
    
    all_quality_scores = [details['best_person_quality'] for details in person_match_details.values() if details['best_person_quality'] > 0]
    average_quality = float(np.mean(all_quality_scores)) if all_quality_scores else 0.0
    
    stats = {
        'total_persons': int(total_persons),
        'matched_persons': int(matched_persons),
        'unique_matched_students': int(unique_matched_students),
        'person_match_rate': float(round((matched_persons / total_persons * 100), 2)) if total_persons > 0 else 0.0,
        'student_recognition_rate': float(round((unique_matched_students / len(enhanced_library) * 100), 2)) if len(enhanced_library) > 0 else 0.0,
        'average_similarity': float(round(average_similarity, 4)),
        'average_quality_score': float(round(average_quality, 2)),
        'enhanced_library_size': len(enhanced_library)
    }
    
    return matched_students, person_match_details, stats


def save_performance_log(performance_data, log_file="enhanced_person_performance_log.json"):
    """保存性能日志到文件"""
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        else:
            existing_data = []
        
        existing_data.append(performance_data)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)
        
        print(f"Performance log saved to {log_file}")
    except Exception as e:
        print(f"Error saving performance log: {e}")


def process_529_with_enhanced_features(base_folder="NB116_person"):
    """使用增强特征库处理529NB116的课程"""
    # 开始总体处理性能监控
    performance_monitor.start_monitoring()
    
    print("=== Enhanced Person Attendance Processing ===")
    print("Step 1: Building enhanced feature library from 508, 515, 522...")
    
    # 构建增强特征库
    enhanced_library = build_enhanced_feature_library(base_folder)
    
    if not enhanced_library:
        print("Failed to build enhanced feature library")
        return
    
    print(f"Enhanced library built with {len(enhanced_library)} students")
    
    print("\nStep 2: Processing 529NB116 classes...")
    
    # 处理529NB116的所有课程
    target_date = "529NB116"
    target_path = os.path.join(base_folder, target_date)
    
    if not os.path.exists(target_path):
        print(f"Target date folder {target_path} does not exist")
        return
    
    processed_count = 0
    success_count = 0
    all_results = {}
    
    for class_folder in os.listdir(target_path):
        if class_folder.startswith("Class_"):
            class_path = os.path.join(target_path, class_folder)
            if not os.path.isdir(class_path):
                continue
            
            processed_count += 1
            print(f"\nProcessing {target_date}/{class_folder}...")
            
            # 使用增强特征库处理课程
            matched_students, person_match_details, stats = process_target_class_with_enhanced_library(
                class_path, enhanced_library
            )
            
            if matched_students is not None:
                success_count += 1
                
                # 输出结果
                print(f"  Total persons: {stats['total_persons']}")
                print(f"  Matched persons: {stats['matched_persons']}")
                print(f"  Unique matched students: {stats['unique_matched_students']}")
                print(f"  Person match rate: {stats['person_match_rate']}%")
                print(f"  Student recognition rate: {stats['student_recognition_rate']}%")
                print(f"  Average similarity: {stats['average_similarity']}")
                
                # 保存结果
                attendance_file = os.path.join(class_path, 'enhanced_person_attendance.json')
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                attendance_data = {
                    'class_folder': class_path,
                    'processing_method': 'enhanced_multi_frame_reid',
                    'enhanced_library_info': {
                        'training_dates': ["508NB116", "515NB116", "522NB116"],
                        'total_students_in_library': len(enhanced_library),
                        'total_features_in_library': sum(len(pool.get_features()) for pool in enhanced_library.values())
                    },
                    'matched_students': {k: v for k, v in matched_students.items()},
                    'person_match_details': person_match_details,
                    'stats': {
                        **stats,
                        'processing_time': current_time
                    }
                }
                
                with open(attendance_file, 'w', encoding='utf-8') as f:
                    json.dump(attendance_data, f, indent=4, ensure_ascii=False)
                
                print(f"  Saved to: {attendance_file}")
                all_results[f"{target_date}/{class_folder}"] = stats
            else:
                print(f"  Failed to process {class_path}")
    
    # 停止性能监控
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
    
    # 计算平均识别率
    if all_results:
        avg_person_match_rate = np.mean([r['person_match_rate'] for r in all_results.values()])
        avg_student_recognition_rate = np.mean([r['student_recognition_rate'] for r in all_results.values()])
        avg_similarity = np.mean([r['average_similarity'] for r in all_results.values()])
        
        print(f"\n=== Average Performance ===")
        print(f"Average person match rate: {avg_person_match_rate:.2f}%")
        print(f"Average student recognition rate: {avg_student_recognition_rate:.2f}%")
        print(f"Average similarity score: {avg_similarity:.4f}")
    
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
        'processing_method': 'enhanced_multi_frame_reid',
        'target_date': target_date,
        'enhanced_library_size': len(enhanced_library),
        'processing_summary': {
            'total_classes': processed_count,
            'successful_classes': success_count,
            'failed_classes': processed_count - success_count,
            'success_rate': round((success_count / processed_count) * 100, 2) if processed_count > 0 else 0
        },
        'average_performance': {
            'avg_person_match_rate': round(avg_person_match_rate, 2) if all_results else 0,
            'avg_student_recognition_rate': round(avg_student_recognition_rate, 2) if all_results else 0,
            'avg_similarity_score': round(avg_similarity, 4) if all_results else 0
        } if all_results else {},
        'detailed_results': all_results,
        'overall_performance': overall_performance
    }
    
    save_performance_log(performance_log_data)


if __name__ == "__main__":
    # 使用增强特征库处理529NB116
    process_529_with_enhanced_features("NB116_person")