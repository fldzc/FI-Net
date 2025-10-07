import json
import os
import cv2
import numpy as np
import ast
from tqdm import tqdm
from insightface.app import FaceAnalysis
from datetime import datetime

app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_thresh=0.4, det_size=(320, 320))


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
                        # Check if the string is empty or contains only whitespace
                        if not face["face"].strip():
                            print(f"Empty faces string for student {face.get('xh', 'unknown')}, skipping")
                            continue
                        try:
                            face_list = ast.literal_eval(face["face"])
                            # Normalize the feature vector
                            face_array = np.array(face_list)
                            if face_array.size > 0:  # Ensure the array is not empty
                                image_feats[face["xh"]] = face_array / np.linalg.norm(face_array)
                        except (ValueError, SyntaxError) as e:
                            print(f"Error parsing faces feature for student {face.get('xh', 'unknown')}: {e}, skipping")
                            continue
                    elif isinstance(face["face"], list):
                        # Normalize the feature vector
                        face_array = np.array(face["face"])
                        if face_array.size > 0:  # Ensure the array is not empty
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
        if face.pose[0] > -20:  # Filter side faces
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
        skipped_multi_face = 0
        
        # Iterate all images of this person and select the highest quality one
        for image_file in image_files:
            image_path = os.path.join(person_path, image_file)
            image = cv2.imread(image_path)
            
            if image is None:
                continue
                
            # Check the number of faces; skip images with multiple faces
            faces = app.get(image)
            if len(faces) > 1:
                print(f"跳过多人脸图片: {image_file} (检测到 {len(faces)} 张人脸)")
                skipped_multi_face += 1
                continue
            elif len(faces) == 0:
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
            print(f"Person {person_folder}: 总图片 {len(image_files)}, 跳过多人脸 {skipped_multi_face}, 选中最优图片: {os.path.basename(best_image_path)}")
        else:
            print(f"Person {person_folder}: 总图片 {len(image_files)}, 跳过多人脸 {skipped_multi_face}, 无有效单人脸图片")
    
    return best_faces


def process_person_matching(image_feats, best_faces, similarity_threshold=0.2):
    """Process person-to-student face matching."""
    matched_students = {}  # Store matched students and their similarity scores
    person_match_details = {}  # Store matching details for each person
    successful_students = []  # Store IDs of successfully recognized students
    
    print(f"Processing {len(best_faces)} persons against {len(image_feats)} known students")
    
    for person_id, person_data in best_faces.items():
        face_feat = person_data['embedding']
        
        best_match = None
        best_score = float('-inf')
        
        # Compare with all known student features
        for xh, img_feat in image_feats.items():
            score = np.dot(img_feat, face_feat.T)
            if score > best_score:
                best_score = score
                best_match = xh
        
        # Record matching details for each person
        person_match_details[person_id] = {
            'best_match': best_match,
            'similarity_score': float(best_score),
            'image_path': person_data['image_path'],
            'quality_score': float(person_data['quality_score']),
            'matched': bool(best_score >= similarity_threshold)
        }
        
        # If similarity exceeds the threshold, record the match
        if best_score >= similarity_threshold:
            if best_match not in matched_students:
                matched_students[best_match] = []
                successful_students.append(best_match)  # Add to the successfully recognized list
            matched_students[best_match].append({
                'person_id': person_id,
                'similarity_score': float(best_score),
                'quality_score': float(person_data['quality_score'])
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
    
    return matched_students, person_match_details, successful_students, {
        'total_persons': int(total_persons),
        'matched_persons': int(matched_persons),
        'unique_matched_students': int(unique_matched_students),
        'person_match_rate': float(round((matched_persons / total_persons * 100), 2)) if total_persons > 0 else 0.0,
        'student_recognition_rate': float(round((unique_matched_students / len(image_feats) * 100), 2)) if len(image_feats) > 0 else 0.0,
        'average_similarity': float(round(average_similarity, 4)),
        'average_quality_score': float(round(average_quality, 2))
    }


def process_class_folder(class_folder):
    """Process a single class folder."""
    attendance_file = os.path.join(class_folder, 'person_attendance_best_face.json')
    
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

    # Select the best face for each person
    best_faces = select_best_face_per_person(class_folder)
    
    if not best_faces:
        print(f"No valid person faces found in {class_folder}")
        return False
    
    # Perform person matching
    matched_students, person_match_details, successful_students, stats = process_person_matching(image_feats, best_faces)
    
    # Output results
    print(f"\n=== 识别统计结果 ===")
    print(f"Total persons: {stats['total_persons']}")
    print(f"Matched persons: {stats['matched_persons']}")
    print(f"Unique matched students: {stats['unique_matched_students']}")
    print(f"Person match rate: {stats['person_match_rate']}%")
    print(f"Student recognition rate: {stats['student_recognition_rate']}%")
    print(f"Average similarity score: {stats['average_similarity']}")
    print(f"Average quality score: {stats['average_quality_score']}")
    
    # Output successfully recognized student IDs
    print(f"\n=== 成功识别的学号 ({len(successful_students)}个) ===")
    if successful_students:
        successful_students.sort()  # Sort for display
        for i, student_id in enumerate(successful_students, 1):
            similarity_scores = [match['similarity_score'] for match in matched_students[student_id]]
            max_similarity = max(similarity_scores)
            print(f"{i:2d}. {student_id} (最高相似度: {max_similarity:.4f})")
    else:
        print("无成功识别的学号")
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Processing time: {current_time}")
    
    # Save attendance results
    attendance = {
        'class_folder': class_folder,
        'faces_json_source': faces_file,
        'matched_students': {k: v for k, v in matched_students.items()},
        'successful_students': successful_students,  # List of successfully recognized student IDs
        'person_match_details': person_match_details,
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
    """Process all classes under the NB116_person folder."""
    if not os.path.exists(base_folder):
        print(f"Base folder {base_folder} does not exist")
        return
    
    processed_count = 0
    success_count = 0
    
    print(f"Starting to process NB116_person courses in {base_folder}")
    
    # Iterate over all date folders
    for date_folder in os.listdir(base_folder):
        date_path = os.path.join(base_folder, date_folder)
        if not os.path.isdir(date_path):
            continue
        
        print(f"\nProcessing date folder: {date_folder}")
        
        # Iterate over all class folders
        for class_folder in os.listdir(date_path):
            if class_folder.startswith("Class_"):
                class_path = os.path.join(date_path, class_folder)
                if os.path.isdir(class_path):
                    processed_count += 1
                    if process_class_folder(class_path):
                        success_count += 1
    
    # Output overall summary
    print(f"\n=== Processing Summary ===")
    print(f"Total classes processed: {processed_count}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {processed_count - success_count}")
    if processed_count > 0:
        success_rate = (success_count / processed_count) * 100
        print(f"Success rate: {success_rate:.2f}%")


if __name__ == "__main__":
    # Process the NB116_person folder
    process_nb116_person_folder(r"dataset/NB116_person_spatial")