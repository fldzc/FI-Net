import json
import os
import time
import cv2
import numpy as np
import requests
import ast
from tqdm import tqdm
from insightface.app import FaceAnalysis
from datetime import datetime

app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# Define class time slots
class_times = [
    ('08:00', '08:50'),  # Period 1
    ('08:50', '09:40'),  # Period 2
    ('09:50', '10:40'),  # Period 3
    ('10:40', '11:30'),  # Period 4
    ('11:30', '12:15'),  # Period 5
    ('13:30', '14:20'),  # Period 6
    ('14:20', '15:10'),  # Period 7
    ('15:20', '16:10'),  # Period 8
    ('16:10', '16:55'),  # Period 9
    ('18:30', '19:20'),  # Period 10
    ('19:20', '20:10'),  # Period 11
    ('20:10', '20:55'),  # Period 12
]


def load_face_features(file_path):
    """Load faces.json for each class to get student IDs and face features.
    Handles the case where the "faces" field is a string or null."""
    try:
        with open(file_path, 'r') as f:
            faces_data = json.load(f)
        if faces_data is None:
            return None
        if not faces_data or not isinstance(faces_data, list):
            return None
        image_feats = {}
        for face in faces_data:
            if face.get("faces") is not None:
                if isinstance(face["faces"], str):
                    try:
                        face_list = ast.literal_eval(face["faces"])
                        image_feats[face["xh"]] = np.array(face_list)
                    except (ValueError, SyntaxError) as e:
                        print(f"Error parsing faces feature for student {face['xh']}: {e}")
                elif isinstance(face["faces"], list):
                    image_feats[face["xh"]] = np.array(face["faces"])
        if not image_feats:
            return None
        return image_feats
    except Exception as e:
        return None


def detect_faces_and_landmarks(image):
    if image is None:
        print(f"加载图像失败")
        return []
    results = []
    faces = app.get(image)
    for face in faces:
        bbox = face.bbox
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
        if face.pose[0] > -20:
            results.append({
                'box': (x, y, w, h),
                'embedding': face.embedding,
            })
    return results


# def deduplicate_faces(face_embeddings, similarity_threshold=0.6):
#     """根据人脸特征去重，返回去重后的特征列表"""
#     if not face_embeddings:
#         return []
#
#     unique_faces = [face_embeddings[0]]
#     for embedding in face_embeddings[1:]:
#         is_duplicate = False
#         embedding_norm = embedding / np.linalg.norm(embedding)
#         for unique_embedding in unique_faces:
#             unique_norm = unique_embedding / np.linalg.norm(unique_embedding)
#             similarity = np.dot(unique_norm, embedding_norm.T)
#             if similarity > similarity_threshold:
#                 is_duplicate = True
#                 break
#         if not is_duplicate:
#             unique_faces.append(embedding)
#     return unique_faces


def process_selected_frames(image_feats, frame_folder, similarity_threshold=0.3, dedup_threshold=0.6):
    app.prepare(ctx_id=0, det_thresh=0.4, det_size=(1280, 1280))
    matched_images = set()
    all_face_embeddings = []
    processed_count = 0
    frame_files = [f for f in os.listdir(frame_folder) if f.endswith('.jpg')]
    pbar = tqdm(frame_files, desc="Processing selected frames", unit="frame")

    for frame_file in pbar:
        frame_path = os.path.join(frame_folder, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        face_results = detect_faces_and_landmarks(frame)
        for face_result in face_results:
            x, y, w, h = face_result['box']
            face_feat = face_result['embedding']
            all_face_embeddings.append(face_feat)
            face_feat = face_feat / np.linalg.norm(face_feat)
            best_match = None
            best_score = float('-inf')
            for xh, img_feat in image_feats.items():
                score = np.dot(img_feat, face_feat.T)
                if score > best_score:
                    best_score = score
                    best_match = xh
            if best_score >= similarity_threshold:
                matched_images.add(best_match)
        processed_count += 1

    # Deduplicate detected faces (optional)
    # unique_face_embeddings = deduplicate_faces(all_face_embeddings, dedup_threshold)
    # total_detected_faces = len(unique_face_embeddings)

    # Compute statistics
    total_matched = len(matched_images)
    # total_unmatched = total_detected_faces - total_matched
    recognition_rate = (total_matched / len(image_feats) * 100) if len(image_feats) > 0 else 0
    total_faces_in_json = len(image_feats)

    # Output results
    print(f"\nTotal processed frames: {processed_count}")
    print("Matched students:", matched_images)
    # print(f"Total detected faces (actual attendance, deduplicated): {total_detected_faces}")
    print(f"Total matched images: {total_matched}")
    # print(f"Total unmatched faces: {total_unmatched}")
    print(f"Recognition rate: {recognition_rate:.2f}%")
    print(f"Total faces in faces.json: {total_faces_in_json}")
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Current time: {current_time}")

    # Return stats for saving into attendance.json
    stats = {
        'total_processed_frames': processed_count,
        # 'total_detected_faces': total_detected_faces,
        'total_matched_images': total_matched,
        # 'total_unmatched_faces': total_unmatched,
        'recognition_rate': round(recognition_rate, 2),  # keep two decimals
        'total_faces_in_json': total_faces_in_json,
        'processing_time': current_time
    }
    return matched_images, stats


def process_class_folder(class_folder, classroom, class_end_time):
    class_processed = os.path.join(class_folder, 'class_processed.txt')
    attendance_file = os.path.join(class_folder, 'attendance.json')
    faces_file = os.path.join(class_folder, 'faces.json')
    sent_data_file = os.path.join(class_folder, 'sent_data.json')
    skip_flag_file = os.path.join(class_folder, 'attendance.json')

    if os.path.exists(class_processed) and not os.path.exists(attendance_file) and os.path.exists(faces_file):
        image_feats = load_face_features(faces_file)
        if image_feats is None:
            # Optionally clean frames if needed
            with open(skip_flag_file, 'w') as f:
                f.write(f"Skipped at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} due to invalid faces features.\n")
            return

        matched_images, stats = process_selected_frames(image_feats, class_folder)
        attendance = {
            'class_folder': class_folder,
            'matched_images': list(matched_images),
            'attendance_count': len(matched_images),
            'stats': stats  # 添加统计信息
        }
        with open(attendance_file, 'w') as f:
            json.dump(attendance, f, indent=4)
        print(f"Processed {class_folder} and saved attendance data with stats.")

        sent_data_list = []
        url = "http://10.80.252.104/api/stu/face/add"
        for xh in matched_images:
            data = {
                "actWho": xh,
                "actWhen": class_end_time,
                "actWhere": classroom
            }
            sent_data_list.append(data)
            try:
                response = requests.post(url, json=data)
                # Print raw response for debugging
                print(f"API response for {xh}: Status code={response.status_code}, Content={response.text}")
                
                # Check empty response
                if not response.text.strip():
                    print(f"API returned empty response for {xh}")
                    continue
                    
                # Try parsing JSON
                response_data = response.json()
                if response_data.get("status") == 200 and response_data.get("result") is True:
                    print(f"API success for {xh}: {response_data['message']}")
                else:
                    print(f"API failed for {xh}: {response_data}")
            except requests.exceptions.RequestException as e:
                print(f"Request error for {xh}: {e}")
            except json.JSONDecodeError as e:
                print(f"JSON decode error for {xh}: {e}. Response content: {response.text}")
            except Exception as e:
                print(f"Unexpected error for {xh}: {e}")
        with open(sent_data_file, 'w') as f:
            json.dump(sent_data_list, f, indent=4)

        # Optionally clean frames if needed


def monitor_folders(base_folder):
    while True:
        for root, dirs, files in os.walk(base_folder):
            for dir_name in dirs:
                if dir_name.startswith("Class_"):
                    class_folder = os.path.join(root, dir_name)
                    classroom = os.path.basename(os.path.dirname(class_folder))
                    date_part = os.path.basename(os.path.dirname(os.path.dirname(class_folder))).replace('_streams', '')
                    class_index = int(dir_name.split('_')[1]) - 1
                    if 0 <= class_index < len(class_times):
                        class_end_time = f"{date_part} {class_times[class_index][1]}:00"
                        process_class_folder(class_folder, classroom, class_end_time)
                    else:
                        print(f"Invalid class index for {dir_name}")
        time.sleep(60)


frame_folder_base = "stream"
monitor_folders(frame_folder_base)