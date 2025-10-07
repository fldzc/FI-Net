import argparse
import os
import cv2
import numpy as np
import glob
import json
import shutil
from ultralytics import YOLO
from tqdm import tqdm
from collections import defaultdict
import re
from insightface.app import FaceAnalysis

# 初始化人脸检测
face_app = FaceAnalysis(allowed_modules=['detection'], name='antelopev2', 
                     providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_thresh=0.6, det_size=(320, 320))

def detect_persons_yolo(image, yolo_model, conf=0.2, iou=0.2, max_persons=100, face_detection=True):
    """
    使用YOLO检测图像中的人体并返回边界框信息
    """
    # 推理
    results = yolo_model.predict(
        source=image,
        conf=conf,
        iou=iou,
        classes=[0],  # 只检测人
        max_det=max_persons,
        agnostic_nms=False,
        imgsz=1920,
        augment=True
    )[0]
    
    # 如果没有检测到人体，返回空列表
    if len(results.boxes) == 0:
        return [], []
    
    # 获取所有人体边界框
    person_images = []
    person_boxes = []
    
    # 处理检测结果
    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0])
        conf_score = float(box.conf[0])
        
        if cls_id == 0:  # COCO 中 person 的 class id 是 0
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 裁剪人体区域
            person_image = image[y1:y2, x1:x2]
            
            # 如果启用人脸检测，检查是否包含人脸
            has_face = True  # 默认为True，如果不启用人脸检测
            if face_detection and person_image.size > 0:
                # 检测人脸
                faces = face_app.get(person_image)
                has_face = len(faces) > 0
            
            # 只有在不启用人脸检测或检测到人脸时才添加人体
            if has_face:
                person_images.append(person_image)
                person_boxes.append([x1, y1, x2, y2, conf_score])
    
    return person_images, person_boxes

def parse_person_image_name(image_name):
    """
    解析person图片文件名，提取frame信息
    
    Args:
        image_name: 如 'frame_10000_faces_17_person_0_0.jpg'
    
    Returns:
        tuple: (frame_name, person_id, image_index) 或 (None, None, None)
    """
    # 移除扩展名
    base_name = os.path.splitext(image_name)[0]
    
    # 使用正则表达式匹配文件名模式
    pattern = r'frame_(\d+)_faces_(\d+)_person_(\d+)_(\d+)'
    match = re.match(pattern, base_name)
    
    if match:
        frame_num = match.group(1)
        faces_num = match.group(2)
        person_id = int(match.group(3))
        image_index = int(match.group(4))
        
        frame_name = f"frame_{frame_num}_faces_{faces_num}.jpg"
        return frame_name, person_id, image_index
    
    return None, None, None

def find_matching_person_box(person_image, person_boxes, person_images, similarity_threshold=0.95):
    """
    通过图像相似度找到匹配的边界框
    
    Args:
        person_image: 要匹配的person图像
        person_boxes: 检测到的所有边界框
        person_images: 检测到的所有person图像
        similarity_threshold: 相似度阈值
    
    Returns:
        tuple: (匹配的边界框, 相似度) 或 (None, 0)
    """
    if not person_images:
        return None, 0
    
    best_match_idx = -1
    best_similarity = 0
    
    # 调整person_image大小以便比较
    target_height, target_width = person_image.shape[:2]
    
    for i, detected_image in enumerate(person_images):
        # 调整检测到的图像大小
        detected_resized = cv2.resize(detected_image, (target_width, target_height))
        
        # 计算结构相似性
        # 转换为灰度图
        gray1 = cv2.cvtColor(person_image, cv2.COLOR_BGR2GRAY) if len(person_image.shape) == 3 else person_image
        gray2 = cv2.cvtColor(detected_resized, cv2.COLOR_BGR2GRAY) if len(detected_resized.shape) == 3 else detected_resized
        
        # 计算归一化互相关
        result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
        similarity = np.max(result)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match_idx = i
    
    if best_similarity >= similarity_threshold and best_match_idx >= 0:
        return person_boxes[best_match_idx], best_similarity
    
    return None, best_similarity

def process_single_course(nb116_course_path, nb116_person_course_path, yolo_model, args):
    """
    处理单个课程，为person图片添加位置信息
    
    Args:
        nb116_course_path: 原始NB116课程路径
        nb116_person_course_path: NB116_person课程路径
        yolo_model: YOLO模型
        args: 命令行参数
    
    Returns:
        dict: 处理结果统计
    """
    print(f"\n处理课程: {nb116_person_course_path}")
    
    # 获取所有person文件夹（包括学号命名的文件夹）
    all_items = os.listdir(nb116_person_course_path)
    person_folders = []
    
    for item in all_items:
        item_path = os.path.join(nb116_person_course_path, item)
        # 只处理文件夹，排除JSON文件和其他文件
        if os.path.isdir(item_path) and not item.endswith('.json'):
            person_folders.append(item_path)
    
    if not person_folders:
        print("警告: 没有找到person文件夹")
        return {'processed': 0, 'failed': 0, 'total': 0}
    
    stats = {'processed': 0, 'failed': 0, 'total': 0}
    
    # 为每个person文件夹处理图片
    for person_folder in person_folders:
        person_id = os.path.basename(person_folder)
        print(f"处理{person_id}...")
        
        # 获取该person文件夹中的所有图片
        person_images_paths = glob.glob(os.path.join(person_folder, "*.jpg"))
        
        for img_path in tqdm(person_images_paths, desc=f"Student {person_id}"):
            stats['total'] += 1
            
            img_name = os.path.basename(img_path)
            
            # 检查是否已经包含位置信息
            if '_x1_' in img_name and '_y1_' in img_name and '_x2_' in img_name and '_y2_' in img_name:
                print(f"跳过已处理的图片: {img_name}")
                stats['processed'] += 1
                continue
            
            # 解析文件名获取frame信息
            frame_name, orig_person_id, img_index = parse_person_image_name(img_name)
            
            if not frame_name:
                print(f"无法解析文件名: {img_name}")
                stats['failed'] += 1
                continue
            
            # 查找对应的原始frame图片
            original_frame_path = os.path.join(nb116_course_path, frame_name)
            
            if not os.path.exists(original_frame_path):
                print(f"找不到原始frame图片: {original_frame_path}")
                stats['failed'] += 1
                continue
            
            # 读取原始图片和person图片
            original_image = cv2.imread(original_frame_path)
            person_image = cv2.imread(img_path)
            
            if original_image is None or person_image is None:
                print(f"无法读取图片: {original_frame_path} 或 {img_path}")
                stats['failed'] += 1
                continue
            
            # 使用YOLO检测原始图片中的人体
            detected_person_images, detected_person_boxes = detect_persons_yolo(
                original_image, yolo_model, 
                conf=args.conf, iou=args.iou, max_persons=args.max_persons,
                face_detection=args.face_detection
            )
            
            if not detected_person_boxes:
                print(f"在原始图片中未检测到人体: {frame_name}")
                stats['failed'] += 1
                continue
            
            # 找到匹配的边界框
            matching_box, similarity = find_matching_person_box(
                person_image, detected_person_boxes, detected_person_images,
                args.similarity_threshold
            )
            
            if matching_box is None:
                print(f"无法找到匹配的边界框: {img_name} (最高相似度: {similarity:.3f})")
                stats['failed'] += 1
                continue
            
            # 提取位置信息
            x1, y1, x2, y2, conf_score = matching_box
            
            # 生成新的文件名
            base_name = os.path.splitext(img_name)[0]
            ext = os.path.splitext(img_name)[1]
            new_name = f"{base_name}_x1_{x1}_y1_{y1}_x2_{x2}_y2_{y2}{ext}"
            new_path = os.path.join(person_folder, new_name)
            
            # 重命名文件
            try:
                if args.copy_mode:
                    shutil.copy2(img_path, new_path)
                    print(f"复制: {img_name} -> {new_name}")
                else:
                    os.rename(img_path, new_path)
                    print(f"重命名: {img_name} -> {new_name}")
                
                stats['processed'] += 1
                
            except Exception as e:
                print(f"重命名失败: {img_name} -> {new_name}, 错误: {e}")
                stats['failed'] += 1
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="为NB116_person中的图片添加位置信息")
    parser.add_argument(
        "--nb116-folder",
        default="dataset/NB116",
        help="原始NB116文件夹路径"
    )
    parser.add_argument(
        "--nb116-person-folder",
        default=r"C:\Project\Classroom-Reid\dataset\NB116_person_xh",
        help="NB116_person文件夹路径"
    )
    parser.add_argument(
        "--yolo-model",
        default="models/yolo12x.pt",
        help="YOLO模型路径"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.2,
        help="YOLO检测置信度阈值"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.2,
        help="YOLO NMS的IoU阈值"
    )
    parser.add_argument(
        "--max-persons",
        type=int,
        default=100,
        help="每张图片最多检测的人数"
    )
    parser.add_argument(
        "--face-detection",
        action='store_true',
        default=True,
        help="是否启用人脸检测过滤"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.8,
        help="图像相似度匹配阈值"
    )
    parser.add_argument(
        "--copy-mode",
        action='store_true',
        help="复制模式：保留原文件，创建带位置信息的新文件"
    )
    parser.add_argument(
        "--specific-course",
        help="只处理特定课程，格式如: 508NB116/Class_3"
    )
    
    args = parser.parse_args()
    
    nb116_path = args.nb116_folder
    nb116_person_path = args.nb116_person_folder
    
    if not os.path.exists(nb116_path):
        print(f"错误: NB116文件夹不存在: {nb116_path}")
        return
    
    if not os.path.exists(nb116_person_path):
        print(f"错误: NB116_person文件夹不存在: {nb116_person_path}")
        return
    
    # 加载YOLO模型
    print(f"加载YOLO模型: {args.yolo_model}")
    yolo_model = YOLO(args.yolo_model)
    
    total_stats = {'processed': 0, 'failed': 0, 'total': 0}
    
    # 如果指定了特定课程
    if args.specific_course:
        parts = args.specific_course.split('/')
        if len(parts) == 2:
            date_folder, class_folder = parts
            nb116_course_path = os.path.join(nb116_path, date_folder, class_folder)
            nb116_person_course_path = os.path.join(nb116_person_path, date_folder, class_folder)
            
            if os.path.exists(nb116_course_path) and os.path.exists(nb116_person_course_path):
                stats = process_single_course(nb116_course_path, nb116_person_course_path, yolo_model, args)
                for key in total_stats:
                    total_stats[key] += stats[key]
            else:
                print(f"课程路径不存在: {nb116_course_path} 或 {nb116_person_course_path}")
        else:
            print("特定课程格式错误，应为: 日期文件夹/班级文件夹")
        
    else:
        # 遍历所有日期文件夹
        date_folders = [f for f in os.listdir(nb116_person_path) 
                       if os.path.isdir(os.path.join(nb116_person_path, f)) and 'NB116' in f]
        
        for date_folder in date_folders:
            print(f"\n处理日期文件夹: {date_folder}")
            
            nb116_date_path = os.path.join(nb116_path, date_folder)
            nb116_person_date_path = os.path.join(nb116_person_path, date_folder)
            
            if not os.path.exists(nb116_date_path):
                print(f"跳过: 原始NB116中不存在对应的日期文件夹: {nb116_date_path}")
                continue
            
            # 遍历所有班级文件夹
            class_folders = [f for f in os.listdir(nb116_person_date_path) 
                            if os.path.isdir(os.path.join(nb116_person_date_path, f)) and f.startswith('Class_')]
            
            for class_folder in class_folders:
                nb116_class_path = os.path.join(nb116_date_path, class_folder)
                nb116_person_class_path = os.path.join(nb116_person_date_path, class_folder)
                
                if not os.path.exists(nb116_class_path):
                    print(f"跳过: 原始NB116中不存在对应的班级文件夹: {nb116_class_path}")
                    continue
                
                stats = process_single_course(nb116_class_path, nb116_person_class_path, yolo_model, args)
                
                for key in total_stats:
                    total_stats[key] += stats[key]
    
    # 显示最终统计信息
    print("\n=== 处理完成统计 ===")
    print(f"总图片数: {total_stats['total']}")
    print(f"成功处理: {total_stats['processed']}")
    print(f"处理失败: {total_stats['failed']}")
    if total_stats['total'] > 0:
        success_rate = total_stats['processed'] / total_stats['total'] * 100
        print(f"成功率: {success_rate:.1f}%")
    
    print("\n位置信息添加完成！")

if __name__ == "__main__":
    main()