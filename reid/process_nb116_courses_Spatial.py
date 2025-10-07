# encoding: utf-8
"""
@author:  AI Assistant
@contact: Based on process_stream_courses.py
"""

import argparse
import sys
import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.backends import cudnn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from ultralytics import YOLO
import shutil
from tqdm import tqdm
from collections import defaultdict
import glob
import json
from insightface.app import FaceAnalysis

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from predictor import FeatureExtractionDemo

cudnn.benchmark = True
setup_logger(name="fastreid")
face_app = FaceAnalysis(allowed_modules=['detection'], name='antelopev2', 
                     providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_thresh=0.6, det_size=(320, 320))


def setup_cfg(args):
    """加载配置文件和命令行参数"""
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="处理NB116目录中所有课程的人员ID识别与组织")
    parser.add_argument(
        "--config-file",
        default=r"C:\Project\Classroom-Reid\reid\config\bagtricks_R50-ibn.yml",
        metavar="FILE",
        help="配置文件路径",
    )
    parser.add_argument(
        "--parallel",
        action='store_true',
        help='是否使用多进程进行特征提取'
    )
    parser.add_argument(
        "--nb116-folder",
        default=r"C:\Project\Classroom-Reid\dataset\NB116",
        help="NB116文件夹路径",
    )
    parser.add_argument(
        "--output-folder",
        default=r"C:\Project\Classroom-Reid\dataset\NB116_person_spatial",
        help="保存结果的文件夹路径",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="相似度阈值，高于此值被认为是同一个人",
    )
    parser.add_argument(
        "--yolo-model",
        default="models/yolo12x.pt",
        help="YOLO模型路径",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.2,
        help="YOLO检测置信度阈值",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.2,
        help="YOLO NMS的IoU阈值",
    )
    parser.add_argument(
        "--max-persons",
        type=int,
        default=100,
        help="每张图片最多检测的人数",
    )
    parser.add_argument(
        "--position-weight",
        type=float,
        default=0.6,
        help="位置相似度的权重",
    )
    parser.add_argument(
        "--save-visualizations",
        action='store_true',
        help='是否保存匹配可视化结果'
    )
    parser.add_argument(
        "--face-detection",
        action='store_true',
        default=True,
        help='是否启用人脸检测，只保留检测到人脸的人体图像'
    )
    parser.add_argument(
        "--face-det-thresh",
        type=float,
        default=0.5,
        help="人脸检测置信度阈值",
    )
    parser.add_argument(
        "--face-det-size",
        type=int,
        default=320,
        help="人脸检测输入图像大小",
    )
    parser.add_argument(
        "--resume",
        action='store_true',
        default=True,
        help='是否启用断点续传，跳过已处理的课程（默认启用）'
    )
    parser.add_argument(
        "--force-reprocess",
        action='store_true',
        help='强制重新处理所有课程，忽略已有结果'
    )
    parser.add_argument(
        "--opts",
        help="使用命令行 'KEY VALUE' 对修改配置选项",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def detect_persons_yolo(image, yolo_model, conf=0.4, iou=0.7, max_persons=10, face_detection=True, face_det_thresh=0.6, face_det_size=320):
    """
    使用YOLOv8检测图像中的所有人体并返回裁剪后的人体图像和边界框
    如果启用人脸检测，则只保留同时检测到人脸的人体图像
    """
    # 创建可视化图像
    annotated_image = image.copy()
    
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
        return [], [], annotated_image
    
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
                
                # 如果检测到人脸，在可视化图像上标注
                if has_face:
                    for face in faces:
                        # 获取人脸边界框（相对于人体图像）
                        face_box = face.bbox.astype(int)
                        # 转换为相对于原始图像的坐标
                        fx1, fy1, fx2, fy2 = face_box[0] + x1, face_box[1] + y1, face_box[2] + x1, face_box[3] + y1
                        # 在可视化图像上绘制人脸边界框
                        cv2.rectangle(annotated_image, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
            
            # 只有在不启用人脸检测或检测到人脸时才添加人体
            if has_face:
                person_images.append(person_image)
                person_boxes.append([x1, y1, x2, y2, conf_score])
                
                # 在可视化图像上绘制边界框和ID
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_image, f"Person {i+1} ({conf_score:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return person_images, person_boxes, annotated_image


def extract_features_batch(person_images, demo):
    """批量提取人体特征"""
    features = []
    
    for img in person_images:
        # 提取特征
        feat = demo.run_on_image(img)
        # 归一化特征
        feat = F.normalize(feat)
        features.append(feat)
    
    # 如果没有检测到人体，返回空张量
    if not features:
        return torch.zeros((0, 0))
    
    # 将特征列表堆叠成矩阵
    return torch.cat(features, dim=0)


def compute_similarity_matrix(features1, features2):
    """计算两组特征之间的相似度矩阵"""
    # 如果任一组特征为空，返回空矩阵
    if features1.shape[0] == 0 or features2.shape[0] == 0:
        return torch.zeros((0, 0))
    
    # 计算余弦相似度矩阵
    similarity_matrix = torch.mm(features1, features2.t())
    return similarity_matrix


def compute_position_similarity(boxes1, boxes2, img_shape):
    """计算两组边界框之间的位置相似度"""
    if not boxes1 or not boxes2:
        return torch.zeros((0, 0))
    
    n1, n2 = len(boxes1), len(boxes2)
    position_similarity = torch.zeros((n1, n2))
    
    # 计算IoU和归一化的位置距离相似度
    for i in range(n1):
        for j in range(n2):
            box1 = boxes1[i][:4]  # [x1, y1, x2, y2]
            box2 = boxes2[j][:4]  # [x1, y1, x2, y2]
            
            # 计算IoU
            x_left = max(box1[0], box2[0])
            y_top = max(box1[1], box2[1])
            x_right = min(box1[2], box2[2])
            y_bottom = min(box1[3], box2[3])
            
            # 计算交集面积
            if x_right < x_left or y_bottom < y_top:
                intersection_area = 0
            else:
                intersection_area = (x_right - x_left) * (y_bottom - y_top)
            
            # 计算各自面积
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            
            # 计算并集面积
            union_area = box1_area + box2_area - intersection_area
            
            # 计算IoU
            iou = intersection_area / union_area if union_area > 0 else 0
            
            # 计算中心点距离的相似度
            center1_x = (box1[0] + box1[2]) / 2 / img_shape[1]  # 归一化到[0,1]
            center1_y = (box1[1] + box1[3]) / 2 / img_shape[0]
            center2_x = (box2[0] + box2[2]) / 2 / img_shape[1]
            center2_y = (box2[1] + box2[3]) / 2 / img_shape[0]
            
            # 计算归一化欧氏距离
            center_dist = torch.sqrt(torch.tensor(
                (center1_x - center2_x)**2 + (center1_y - center2_y)**2
            ))
            
            # 将距离转换为相似度（距离越小，相似度越高）
            center_sim = 1.0 - min(center_dist, 1.0)
            
            # 结合IoU和中心点距离的相似度
            position_similarity[i, j] = 0.5 * iou + 0.5 * center_sim
    
    return position_similarity


def combine_similarity_matrices(feature_sim, position_sim, position_weight=0.3):
    """结合外观特征相似度和位置相似度"""
    if feature_sim.shape[0] == 0 or feature_sim.shape[1] == 0:
        return feature_sim
    
    # 结合外观特征和位置信息
    combined_sim = (1 - position_weight) * feature_sim + position_weight * position_sim
    
    return combined_sim


def match_persons(similarity_matrix, threshold=0.5):
    """根据相似度矩阵匹配两张图片中的人"""
    matches = []
    
    # 如果相似度矩阵为空，返回空列表
    if similarity_matrix.shape[0] == 0 or similarity_matrix.shape[1] == 0:
        return matches
    
    # 将相似度矩阵转换为numpy数组
    sim_matrix = similarity_matrix.cpu().numpy()
    
    # 贪心匹配：每次找到最大相似度的那一对
    while sim_matrix.size > 0:
        # 找到最大相似度的位置
        idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
        i, j = idx
        similarity = sim_matrix[i, j]
        
        # 如果最大相似度低于阈值，结束匹配
        if similarity < threshold:
            break
        
        # 添加匹配结果
        matches.append((i, j, similarity))
        
        # 将已匹配的行和列设为-1，避免重复匹配
        sim_matrix[i, :] = -1
        sim_matrix[:, j] = -1
    
    return matches


def is_course_processed(output_path):
    """
    检查课程是否已经处理完成
    
    Args:
        output_path: 输出文件夹路径
    
    Returns:
        bool: 如果已处理完成返回True，否则返回False
    """
    # 检查输出目录是否存在
    if not os.path.exists(output_path):
        return False
    
    # 检查是否存在person_stats.json文件（处理完成的标志）
    stats_file = os.path.join(output_path, "person_stats.json")
    if not os.path.exists(stats_file):
        return False
    
    # 检查是否存在至少一个person文件夹
    person_folders = [f for f in os.listdir(output_path) if f.startswith("person_") and os.path.isdir(os.path.join(output_path, f))]
    
    return len(person_folders) > 0


def process_single_course(course_path, output_path, demo, yolo_model, args):
    """
    处理单个课程文件夹
    
    Args:
        course_path: 课程文件夹路径
        output_path: 输出文件夹路径
        demo: 特征提取器
        yolo_model: YOLO模型
        args: 命令行参数
    
    Returns:
        str: 'processed', 'skipped', 或 'error'
    """
    # 检查是否已经处理过（除非强制重新处理）
    if args.resume and not args.force_reprocess and is_course_processed(output_path):
        print(f"\n跳过已处理的课程: {course_path}")
        return 'skipped'
    
    print(f"\n处理课程: {course_path}")
    
    # 如果强制重新处理，先清理输出目录
    if args.force_reprocess and os.path.exists(output_path):
        print(f"清理已有输出目录: {output_path}")
        shutil.rmtree(output_path)
    
    # 获取文件夹中所有图片（只处理frame开头的图片）
    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in img_extensions:
        pattern = os.path.join(course_path, f"frame*{ext}")
        image_files.extend(glob.glob(pattern))
    
    # 确保找到图片
    if not image_files:
        print(f"警告: 在文件夹 {course_path} 中没有找到frame图片")
        return 'error'
    
    print(f"找到 {len(image_files)} 张frame图片")
    
    # 存储每张图片中检测到的人体信息
    all_person_images = []  # 存储所有图片中的人体裁剪图像
    all_person_boxes = []   # 存储所有图片中的人体边界框
    all_source_images = []  # 存储原始图像
    all_image_indices = []  # 存储每个人体所属的图片索引
    all_image_names = []    # 存储图片名称
    
    # 处理每张图片
    print("正在检测所有图片中的人体...")
    for img_idx, img_path in enumerate(tqdm(image_files)):
        # 读取图片
        image = cv2.imread(img_path)
        if image is None:
            print(f"无法读取图片: {img_path}")
            continue
        
        # 检测人体
        person_images, person_boxes, _ = detect_persons_yolo(
            image, yolo_model, conf=args.conf, iou=args.iou, max_persons=args.max_persons,
            face_detection=args.face_detection, face_det_thresh=args.face_det_thresh, 
            face_det_size=args.face_det_size)
        
        if not person_images:
            continue
        
        # 存储检测结果
        all_person_images.extend(person_images)
        all_person_boxes.extend(person_boxes)
        all_source_images.extend([image] * len(person_images))
        all_image_indices.extend([img_idx] * len(person_images))
        all_image_names.extend([os.path.basename(img_path)] * len(person_images))
    
    # 检查是否检测到人体
    if not all_person_images:
        print("警告: 在所有图片中均未检测到人体")
        return 'error'
    
    print(f"在所有图片中共检测到 {len(all_person_images)} 个人体")
    
    # 提取所有人体特征
    print("提取所有人体特征...")
    all_features = extract_features_batch(all_person_images, demo)
    
    # 创建人体ID组（初始时每个人是单独的一组）
    person_ids = list(range(len(all_person_images)))
    
    # 已处理的图片对
    processed_pairs = set()
    
    # 迭代匹配不同图片中的相同人物
    print("匹配不同图片中的相同人物...")
    for i in range(len(image_files)):
        for j in range(i+1, len(image_files)):
            # 如果已经处理过这对图片，则跳过
            if (i, j) in processed_pairs:
                continue
            
            # 标记为已处理
            processed_pairs.add((i, j))
            
            # 获取两张图片中的人体索引
            img1_person_indices = [idx for idx, img_idx in enumerate(all_image_indices) if img_idx == i]
            img2_person_indices = [idx for idx, img_idx in enumerate(all_image_indices) if img_idx == j]
            
            if not img1_person_indices or not img2_person_indices:
                continue
            
            # 获取两张图片中的人体特征
            features1 = all_features[img1_person_indices]
            features2 = all_features[img2_person_indices]
            
            # 获取两张图片中的人体边界框
            boxes1 = [all_person_boxes[idx] for idx in img1_person_indices]
            boxes2 = [all_person_boxes[idx] for idx in img2_person_indices]
            
            # 计算特征相似度矩阵
            feature_similarity = compute_similarity_matrix(features1, features2)
            
            # 计算位置相似度矩阵
            img_shape = all_source_images[img1_person_indices[0]].shape[:2]  # (height, width)
            position_similarity = compute_position_similarity(boxes1, boxes2, img_shape)
            
            # 结合特征和位置相似度
            combined_similarity = combine_similarity_matrices(
                feature_similarity, position_similarity, args.position_weight)
            
            # 匹配人物
            matches = match_persons(combined_similarity, args.threshold)
            
            # 更新人物ID
            for idx1_local, idx2_local, similarity in matches:
                idx1_global = img1_person_indices[idx1_local]
                idx2_global = img2_person_indices[idx2_local]
                
                # 将两个人体分配到同一个ID组
                id1 = person_ids[idx1_global]
                id2 = person_ids[idx2_global]
                
                # 合并ID组：将所有id2的人体改为id1
                for k in range(len(person_ids)):
                    if person_ids[k] == id2:
                        person_ids[k] = id1
    
    # 重新分配连续的ID
    unique_ids = list(set(person_ids))
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    final_person_ids = [id_mapping[pid] for pid in person_ids]
    
    print(f"识别出 {len(unique_ids)} 个不同的人")
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 按人物ID组织图片
    person_groups = defaultdict(list)
    for idx, person_id in enumerate(final_person_ids):
        person_groups[person_id].append({
            'image': all_person_images[idx],
            'box': all_person_boxes[idx],
            'source_image': all_source_images[idx],
            'image_name': all_image_names[idx],
            'global_idx': idx
        })
    
    # 过滤掉少于5张图片的person
    filtered_person_groups = {pid: data for pid, data in person_groups.items() if len(data) >= 10}
    
    print(f"过滤前有 {len(person_groups)} 个人，过滤后有 {len(filtered_person_groups)} 个人")
    if len(person_groups) > len(filtered_person_groups):
        removed_persons = len(person_groups) - len(filtered_person_groups)
        print(f"移除了 {removed_persons} 个图片数量少于5张的人")
    
    # 保存每个人的图片到对应文件夹
    for person_id, person_data in filtered_person_groups.items():
        person_folder = os.path.join(output_path, f"person_{person_id}")
        os.makedirs(person_folder, exist_ok=True)
        
        for idx, data in enumerate(person_data):
            # 保存人体裁剪图像，在文件名中包含位置信息
            image_name = data['image_name']
            base_name = os.path.splitext(image_name)[0]
            
            # 获取边界框位置信息
            box = data['box']
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            
            # 在文件名中添加位置信息
            person_image_path = os.path.join(person_folder, f"{base_name}_person_{person_id}_{idx}_pos_{x1}_{y1}_{x2}_{y2}.jpg")
            cv2.imwrite(person_image_path, data['image'])
    
    # 保存统计信息
    stats = {
        'total_images': len(image_files),
        'total_persons_detected': len(all_person_images),
        'unique_persons_before_filter': len(unique_ids),
        'unique_persons_after_filter': len(filtered_person_groups),
        'filtered_out_persons': len(person_groups) - len(filtered_person_groups),
        'person_distribution': {f"person_{pid}": len(filtered_person_groups[pid]) for pid in filtered_person_groups}
    }
    
    stats_path = os.path.join(output_path, "person_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"处理完成，结果保存到: {output_path}")
    print(f"统计信息: {stats}")
    
    return 'processed'


def main():
    """主函数"""
    args = get_parser().parse_args()
    
    # 设置配置
    cfg = setup_cfg(args)
    
    # 设置预训练模型路径
    cfg.defrost()
    cfg.MODEL.WEIGHTS = "models/veriwild_bot_R50-ibn.pth"
    cfg.MODEL.DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    
    # 创建特征提取器
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)
    
    # 加载YOLO模型
    print(f"加载YOLO模型: {args.yolo_model}")
    yolo_model = YOLO(args.yolo_model)
    
    # 获取NB116目录下所有的日期文件夹
    nb116_path = args.nb116_folder
    if not os.path.exists(nb116_path):
        print(f"错误: NB116文件夹不存在: {nb116_path}")
        return
    
    # 遍历所有日期文件夹（如508NB116、515NB116等）
    date_folders = [f for f in os.listdir(nb116_path) if os.path.isdir(os.path.join(nb116_path, f)) and 'NB116' in f]
    
    if not date_folders:
        print(f"错误: 在 {nb116_path} 中没有找到NB116相关的日期文件夹")
        return
    
    print(f"找到 {len(date_folders)} 个日期文件夹: {date_folders}")
    
    # 统计信息
    total_courses = 0
    processed_courses = 0
    skipped_courses = 0
    error_courses = 0
    
    # 处理每个日期文件夹
    for date_folder in tqdm(date_folders, desc="处理日期文件夹"):
        date_path = os.path.join(nb116_path, date_folder)
        
        # 获取该日期下的所有班级文件夹（Class_3、Class_4等）
        class_folders = [f for f in os.listdir(date_path) if os.path.isdir(os.path.join(date_path, f)) and f.startswith('Class_')]
        
        for class_folder in tqdm(class_folders, desc=f"处理{date_folder}中的班级", leave=False):
            class_path = os.path.join(date_path, class_folder)
            total_courses += 1
            
            # 创建对应的输出路径
            output_class_path = os.path.join(args.output_folder, date_folder, class_folder)
            
            # 处理单个课程
            try:
                result = process_single_course(class_path, output_class_path, demo, yolo_model, args)
                if result == 'processed':
                    processed_courses += 1
                elif result == 'skipped':
                    skipped_courses += 1
                elif result == 'error':
                    error_courses += 1
            except Exception as e:
                print(f"处理课程 {class_path} 时出错: {e}")
                error_courses += 1
                continue
    
    # 显示最终统计信息
    print("\n=== 处理完成统计 ===")
    print(f"总课程数: {total_courses}")
    print(f"已处理: {processed_courses}")
    print(f"已跳过: {skipped_courses}")
    print(f"处理错误: {error_courses}")
    print(f"处理成功率: {processed_courses/(total_courses-skipped_courses)*100:.1f}%" if (total_courses-skipped_courses) > 0 else "处理成功率: 0%")
    print("\n所有NB116课程处理完成！")


if __name__ == "__main__":
    main()