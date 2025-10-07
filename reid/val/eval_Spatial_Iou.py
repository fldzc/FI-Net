# encoding: utf-8
"""
@author:  AI Assistant
@contact: Person Re-identification Evaluation with Spatial Information for NB116_person_spatial Dataset
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
from tqdm import tqdm
from collections import defaultdict
import glob
import json
import re
from sklearn.metrics import roc_curve, auc, normalized_mutual_info_score, confusion_matrix
import seaborn as sns
from scipy.optimize import linear_sum_assignment

sys.path.append('.')
sys.path.append('../..')

from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from reid.predictor import FeatureExtractionDemo

cudnn.benchmark = True
setup_logger(name="fastreid")


def setup_cfg(args):
    """Load config file and CLI options."""
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="评估NB116_person_spatial数据集的空间信息人员重识别效果")
    parser.add_argument(
        "--config-file",
        default="reid/config/bagtricks_R50-ibn.yml",
        metavar="FILE",
        help="配置文件路径",
    )
    parser.add_argument(
        "--dataset-path",
        default="dataset/NB116_person_spatial",
        help="数据集路径",
    )
    parser.add_argument(
        "--output-path",
        default="reid/val/results_spatial",
        help="评估结果保存路径",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="相似度阈值，用于判断是否为同一人",
    )
    parser.add_argument(
        "--position-weight",
        type=float,
        default=0.1,
        help="位置相似度的权重（0-1之间，0表示仅使用外观特征）",
    )
    parser.add_argument(
        "--max-rank",
        type=int,
        default=10,
        help="CMC曲线的最大rank值",
    )
    parser.add_argument(
        "--save-visualizations",
        action='store_true',
        help='是否保存可视化结果'
    )
    parser.add_argument(
        "--parallel",
        action='store_true',
        help='是否使用多进程进行特征提取'
    )
    parser.add_argument(
        "--opts",
        help="使用命令行 'KEY VALUE' 对修改配置选项",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def compute_accuracy(y_true, y_pred):
    """
    Compute clustering accuracy (ACC) using Hungarian algorithm for label matching.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # 获取唯一的真实标签和预测标签
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    
    # 创建一个从标签到索引的映射
    true_map = {label: i for i, label in enumerate(true_labels)}
    pred_map = {label: i for i, label in enumerate(pred_labels)}
    
    # 将标签转换为整数索引
    y_true_int = np.array([true_map[label] for label in y_true])
    y_pred_int = np.array([pred_map[label] for label in y_pred])
    
    # 构建混淆矩阵作为代价矩阵
    contingency = confusion_matrix(y_true_int, y_pred_int)
    
    # 使用匈牙利算法找到最优匹配（最大化对角线元素之和）
    row_ind, col_ind = linear_sum_assignment(-contingency)
    
    # 计算匹配上的样本总数
    correctly_assigned = contingency[row_ind, col_ind].sum()
    
    # 总样本数
    total_samples = len(y_true)
    
    return correctly_assigned / total_samples if total_samples > 0 else 0.0


def parse_position_from_filename(filename):
    """
    Parse position information from filename.
    
    Args:
        filename: e.g., "frame_xxx_person_0_0_pos_663_411_818_539.jpg"
    
    Returns:
        tuple: (x1, y1, x2, y2) or None
    """
    # Match pattern: pos_x1_y1_x2_y2
    pattern = r'pos_(\d+)_(\d+)_(\d+)_(\d+)'
    match = re.search(pattern, filename)
    
    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        return (x1, y1, x2, y2)
    
    return None


def load_dataset_with_positions(dataset_path):
    """
    Load dataset with position information.
    
    Returns:
        dict: A dictionary of all courses including positions.
    """
    print("正在加载数据集（包含位置信息）...")
    dataset = {}
    
    # 遍历所有日期文件夹
    date_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f)) and 'NB116' in f]
    
    for date_folder in sorted(date_folders):
        date_path = os.path.join(dataset_path, date_folder)
        
        # 获取该日期下的所有课程文件夹
        class_folders = [f for f in os.listdir(date_path) if os.path.isdir(os.path.join(date_path, f)) and f.startswith('Class_')]
        
        for class_folder in sorted(class_folders):
            class_path = os.path.join(date_path, class_folder)
            course_key = f"{date_folder}_{class_folder}"
            
            # 读取统计信息
            stats_file = os.path.join(class_path, "person_stats.json")
            if not os.path.exists(stats_file):
                print(f"警告: 未找到统计文件 {stats_file}")
                continue
            
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            
            # 获取所有person文件夹 (兼容 "person_id" 和纯数字 "id" 格式)
            person_folders = []
            for f in os.listdir(class_path):
                if os.path.isdir(os.path.join(class_path, f)):
                    if f.startswith("person_") or f.isdigit():
                        person_folders.append(f)
            
            course_data = {
                'path': class_path,
                'stats': stats,
                'persons': {},
                'all_images': [],
                'all_positions': [],
                'ground_truth_labels': [],
                'image_paths': []
            }
            
            # 加载每个person的图片和位置信息
            for person_folder in person_folders:
                try:
                    # 兼容 "person_1" 和 "1" 两种格式
                    if person_folder.startswith("person_"):
                        person_id = int(person_folder.split('_')[1])
                    else:
                        person_id = int(person_folder)
                except (ValueError, IndexError):
                    continue # 如果文件夹名不是预期的格式，就跳过

                person_path = os.path.join(class_path, person_folder)
                
                # 获取该person的所有图片
                img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
                image_files = []
                for ext in img_extensions:
                    image_files.extend(glob.glob(os.path.join(person_path, ext)))
                
                images = []
                positions = []
                for img_path in sorted(image_files):
                    img = cv2.imread(img_path)
                    if img is not None:
                        # 解析位置信息
                        filename = os.path.basename(img_path)
                        position = parse_position_from_filename(filename)
                        
                        if position is not None:
                            images.append(img)
                            positions.append(position)
                            course_data['all_images'].append(img)
                            course_data['all_positions'].append(position)
                            course_data['ground_truth_labels'].append(person_id)
                            course_data['image_paths'].append(img_path)
                
                if images: # 只有当该文件夹下有有效图片时才添加
                    course_data['persons'][person_id] = {
                        'images': images,
                        'positions': positions,
                        'image_paths': sorted(image_files)
                    }
            
            dataset[course_key] = course_data
            print(f"加载课程 {course_key}: {len(course_data['persons'])} 个人, {len(course_data['all_images'])} 张图片")
    
    return dataset


def extract_features_batch(images, demo):
    """Extract image features in batch."""
    features = []
    
    for img in images:
        # 提取特征
        feat = demo.run_on_image(img)
        # 归一化特征
        feat = F.normalize(feat)
        features.append(feat)
    
    if not features:
        return torch.zeros((0, 0))
    
    return torch.cat(features, dim=0)


def compute_position_similarity(positions1, positions2, img_shape=(1080, 1920)):
    """
    Compute similarity between two sets of positions.
    
    Args:
        positions1: list of (x1, y1, x2, y2)
        positions2: list of (x1, y1, x2, y2)
        img_shape: image shape (height, width)
    
    Returns:
        torch.Tensor: position similarity matrix
    """
    if not positions1 or not positions2:
        return torch.zeros((0, 0))
    
    n1, n2 = len(positions1), len(positions2)
    position_similarity = torch.zeros((n1, n2))
    
    for i in range(n1):
        for j in range(n2):
            box1 = positions1[i]  # (x1, y1, x2, y2)
            box2 = positions2[j]  # (x1, y1, x2, y2)
            
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
            position_similarity[i, j] = 1 * iou + 0 * center_sim
    
    return position_similarity


def compute_combined_similarity_matrix(features1, features2, positions1, positions2, position_weight=0.3):
    """
    Compute similarity matrix combining appearance features and positions.
    
    Args:
        features1: features of set 1
        features2: features of set 2
        positions1: positions of set 1
        positions2: positions of set 2
        position_weight: weight for position similarity
    
    Returns:
        torch.Tensor: combined similarity matrix
    """
    if features1.shape[0] == 0 or features2.shape[0] == 0:
        return torch.zeros((0, 0))
    
    # 计算外观特征相似度
    feature_similarity = torch.mm(features1, features2.t())
    
    # 计算位置相似度
    position_similarity = compute_position_similarity(positions1, positions2)
    
    # 结合外观特征和位置信息
    combined_similarity = (1 - position_weight) * feature_similarity + position_weight * position_similarity
    
    return combined_similarity


def perform_spatial_clustering(features, positions, threshold=0.8, position_weight=0.3):
    """
    Cluster based on appearance features and position information.
    
    Args:
        features: feature matrix
        positions: list of positions
        threshold: similarity threshold
        position_weight: weight for position similarity
    
    Returns:
        list: predicted label per image
    """
    n_images = features.shape[0]
    if n_images == 0:
        return []
    
    # 计算组合相似度矩阵
    similarity_matrix = compute_combined_similarity_matrix(
        features, features, positions, positions, position_weight)
    
    # 初始化聚类标签
    predicted_labels = list(range(n_images))
    
    # 基于相似度进行聚类
    for i in range(n_images):
        for j in range(i + 1, n_images):
            if similarity_matrix[i, j] > threshold:
                # 合并聚类：将j的标签改为i的标签
                old_label = predicted_labels[j]
                new_label = predicted_labels[i]
                for k in range(n_images):
                    if predicted_labels[k] == old_label:
                        predicted_labels[k] = new_label
    
    # 重新分配连续的标签
    unique_labels = list(set(predicted_labels))
    label_mapping = {old_label: new_id for new_id, old_label in enumerate(unique_labels)}
    final_labels = [label_mapping[label] for label in predicted_labels]
    
    return final_labels


def compute_cmc_map_spatial(features, positions, labels, position_weight=0.3, max_rank=10):
    """
    Compute CMC and mAP metrics with spatial information.
    
    Args:
        features: all features
        positions: all positions
        labels: all labels (person IDs)
        position_weight: weight for position similarity
        max_rank: maximum rank for CMC curve
    
    Returns:
        tuple: (cmc, mAP)
    """
    num_samples = features.shape[0]
    
    if num_samples == 0:
        return np.zeros(max_rank), 0.0
    
    all_cmc = []
    all_AP = []
    
    # 对每个样本作为查询进行评估
    for q_idx in range(num_samples):
        q_label = labels[q_idx]
        q_feature = features[q_idx:q_idx+1]
        q_position = [positions[q_idx]]
        
        # 获取除查询样本外的所有样本作为画廊
        gallery_indices = [i for i in range(num_samples) if i != q_idx]
        if not gallery_indices:
            continue
        
        # 获取画廊的特征、位置和标签
        gallery_features = features[gallery_indices]
        gallery_positions = [positions[i] for i in gallery_indices]
        gallery_labels = [labels[i] for i in gallery_indices]
        
        # 计算组合相似度
        combined_similarities = compute_combined_similarity_matrix(
            q_feature, gallery_features, q_position, gallery_positions, position_weight)
        similarities = combined_similarities[0].cpu().numpy()  # 取第一行
        
        # 按相似度排序
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_gallery_labels = [gallery_labels[i] for i in sorted_indices]
        
        # 找到正确匹配的位置
        matches = np.array([label == q_label for label in sorted_gallery_labels])
        
        if not matches.any():
            continue
        
        # 计算CMC - 修正版本
        cmc_tmp = np.zeros(max_rank)
        if matches.any():
            # 找到第一个正确匹配的位置（0-based index）
            first_correct = np.where(matches)[0][0]
            # 对于所有rank >= first_correct的位置，CMC = 1
            for rank in range(max_rank):
                if first_correct <= rank:
                    cmc_tmp[rank] = 1
        all_cmc.append(cmc_tmp)
        
        # 计算AP - 修正版本（标准的Average Precision计算）
        if matches.any():
            num_rel = matches.sum()
            if num_rel > 0:
                ap = 0.0
                num_correct = 0
                for k in range(len(matches)):
                    if matches[k]:  # 如果第k个结果是正确的
                        num_correct += 1
                        precision_at_k = num_correct / (k + 1)
                        ap += precision_at_k
                
                # 平均精度 = 所有正确结果处精度的平均
                ap = ap / num_rel
                all_AP.append(ap)
    
    # 计算平均CMC和mAP
    if all_cmc:
        cmc = np.array(all_cmc).mean(axis=0)
    else:
        cmc = np.zeros(max_rank)
    
    if all_AP:
        mAP = np.mean(all_AP)
    else:
        mAP = 0.0
    
    return cmc, mAP


def compute_roc_curve_spatial(features, positions, labels, position_weight=0.3):
    """
    Compute ROC curve with spatial information.
    
    Args:
        features: image features
        positions: positions
        labels: image labels
        position_weight: weight for position similarity
    
    Returns:
        tuple: (fpr, tpr, thresholds, auc_score)
    """
    n_samples = features.shape[0]
    
    # 计算所有样本对之间的组合相似度
    combined_similarity_matrix = compute_combined_similarity_matrix(
        features, features, positions, positions, position_weight).cpu().numpy()
    
    # 生成正负样本对
    y_true = []
    y_scores = []
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            # 相同标签为正样本，不同标签为负样本
            y_true.append(1 if labels[i] == labels[j] else 0)
            y_scores.append(combined_similarity_matrix[i, j])
    
    if not y_true:
        return [0], [0], [0], 0
    
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)
    
    return fpr, tpr, thresholds, auc_score


def compute_relative_error(predicted_counts, ground_truth_counts):
    """
    Compute Relative Error (RE).
    
    Args:
        predicted_counts: predicted number of persons
        ground_truth_counts: ground truth number of persons
    
    Returns:
        float: relative error
    """
    if ground_truth_counts == 0:
        return float('inf') if predicted_counts > 0 else 0
    
    return abs(predicted_counts - ground_truth_counts) / ground_truth_counts


def evaluate_course_spatial(course_data, demo, args):
    """
    Evaluate spatial re-identification performance for a single course.
    
    Args:
        course_data: course data
        demo: feature extractor
        args: arguments
    
    Returns:
        dict: evaluation results
    """
    all_images = course_data['all_images']
    all_positions = course_data['all_positions']
    ground_truth_labels = course_data['ground_truth_labels']
    
    if not all_images:
        return None
    
    print(f"  提取 {len(all_images)} 张图片的特征...")
    
    # 提取所有图片的特征
    features = extract_features_batch(all_images, demo)
    
    # 打印数据集统计信息
    unique_persons = len(set(ground_truth_labels))
    person_counts = {}
    for label in ground_truth_labels:
        person_counts[label] = person_counts.get(label, 0) + 1
    avg_images_per_person = np.mean(list(person_counts.values()))
    print(f"  数据集统计: {unique_persons}个人, 平均每人{avg_images_per_person:.1f}张图片")
    print(f"  空间权重: {args.position_weight:.2f}")
    
    # 执行空间聚类获得预测标签
    predicted_labels = perform_spatial_clustering(
        features, all_positions, args.threshold, args.position_weight)
    
    # 计算CMC和mAP（包含空间信息）
    cmc, mAP = compute_cmc_map_spatial(
        features, all_positions, ground_truth_labels, args.position_weight, args.max_rank)
    
    # 计算ROC曲线（包含空间信息）
    fpr, tpr, thresholds, auc_score = compute_roc_curve_spatial(
        features, all_positions, ground_truth_labels, args.position_weight)
    
    # 计算相对误差
    ground_truth_count = len(set(ground_truth_labels))
    predicted_count = len(set(predicted_labels))
    relative_error = compute_relative_error(predicted_count, ground_truth_count)
    
    # 计算NMI
    nmi_score = normalized_mutual_info_score(ground_truth_labels, predicted_labels)

    # 计算ACC
    acc_score = compute_accuracy(ground_truth_labels, predicted_labels)
    
    results = {
        'ground_truth_count': ground_truth_count,
        'predicted_count': predicted_count,
        'relative_error': relative_error,
        'position_weight': args.position_weight,
        'nmi': nmi_score,
        'acc': acc_score,
        'cmc': cmc.tolist(),
        'mAP': mAP,
        'rank1': cmc[0] if len(cmc) > 0 else 0,
        'rank5': cmc[4] if len(cmc) > 4 else 0,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'auc': auc_score,
        'predicted_labels': predicted_labels,
        'ground_truth_labels': ground_truth_labels
    }
    
    return results


def save_visualizations_spatial(all_results, output_path):
    """Save visualizations with spatial information."""
    os.makedirs(output_path, exist_ok=True)
    
    # Set font (supports Chinese characters if present)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. CMC曲线
    plt.figure(figsize=(12, 8))
    for course_name, results in all_results.items():
        if results and 'cmc' in results:
            ranks = range(1, len(results['cmc']) + 1)
            pos_weight = results.get('position_weight', 0.0)
            plt.plot(ranks, results['cmc'], marker='o', 
                    label=f"{course_name} (Rank-1: {results['rank1']:.3f}, Pos-W: {pos_weight:.2f})")
    
    plt.xlabel('Rank')
    plt.ylabel('Recognition Rate')
    plt.title('CMC Curve (with Spatial Information)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'cmc_curve_spatial.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC曲线
    plt.figure(figsize=(12, 8))
    for course_name, results in all_results.items():
        if results and 'fpr' in results and 'tpr' in results:
            pos_weight = results.get('position_weight', 0.0)
            plt.plot(results['fpr'], results['tpr'], 
                    label=f"{course_name} (AUC: {results['auc']:.3f}, Pos-W: {pos_weight:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (TAR@FAR with Spatial Information)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'roc_curve_spatial.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 位置权重影响分析
    if len(all_results) > 0:
        course_names = []
        relative_errors = []
        rank1_scores = []
        position_weights = []
        
        for course_name, results in all_results.items():
            if results:
                course_names.append(course_name)
                relative_errors.append(results['relative_error'])
                rank1_scores.append(results['rank1'])
                position_weights.append(results.get('position_weight', 0.0))
        
        if course_names:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # Rank-1与位置权重的关系
            x_pos = np.arange(len(course_names))
            bars1 = ax1.bar(x_pos, rank1_scores, alpha=0.7, color='blue')
            ax1.set_xlabel('Course')
            ax1.set_ylabel('Rank-1 Score')
            ax1.set_title(f'Rank-1 Performance with Position Weight = {position_weights[0]:.2f}')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(course_names, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # 相对误差
            bars2 = ax2.bar(x_pos, relative_errors, alpha=0.7, color='red')
            ax2.set_xlabel('Course')
            ax2.set_ylabel('Relative Error')
            ax2.set_title('Person Count Relative Error')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(course_names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'spatial_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()


def main():
    """Main entry point."""
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
    
    # 加载数据集（包含位置信息）
    dataset = load_dataset_with_positions(args.dataset_path)
    
    if not dataset:
        print("错误: 未找到有效的数据集")
        return
    
    print(f"\n找到 {len(dataset)} 个课程，开始空间信息评估...")
    print(f"位置权重设置: {args.position_weight}")
    
    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    
    # 评估所有课程
    all_results = {}
    overall_stats = {
        'rank1_scores': [],
        'rank5_scores': [],
        'mAP_scores': [],
        'auc_scores': [],
        'relative_errors': [],
        'nmi_scores': [],
        'acc_scores': []
    }
    
    for course_name, course_data in tqdm(dataset.items(), desc="评估课程"):
        print(f"\n评估课程: {course_name}")
        
        try:
            results = evaluate_course_spatial(course_data, demo, args)
            if results:
                all_results[course_name] = results
                
                # 收集统计信息
                overall_stats['rank1_scores'].append(results['rank1'])
                overall_stats['rank5_scores'].append(results['rank5'])
                overall_stats['mAP_scores'].append(results['mAP'])
                overall_stats['auc_scores'].append(results['auc'])
                overall_stats['relative_errors'].append(results['relative_error'])
                overall_stats['nmi_scores'].append(results['nmi'])
                overall_stats['acc_scores'].append(results['acc'])
                
                print(f"  结果: Rank-1: {results['rank1']:.3f}, Rank-5: {results['rank5']:.3f}, "
                      f"mAP: {results['mAP']:.3f}, AUC: {results['auc']:.3f}, "
                      f"ACC: {results['acc']:.3f}, NMI: {results['nmi']:.3f}, RE: {results['relative_error']:.3f}")
            else:
                print(f"  跳过课程 {course_name} (无有效数据)")
                
        except Exception as e:
            print(f"  评估课程 {course_name} 时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 计算总体统计
    if overall_stats['rank1_scores']:
        summary = {
            'mean_rank1': np.mean(overall_stats['rank1_scores']),
            'mean_rank5': np.mean(overall_stats['rank5_scores']),
            'mean_mAP': np.mean(overall_stats['mAP_scores']),
            'mean_AUC': np.mean(overall_stats['auc_scores']),
            'mean_RE': np.mean(overall_stats['relative_errors']),
            'mean_NMI': np.mean(overall_stats['nmi_scores']),
            'mean_ACC': np.mean(overall_stats['acc_scores']),
            'std_rank1': np.std(overall_stats['rank1_scores']),
            'std_rank5': np.std(overall_stats['rank5_scores']),
            'std_mAP': np.std(overall_stats['mAP_scores']),
            'std_AUC': np.std(overall_stats['auc_scores']),
            'std_RE': np.std(overall_stats['relative_errors']),
            'std_NMI': np.std(overall_stats['nmi_scores']),
            'std_ACC': np.std(overall_stats['acc_scores']),
            'position_weight': args.position_weight,
            'total_courses': len(all_results)
        }
        
        # 保存详细结果
        detailed_results = {
            'summary': summary,
            'course_results': all_results,
            'parameters': {
                'threshold': args.threshold,
                'position_weight': args.position_weight,
                'max_rank': args.max_rank,
                'dataset_path': args.dataset_path
            }
        }
        
        results_file = os.path.join(args.output_path, 'evaluation_results_spatial.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        # 保存可视化结果
        if args.save_visualizations:
            save_visualizations_spatial(all_results, args.output_path)
        
        # 打印总体结果
        print("\n" + "="*70)
        print("空间信息评估完成！总体结果:")
        print("="*70)
        print(f"评估课程数: {summary['total_courses']}")
        print(f"位置权重: {summary['position_weight']}")
        print(f"平均 Rank-1: {summary['mean_rank1']:.4f} ± {summary['std_rank1']:.4f}")
        print(f"平均 Rank-5: {summary['mean_rank5']:.4f} ± {summary['std_rank5']:.4f}")
        print(f"平均 mAP: {summary['mean_mAP']:.4f} ± {summary['std_mAP']:.4f}")
        print(f"平均 AUC: {summary['mean_AUC']:.4f} ± {summary['std_AUC']:.4f}")
        print(f"平均 ACC: {summary['mean_ACC']:.4f} ± {summary['std_ACC']:.4f}")
        print(f"平均 NMI: {summary['mean_NMI']:.4f} ± {summary['std_NMI']:.4f}")
        print(f"平均相对误差 (RE): {summary['mean_RE']:.4f} ± {summary['std_RE']:.4f}")
        print(f"\n结果已保存到: {args.output_path}")
        print("="*70)
    else:
        print("错误: 没有成功评估任何课程")


if __name__ == '__main__':
    main()
