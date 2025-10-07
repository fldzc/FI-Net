# encoding: utf-8
"""
@author:  AI Assistant
@contact: Person Re-identification Evaluation for NB116_person_spatial Dataset
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
    parser = argparse.ArgumentParser(description="评估NB116_person_spatial数据集的人员重识别效果")
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
        default="reid/val/results",
        help="评估结果保存路径",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="相似度阈值，用于判断是否为同一人",
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


def load_dataset(dataset_path):
    """
    Load dataset and return ground truth information.
    
    Returns:
        dict: dictionary containing all course data
    """
    print("正在加载数据集...")
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
                'ground_truth_labels': [],
                'image_paths': []
            }
            
            # 加载每个person的图片
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
                for img_path in sorted(image_files):
                    img = cv2.imread(img_path)
                    if img is not None:
                        images.append(img)
                        course_data['all_images'].append(img)
                        course_data['ground_truth_labels'].append(person_id)
                        course_data['image_paths'].append(img_path)
                
                if images: # 只有当该文件夹下有有效图片时才添加
                    course_data['persons'][person_id] = {
                        'images': images,
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


def compute_similarity_matrix(features1, features2):
    """Compute similarity matrix between two feature sets."""
    if features1.shape[0] == 0 or features2.shape[0] == 0:
        return torch.zeros((0, 0))
    
    # 计算余弦相似度矩阵
    similarity_matrix = torch.mm(features1, features2.t())
    return similarity_matrix


def perform_clustering(features, threshold=0.5):
    """
    Cluster based on feature similarity to simulate re-identification.
    
    Args:
        features: feature matrix
        threshold: similarity threshold
    
    Returns:
        list: predicted label per image
    """
    n_images = features.shape[0]
    if n_images == 0:
        return []
    
    # 计算相似度矩阵
    similarity_matrix = torch.mm(features, features.t())
    
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


def compute_cmc_map_single_course(features, labels, max_rank=10):
    """
    Compute CMC and mAP within a single course.
    
    Args:
        features: all features
        labels: all labels (person IDs)
        max_rank: maximum rank for CMC curve
    
    Returns:
        tuple: (cmc, mAP)
    """
    num_samples = features.shape[0]
    
    if num_samples == 0:
        return np.zeros(max_rank), 0.0
    
    # 计算相似度矩阵
    similarity_matrix = torch.mm(features, features.t()).cpu().numpy()
    
    all_cmc = []
    all_AP = []
    
    # 对每个样本作为查询进行评估
    for q_idx in range(num_samples):
        q_label = labels[q_idx]
        
        # 获取除查询样本外的所有样本作为画廊
        gallery_indices = [i for i in range(num_samples) if i != q_idx]
        if not gallery_indices:
            continue
            
        # 获取查询与画廊的相似度
        similarities = similarity_matrix[q_idx, gallery_indices]
        gallery_labels = [labels[i] for i in gallery_indices]
        
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


def compute_roc_curve(features, labels):
    """
    Compute ROC curve for TAR@FAR analysis.
    
    Args:
        features: image features
        labels: image labels
    
    Returns:
        tuple: (fpr, tpr, thresholds, auc_score)
    """
    n_samples = features.shape[0]
    
    # 计算所有样本对之间的相似度
    similarity_matrix = torch.mm(features, features.t()).cpu().numpy()
    
    # 生成正负样本对
    y_true = []
    y_scores = []
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            # 相同标签为正样本，不同标签为负样本
            y_true.append(1 if labels[i] == labels[j] else 0)
            y_scores.append(similarity_matrix[i, j])
    
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


def evaluate_course(course_data, demo, args):
    """
    Evaluate re-identification performance for a single course.
    
    Args:
        course_data: course data
        demo: feature extractor
        args: arguments
    
    Returns:
        dict: evaluation results
    """
    all_images = course_data['all_images']
    ground_truth_labels = course_data['ground_truth_labels']
    
    if not all_images:
        return None
    
    print(f"  提取 {len(all_images)} 张图片的特征...")
    
    # 提取所有图片的特征
    features = extract_features_batch(all_images, demo)
    
    # 执行聚类获得预测标签
    predicted_labels = perform_clustering(features, args.threshold)
    
    # 打印数据集统计信息
    unique_persons = len(set(ground_truth_labels))
    person_counts = {}
    for label in ground_truth_labels:
        person_counts[label] = person_counts.get(label, 0) + 1
    avg_images_per_person = np.mean(list(person_counts.values()))
    print(f"  数据集统计: {unique_persons}个人, 平均每人{avg_images_per_person:.1f}张图片")
    
    # 计算CMC和mAP（在同一课程内进行查询匹配）
    cmc, mAP = compute_cmc_map_single_course(features, ground_truth_labels, args.max_rank)
    
    # 计算ROC曲线
    fpr, tpr, thresholds, auc_score = compute_roc_curve(features, ground_truth_labels)
    
    # 计算相对误差
    ground_truth_count = len(set(ground_truth_labels))
    predicted_count = len(set(predicted_labels))
    relative_error = compute_relative_error(predicted_count, ground_truth_count)
    
    # 计算NMI
    nmi_score = normalized_mutual_info_score(ground_truth_labels, predicted_labels)
    
    # 计算ACC
    acc_score = compute_accuracy(ground_truth_labels, predicted_labels)
    
    # 移除聚类准确性指标（ARI和NMI）
    
    results = {
        'ground_truth_count': ground_truth_count,
        'predicted_count': predicted_count,
        'relative_error': relative_error,
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


def save_visualizations(all_results, output_path):
    """Save visualization results."""
    os.makedirs(output_path, exist_ok=True)
    
    # Set font (supports Chinese characters if present)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. CMC曲线
    plt.figure(figsize=(12, 8))
    for course_name, results in all_results.items():
        if results and 'cmc' in results:
            ranks = range(1, len(results['cmc']) + 1)
            plt.plot(ranks, results['cmc'], marker='o', label=f"{course_name} (Rank-1: {results['rank1']:.3f})")
    
    plt.xlabel('Rank')
    plt.ylabel('Recognition Rate')
    plt.title('CMC Curve')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'cmc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC曲线
    plt.figure(figsize=(12, 8))
    for course_name, results in all_results.items():
        if results and 'fpr' in results and 'tpr' in results:
            plt.plot(results['fpr'], results['tpr'], label=f"{course_name} (AUC: {results['auc']:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (TAR@FAR)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 相对误差柱状图
    course_names = []
    relative_errors = []
    ground_truth_counts = []
    predicted_counts = []
    
    for course_name, results in all_results.items():
        if results:
            course_names.append(course_name)
            relative_errors.append(results['relative_error'])
            ground_truth_counts.append(results['ground_truth_count'])
            predicted_counts.append(results['predicted_count'])
    
    if course_names:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # 相对误差
        x_pos = np.arange(len(course_names))
        ax1.bar(x_pos, relative_errors, alpha=0.7)
        ax1.set_xlabel('Course')
        ax1.set_ylabel('Relative Error')
        ax1.set_title('Person Count Relative Error by Course')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(course_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # 人数对比
        width = 0.35
        ax2.bar(x_pos - width/2, ground_truth_counts, width, label='Ground Truth', alpha=0.7)
        ax2.bar(x_pos + width/2, predicted_counts, width, label='Predicted', alpha=0.7)
        ax2.set_xlabel('Course')
        ax2.set_ylabel('Person Count')
        ax2.set_title('Person Count Comparison')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(course_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'relative_error.png'), dpi=300, bbox_inches='tight')
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
    
    # 加载数据集
    dataset = load_dataset(args.dataset_path)
    
    if not dataset:
        print("错误: 未找到有效的数据集")
        return
    
    print(f"\n找到 {len(dataset)} 个课程，开始评估...")
    
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
            results = evaluate_course(course_data, demo, args)
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
            'total_courses': len(all_results)
        }
        
        # 保存详细结果
        detailed_results = {
            'summary': summary,
            'course_results': all_results,
            'parameters': {
                'threshold': args.threshold,
                'max_rank': args.max_rank,
                'dataset_path': args.dataset_path
            }
        }
        
        results_file = os.path.join(args.output_path, 'evaluation_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        # 保存可视化结果
        if args.save_visualizations:
            save_visualizations(all_results, args.output_path)
        
        # 打印总体结果
        print("\n" + "="*60)
        print("评估完成！总体结果:")
        print("="*60)
        print(f"评估课程数: {summary['total_courses']}")
        print(f"平均 Rank-1: {summary['mean_rank1']:.4f} ± {summary['std_rank1']:.4f}")
        print(f"平均 Rank-5: {summary['mean_rank5']:.4f} ± {summary['std_rank5']:.4f}")
        print(f"平均 mAP: {summary['mean_mAP']:.4f} ± {summary['std_mAP']:.4f}")
        print(f"平均 AUC: {summary['mean_AUC']:.4f} ± {summary['std_AUC']:.4f}")
        print(f"平均 ACC: {summary['mean_ACC']:.4f} ± {summary['std_ACC']:.4f}")
        print(f"平均 NMI: {summary['mean_NMI']:.4f} ± {summary['std_NMI']:.4f}")
        print(f"平均相对误差 (RE): {summary['mean_RE']:.4f} ± {summary['std_RE']:.4f}")
        print(f"\n结果已保存到: {args.output_path}")
        print("="*60)
    else:
        print("错误: 没有成功评估任何课程")


if __name__ == '__main__':
    main()
