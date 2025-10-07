import os
import glob
import cv2
import numpy as np
import torch
from models.fusion_models import ProxyFusion
from insightface.app import FaceAnalysis
from PIL import Image


# 配置参数
# IMG_DIR = r"C:\Project\Classroom-Reid\dataset\NB116_person\508NB116\Class_4\person_32"  # 修改为你的图片文件夹路径
# TEST_IMG_PATH = r"C:\Project\Classroom-Reid\dataset\images\faces_images\3230637027.jpg"  # 测试图片路径
IMG_DIR = r"C:\Project\Classroom-Reid\dataset\NB116_person_xh\508NB116\Class_3\3230411002"  # 修改为你的图片文件夹路径
TEST_IMG_PATH = r"C:\Project\Classroom-Reid\dataset\images\faces_images\3230411002.jpg"  # 测试图片路径
MODEL_PATH = r"C:\Project\Classroom-Reid\face_fusion\ProxyFusion-NeurIPS-main\checkpoints\ProxyFusion_ckpt_Adaface_RetinaFace_Proxy4x4_TAR@FARe-2_0.6910.pth.tar"  # 修改为你的ProxyFusion模型路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============ 质量评估函数 ============
def compute_quality(image):
    """计算人脸图像的简单质量分数"""
    if image is None:
        return 0
    
    # 1. 清晰度（Laplacian 方差）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
    clarity_score = min(1.0, clarity / 1000)  # 归一化

    # 2. 亮度（避免过暗/过亮）
    brightness = np.mean(gray)
    brightness_score = 1 - abs(brightness - 128) / 128

    # 综合质量分数
    return 0.6 * clarity_score + 0.4 * brightness_score


def compute_pose_quality(yaw, pitch, roll):
    """
    计算基于人脸姿态角度的质量分数
    Args:
        yaw: 偏航角 (左右转头) 范围一般在 [-90, 90] 度
        pitch: 俯仰角 (上下点头) 范围一般在 [-90, 90] 度  
        roll: 翻滚角 (左右歪头) 范围一般在 [-180, 180] 度
    Returns:
        pose_score: 姿态质量分数 [0, 1]，越接近正脸质量越高
    """
    # 将角度转换为弧度计算
    yaw_rad = abs(np.radians(yaw))
    pitch_rad = abs(np.radians(pitch)) 
    roll_rad = abs(np.radians(roll))
    
    # 理想的正脸姿态角度都应该接近0
    # 使用高斯函数计算各角度的质量分数，标准差可调节容忍度
    yaw_score = np.exp(-(yaw_rad**2) / (2 * (np.radians(30))**2))      # yaw容忍度±30度
    pitch_score = np.exp(-(pitch_rad**2) / (2 * (np.radians(20))**2))  # pitch容忍度±20度  
    roll_score = np.exp(-(roll_rad**2) / (2 * (np.radians(15))**2))    # roll容忍度±15度
    
    # 综合姿态分数 (可调整各角度权重)
    pose_score = 0.4 * yaw_score + 0.3 * pitch_score + 0.3 * roll_score
    
    return pose_score


def get_face_pose_angles(face):
    """
    从insightface的face对象中提取姿态角度
    Args:
        face: insightface检测到的人脸对象
    Returns:
        yaw, pitch, roll: 三个姿态角度，如果无法获取则返回(0, 0, 0)
    """
    try:
        # 方法1：直接从pose属性获取 (如果存在)
        if hasattr(face, 'pose') and face.pose is not None:
            if len(face.pose) >= 3:
                yaw, pitch, roll = face.pose[:3]
                return float(yaw), float(pitch), float(roll)
        
        # 方法2：从关键点计算姿态角度 (备选方案)
        if hasattr(face, 'kps') and face.kps is not None:
            # 使用5个关键点估算姿态角度
            kps = face.kps.reshape(-1, 2)
            if len(kps) >= 5:
                # 简单的姿态估计：基于眼睛和鼻子的相对位置
                left_eye = kps[0]   # 左眼
                right_eye = kps[1]  # 右眼
                nose = kps[2]       # 鼻子
                left_mouth = kps[3] # 左嘴角
                right_mouth = kps[4] # 右嘴角
                
                # 计算yaw角度 (基于眼睛间距和脸部中心的偏移)
                eye_center = (left_eye + right_eye) / 2
                eye_distance = np.linalg.norm(right_eye - left_eye)
                face_center_x = (left_eye[0] + right_eye[0] + nose[0]) / 3
                yaw = np.degrees(np.arctan2(nose[0] - eye_center[0], eye_distance)) * 2
                
                # 计算pitch角度 (基于眼睛和嘴部的垂直关系)
                mouth_center = (left_mouth + right_mouth) / 2
                eye_mouth_distance = mouth_center[1] - eye_center[1]
                expected_distance = eye_distance * 1.2  # 正常比例
                pitch = np.degrees(np.arctan2(eye_mouth_distance - expected_distance, expected_distance)) * 1.5
                
                # 计算roll角度 (基于眼睛连线的倾斜度)
                roll = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
                
                return float(yaw), float(pitch), float(roll)
        
        # 如果都无法获取，返回默认值
        return 0.0, 0.0, 0.0
        
    except Exception as e:
        print(f"提取姿态角度时出错: {e}")
        return 0.0, 0.0, 0.0


def get_comprehensive_quality_score(image, face):
    """综合质量评估：结合图像质量、检测置信度和人脸姿态"""
    if image is None or face is None:
        return 0
    
    # 1. 图像质量分数
    image_quality = compute_quality(image)
    
    # 2. 人脸检测置信度
    detection_confidence = float(face.det_score)
    
    # 3. 人脸姿态质量分数
    yaw, pitch, roll = get_face_pose_angles(face)
    pose_quality = compute_pose_quality(yaw, pitch, roll)
    
    # 4. 线性组合综合评分 (权重可调整)
    w_image = 0.4      # 图像质量权重
    w_detection = 0.3  # 检测置信度权重
    w_pose = 0.3       # 姿态质量权重
    
    comprehensive_score = (w_image * image_quality + 
                          w_detection * detection_confidence + 
                          w_pose * pose_quality)
    
    return comprehensive_score



def apply_quality_weights_v1(features_list, quality_scores):   
    """
    方案1: 保持模长的质量加权
    features_list: [N, D] numpy array
    quality_scores: [N] numpy array
    """
    if len(quality_scores) == 0:
        return features_list
    
    # 归一化质量分数作为权重
    quality_weights = np.array(quality_scores)
    quality_weights = quality_weights / (np.sum(quality_weights) + 1e-8)
    
    # 对每个特征应用权重，然后重新归一化保持模长
    weighted_features = []
    for i, feat in enumerate(features_list):
        # 先归一化原特征
        feat_norm = feat / np.linalg.norm(feat)
        # 应用权重
        weighted_feat = feat_norm * quality_weights[i]
        # 重新归一化，保持单位向量
        if np.linalg.norm(weighted_feat) > 1e-8:
            weighted_feat = weighted_feat / np.linalg.norm(weighted_feat)
        weighted_features.append(weighted_feat)
    
    return np.array(weighted_features)


def apply_quality_weights_v2(features_list, quality_scores):
    """
    方案2: 加权平均融合（推荐）
    features_list: [N, D] numpy array
    quality_scores: [N] numpy array
    """
    if len(quality_scores) == 0:
        return features_list
    
    # 归一化质量分数作为权重
    quality_weights = np.array(quality_scores)
    quality_weights = quality_weights / (np.sum(quality_weights) + 1e-8)
    
    # 先归一化所有特征
    normalized_features = []
    for feat in features_list:
        normalized_features.append(feat / np.linalg.norm(feat))
    normalized_features = np.array(normalized_features)
    
    # 加权平均
    weighted_avg = np.average(normalized_features, axis=0, weights=quality_weights)
    # 归一化结果
    weighted_avg = weighted_avg / np.linalg.norm(weighted_avg)
    
    # 返回单个融合特征，而不是特征列表
    return weighted_avg


def apply_quality_weights_v3(features_list, quality_scores):
    """
    方案3: 选择性特征融合
    features_list: [N, D] numpy array  
    quality_scores: [N] numpy array
    """
    if len(quality_scores) == 0:
        return features_list
    
    # 设置质量阈值，只保留高质量特征
    quality_threshold = np.mean(quality_scores)
    
    high_quality_indices = quality_scores >= quality_threshold
    if np.sum(high_quality_indices) == 0:
        # 如果没有高质量特征，使用所有特征
        high_quality_indices = np.ones_like(quality_scores, dtype=bool)
    
    # 筛选高质量特征
    selected_features = features_list[high_quality_indices]
    selected_scores = quality_scores[high_quality_indices]
    
    # 对筛选后的特征进行加权平均
    quality_weights = selected_scores / (np.sum(selected_scores) + 1e-8)
    
    normalized_features = []
    for feat in selected_features:
        normalized_features.append(feat / np.linalg.norm(feat))
    normalized_features = np.array(normalized_features)
    
    # 加权平均
    weighted_avg = np.average(normalized_features, axis=0, weights=quality_weights)
    weighted_avg = weighted_avg / np.linalg.norm(weighted_avg)
    
    print(f"选择性融合: 使用了 {len(selected_features)}/{len(features_list)} 个高质量特征")
    return weighted_avg


# 默认使用方案2（推荐）
def apply_quality_weights(features_list, quality_scores, method='v1'):
    """
    质量加权方法选择器
    method: 'v1' (保持模长), 'v2' (加权平均-推荐), 'v3' (选择性融合)
    """
    if method == 'v1':
        return apply_quality_weights_v1(features_list, quality_scores)
    elif method == 'v2':
        return apply_quality_weights_v2(features_list, quality_scores)
    elif method == 'v3':
        return apply_quality_weights_v3(features_list, quality_scores)
    else:
        return apply_quality_weights_v2(features_list, quality_scores)  # 默认使用v2

# ============ 特征提取函数 ============
def extract_features_from_folder_with_quality(img_dir):
    """提取文件夹中的图片特征，包含质量评估"""
    app = FaceAnalysis(name="antelopev2", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_thresh=0.4, det_size=(320, 320))
    
    features = []
    quality_scores = []
    valid_images = []
    skipped_images = []
    
    img_paths = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
    
    print(f"找到 {len(img_paths)} 张图片")
    
    for img_path in img_paths:
        try:
            # 加载图像
            image_cv = cv2.imread(img_path)
            image_pil = np.array(Image.open(img_path).convert("RGB"))
            
            if image_cv is None:
                print(f"无法加载图像: {img_path}")
                skipped_images.append(os.path.basename(img_path))
                continue
            
            # 检测人脸
            faces = app.get(image_pil)
            
            if len(faces) > 1:
                print(f"跳过多人脸图片: {os.path.basename(img_path)} (检测到 {len(faces)} 张人脸)")
                skipped_images.append(os.path.basename(img_path))
                continue
            elif len(faces) == 1:
                best_face = faces[0]
                
                # 计算综合质量分数
                quality_score = get_comprehensive_quality_score(image_cv, best_face)
                
                # 获取详细的质量分数组成
                image_quality = compute_quality(image_cv)
                detection_confidence = float(best_face.det_score)
                yaw, pitch, roll = get_face_pose_angles(best_face)
                pose_quality = compute_pose_quality(yaw, pitch, roll)
                
                features.append(best_face.embedding)
                quality_scores.append(quality_score)
                valid_images.append(os.path.basename(img_path))
                
                print(f"✓ {os.path.basename(img_path)}: 总分 {quality_score:.3f} " +
                      f"(图像:{image_quality:.3f}, 检测:{detection_confidence:.3f}, " + 
                      f"姿态:{pose_quality:.3f} [Y:{yaw:.1f}°,P:{pitch:.1f}°,R:{roll:.1f}°])")
            else:
                print(f"未检测到人脸: {os.path.basename(img_path)}")
                skipped_images.append(os.path.basename(img_path))
                
        except Exception as e:
            print(f"处理图片错误 {img_path}: {e}")
            skipped_images.append(os.path.basename(img_path))
            continue

    if len(features) > 0:
        features = np.stack(features, axis=0)
        quality_scores = np.array(quality_scores)
        
        print(f"\n=== 特征提取统计 ===")
        print(f"总图片数: {len(img_paths)}")
        print(f"有效单人脸图片: {len(valid_images)}")
        print(f"跳过图片: {len(skipped_images)}")
        print(f"提取特征维度: {features.shape}")
        print(f"质量分数范围: {np.min(quality_scores):.3f} - {np.max(quality_scores):.3f}")
        print(f"平均质量分数: {np.mean(quality_scores):.3f}")
        
        return features, quality_scores, valid_images, skipped_images
    else:
        print("未提取到任何特征")
        return np.array([]).reshape(0, 512), np.array([]), [], skipped_images


def fuse_features_with_quality(features, quality_scores, model_path, fusion_method='hybrid'):
    """
    使用质量加权进行特征融合
    fusion_method: 'hybrid' (ProxyFusion+质量加权), 'quality_only' (纯质量加权), 'proxyfusion_only' (纯ProxyFusion)
    """
    print(f"\n=== 开始质量加权特征融合 ===")
    print(f"输入特征形状: {features.shape}")
    print(f"质量分数: {quality_scores}")
    print(f"融合方法: {fusion_method}")
    
    # 方法1: 纯质量加权融合（不使用ProxyFusion）
    if fusion_method == 'quality_only':
        print("使用纯质量加权融合")
        if len(features) > 1:
            fused_feature = apply_quality_weights(features, quality_scores, method='v1')
            print(f"✓ 使用质量加权平均融合了 {len(features)} 个特征")
            return fused_feature
        else:
            single_feature = features[0] / np.linalg.norm(features[0])
            return single_feature
    
    # 加载ProxyFusion模型
    model = ProxyFusion(DIM=512)
    model = model.to(DEVICE)
    
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        
        if "state_dict" in checkpoint and "model_weights" in checkpoint["state_dict"]:
            model.load_state_dict(checkpoint["state_dict"]["model_weights"], strict=False)
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
            
        print("✓ ProxyFusion模型加载成功!")
            
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        print("将使用质量加权平均融合作为备选方案")
        
        # 备选方案：质量加权平均
        if len(features) > 1:
            fused_feature = apply_quality_weights(features, quality_scores, method='v2')
            print(f"✓ 使用质量加权平均融合了 {len(features)} 个特征")
            return fused_feature
        else:
            single_feature = features[0] / np.linalg.norm(features[0])
            return single_feature
    
    # 使用ProxyFusion进行融合
    model.eval()
    with torch.no_grad():
        if len(features) > 1:
            if fusion_method == 'hybrid':
                # 混合方法：先质量加权，再ProxyFusion
                print("使用混合融合方法：先质量选择，再ProxyFusion")
                
                # 选择高质量特征用于ProxyFusion
                quality_threshold = np.percentile(quality_scores, 20)  # 使用中位数作为阈值
                high_quality_indices = quality_scores >= quality_threshold
                
                if np.sum(high_quality_indices) >= 2:
                    # 有足够的高质量特征，使用筛选后的特征
                    selected_features = features[high_quality_indices]
                    print(f"选择了 {len(selected_features)}/{len(features)} 个高质量特征用于ProxyFusion")
                else:
                    # 高质量特征不足，使用所有特征
                    selected_features = features
                    print(f"使用所有 {len(features)} 个特征进行ProxyFusion")
                
                # 归一化特征
                normalized_features = []
                for feat in selected_features:
                    normalized_features.append(feat / np.linalg.norm(feat))
                normalized_features = np.array(normalized_features)
                
                input_tensor = torch.tensor(normalized_features, dtype=torch.float32, device=DEVICE)
                fused = model.eval_fuse_probe(input_tensor)
                fused_feature = fused.mean(dim=0).cpu().numpy()
                fused_feature = fused_feature / np.linalg.norm(fused_feature)
                print(f"✓ 使用混合方法融合完成")
                
            elif fusion_method == 'proxyfusion_only':
                # 纯ProxyFusion方法（不考虑质量）
                print("使用纯ProxyFusion融合")
                normalized_features = []
                for feat in features:
                    normalized_features.append(feat / np.linalg.norm(feat))
                normalized_features = np.array(normalized_features)
                
                input_tensor = torch.tensor(normalized_features, dtype=torch.float32, device=DEVICE)
                fused = model.eval_fuse_probe(input_tensor)
                fused_feature = fused.mean(dim=0).cpu().numpy()
                fused_feature = fused_feature / np.linalg.norm(fused_feature)
                print(f"✓ 使用纯ProxyFusion融合了 {len(features)} 个特征")
            
        else:
            # 只有一个特征，直接归一化返回
            fused_feature = features[0] / np.linalg.norm(features[0])
            print(f"✓ 只有一个特征，直接归一化返回")
    
    print(f"融合后特征形状: {fused_feature.shape}")
    return fused_feature


def extract_single_image_feature_with_quality(img_path):
    """提取单张图片特征，包含质量评估"""
    app = FaceAnalysis(name="antelopev2", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_thresh=0.4, det_size=(320, 320))
    
    try:
        # 加载图像
        image_cv = cv2.imread(img_path)
        image_pil = np.array(Image.open(img_path).convert("RGB"))
        
        if image_cv is None:
            print(f"无法加载图像: {img_path}")
            return None, 0
            
        # 检测人脸
        faces = app.get(image_pil)
        
        if len(faces) > 0:
            best_face = faces[0]
            quality_score = get_comprehensive_quality_score(image_cv, best_face)
            
            # 获取详细质量信息
            image_quality = compute_quality(image_cv)
            detection_confidence = float(best_face.det_score)
            yaw, pitch, roll = get_face_pose_angles(best_face)
            pose_quality = compute_pose_quality(yaw, pitch, roll)
            
            print(f"从 {img_path} 提取到特征")
            print(f"特征维度: {best_face.embedding.shape}")
            print(f"总质量分数: {quality_score:.3f}")
            print(f"  - 图像质量: {image_quality:.3f}")
            print(f"  - 检测置信度: {detection_confidence:.3f}")
            print(f"  - 姿态质量: {pose_quality:.3f} (Yaw:{yaw:.1f}°, Pitch:{pitch:.1f}°, Roll:{roll:.1f}°)")
            
            return best_face.embedding, quality_score
        else:
            print(f"未在 {img_path} 中检测到人脸")
            return None, 0
            
    except Exception as e:
        print(f"处理图片错误 {img_path}: {e}")
        return None, 0


def compute_cosine_similarity(feature1, feature2):
    """计算两个特征向量的余弦相似度"""
    # 归一化特征向量
    feature1_norm = feature1 / np.linalg.norm(feature1)
    feature2_norm = feature2 / np.linalg.norm(feature2)
    
    # 计算余弦相似度
    similarity = np.dot(feature1_norm, feature2_norm)
    return similarity


if __name__ == "__main__":
    print("=== ProxyFusion 质量加权特征融合测试 ===")
    print(f"图片文件夹: {IMG_DIR}")
    print(f"测试图片: {TEST_IMG_PATH}")
    print(f"设备: {DEVICE}")
    
    # 1. 提取文件夹中的特征并进行质量评估
    print("\n正在提取文件夹中的图片特征...")
    folder_features, quality_scores, valid_images, skipped_images = extract_features_from_folder_with_quality(IMG_DIR)
    
    if folder_features.shape[0] == 0:
        print("文件夹中未提取到任何特征，程序退出")
        exit()
    
    # 2. 使用质量加权进行特征融合
    print("\n正在进行质量加权特征融合...")
    fused_feature = fuse_features_with_quality(folder_features, quality_scores, MODEL_PATH)
    
    if fused_feature is None:
        print("特征融合失败，程序退出")
        exit()
    
    # 3. 提取测试图片特征
    print(f"\n正在提取测试图片特征: {TEST_IMG_PATH}")
    test_feature, test_quality = extract_single_image_feature_with_quality(TEST_IMG_PATH)
    
    if test_feature is None:
        print("测试图片特征提取失败，程序退出")
        exit()
    
    # 4. 计算相似度
    print("\n=== 计算相似度 ===")
    similarity = compute_cosine_similarity(fused_feature, test_feature)
    
    # 5. 输出详细结果
    print(f"\n=== 最终结果 ===")
    print(f"融合特征来源:")
    for i, img_name in enumerate(valid_images):
        print(f"  {i+1}. {img_name} (质量: {quality_scores[i]:.3f})")
    
    if skipped_images:
        print(f"\n跳过的图片 ({len(skipped_images)}个):")
        for img_name in skipped_images:
            print(f"  - {img_name}")
    
    print(f"\n融合特征维度: {fused_feature.shape}")
    print(f"测试图片特征维度: {test_feature.shape}")
    print(f"测试图片质量分数: {test_quality:.3f}")
    print(f"余弦相似度: {similarity:.4f}")
    