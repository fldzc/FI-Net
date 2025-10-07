import os
import glob
import os
import numpy as np
import torch
from models.fusion_models import ProxyFusion
from insightface.app import FaceAnalysis
from PIL import Image

# 配置参数
# IMG_DIR = r"C:\Project\Classroom-Reid\dataset\NB116_person\508NB116\Class_4\person_32"  # 修改为你的图片文件夹路径
TEST_IMG_PATH = r"C:\Project\Classroom-Reid\dataset\images\faces_images\3230637027.jpg"  # 测试图片路径
IMG_DIR = r"C:\Project\Classroom-Reid\dataset\NB116_person_xh\508NB116\Class_3\3230411002"  # 修改为你的图片文件夹路径
# TEST_IMG_PATH = r"C:\Project\Classroom-Reid\dataset\images\faces_images\3230411002.jpg"  # 测试图片路径
MODEL_PATH = r"C:\Project\Classroom-Reid\face_fusion\ProxyFusion-NeurIPS-main\checkpoints\ProxyFusion_ckpt_Adaface_RetinaFace_Proxy4x4_TAR@FARe-2_0.6910.pth.tar"  # 修改为你的ProxyFusion模型路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 提取图片特征
def extract_features_from_folder(img_dir):
    app = FaceAnalysis(name="antelopev2", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0 , det_thresh=0.4, det_size=(320, 320))
    features = []
    img_paths = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
    for img_path in img_paths:
        img = np.array(Image.open(img_path).convert("RGB"))
        faces = app.get(img)
        if len(faces) > 0:
            features.append(faces[0].embedding)
        else:
            print(f"未检测到人脸: {img_path}")

    if len(features) > 0:
        features = np.stack(features, axis=0)
        print(f"共提取到 {features.shape[0]} 个特征，维度: {features.shape[1]}")
        return features
    else:
        print("未提取到任何特征")
        return np.array([]).reshape(0, 512)

# 2. 融合特征
def fuse_features(features, model_path):
    model = ProxyFusion(DIM=512)
    model = model.to(DEVICE)
    
    try:
        # 添加自定义的persistent_load函数来处理pickle加载问题
        def persistent_load(storage, location):
            return storage
        
        # 使用pickle_module参数和自定义的persistent_load
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False, pickle_module=torch.serialization.pickle)
        
        # 根据ProxyFusion项目的标准加载方式
        if "state_dict" in checkpoint and "model_weights" in checkpoint["state_dict"]:
            model.load_state_dict(checkpoint["state_dict"]["model_weights"], strict=False)
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
            
        print("模型加载成功!")
            
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请检查模型路径是否正确，或尝试使用不同的checkpoint文件")
        print("可用的模型文件:")
        print("- ./checkpoints/ProxyFusion_ckpt_Adaface_RetinaFace_Proxy4x4_TAR@FARe-2_0.6910.pth.tar")
        print("- ./checkpoints/ProxyFusion_ckpt_Adaface_MTCNN_Proxy4x4_TAR@FARe-2_0.5265.pth.tar")
        return None
    
    model.eval()
    with torch.no_grad():
        normalized_features = []
        for feat in features:
            normalized_features.append(feat / np.linalg.norm(feat))
        normalized_features = np.array(normalized_features)
        input_tensor = torch.tensor(features, dtype=torch.float32, device=DEVICE)
        # 使用eval_fuse_probe方法进行特征融合
        # 这个方法专门用于单个特征集的融合，不需要gallery参数
        fused = model.eval_fuse_probe(input_tensor)  # [N, 512] -> [4, 512]
        # 对融合后的多个特征向量取平均，得到最终的融合特征
        fused_feature = fused.mean(dim=0).cpu().numpy()  # [4, 512] -> [512]
    print("融合后特征 shape:", fused_feature.shape)
    return fused_feature

# 提取单张图片特征
def extract_single_image_feature(img_path):
    app = FaceAnalysis(name="antelopev2", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0 , det_thresh=0.4, det_size=(320, 320))
    
    img = np.array(Image.open(img_path).convert("RGB"))
    faces = app.get(img)
    if len(faces) > 0:
        print(f"从 {img_path} 提取到特征，维度: {faces[0].embedding.shape}")
        return faces[0].embedding
    else:
        print(f"未在 {img_path} 中检测到人脸")
        return None

def compute_similarity_matrix(features1, features2):
    """计算两组特征之间的相似度矩阵"""
    # 如果任一组特征为空，返回空矩阵
    if features1.shape[0] == 0 or features2.shape[0] == 0:
        return torch.zeros((0, 0))
    
    # 计算余弦相似度矩阵
    similarity_matrix = torch.mm(features1, features2.t())
    return similarity_matrix

def compute_cosine_similarity(feature1, feature2):
    """计算两个特征向量的余弦相似度"""
    # 归一化特征向量
    feature1_norm = feature1 / np.linalg.norm(feature1)
    feature2_norm = feature2 / np.linalg.norm(feature2)
    
    # 计算余弦相似度
    similarity = np.dot(feature1_norm, feature2_norm)
    return similarity

if __name__ == "__main__":
    # 1. 提取文件夹中的特征并融合
    print("正在提取文件夹中的图片特征...")
    folder_features = extract_features_from_folder(IMG_DIR)
    
    if folder_features.shape[0] == 0:
        print("文件夹中未提取到任何特征，程序退出")
        exit()
    
    print("正在进行特征融合...")
    fused_feature = fuse_features(folder_features, MODEL_PATH)
    
    if fused_feature is None:
        print("由于模型加载失败，无法进行特征融合")
        exit()
    
    # 2. 提取测试图片特征
    print(f"\n正在提取测试图片特征: {TEST_IMG_PATH}")
    test_feature = extract_single_image_feature(TEST_IMG_PATH)
    
    if test_feature is None:
        print("测试图片特征提取失败，程序退出")
        exit()
    
    # 3. 计算相似度
    print("\n计算相似度...")
    similarity = compute_cosine_similarity(fused_feature, test_feature)
    
    print(f"\n=== 相似度结果 ===")
    print(f"融合特征维度: {fused_feature.shape}")
    print(f"测试图片特征维度: {test_feature.shape}")
    print(f"余弦相似度: {similarity:.4f}")
    
