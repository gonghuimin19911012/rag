# 向量化过程
# -*- coding: utf-8 -*-
"""
BGE-Large-ZH-V1.5 本地向量化封装
支持单文本和批量向量化
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import threading
import torch

import numpy as np
from typing import Union, List, Optional
from sentence_transformers import SentenceTransformer
from config import bge_model

model_path=bge_model['path']

# Round-robin GPU 分配：让多个 worker 均匀分到多张卡，避免全部挤在同一张卡 OOM
# 注意：不要用"选剩余显存最多的 GPU"——每个进程只能看到自己上下文，
# N 个进程同时选会全选 cuda:0，依然 OOM
_gpu_counter = 0
_gpu_counter_lock = threading.Lock()
_gpu_total = torch.cuda.device_count() if torch.cuda.is_available() else 0


def _alloc_next_gpu() -> str:
    """轮询分配下一个 GPU；无 CUDA 时回退到 CPU"""
    global _gpu_counter
    if _gpu_total == 0:
        return "cpu"
    with _gpu_counter_lock:
        gpu_id = _gpu_counter % _gpu_total
        _gpu_counter += 1
    return f"cuda:{gpu_id}"

class BGEEmbedder:
    """BGE 向量化模型封装类"""
    def __init__(self,model_path: str,device: Optional[str] = None,cache_folder: Optional[str] = None): #model_path: str
        """
        初始化向量化模型
        :param model_path: 本地模型路径
        :param device: 设备 (cuda/cpu/mps)，None 则自动选择
        :param cache_folder: 模型缓存目录
        """
        # 验证模型路径
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在：{model_path}")
        
        # 自动选择设备：round-robin 分配，避免所有 worker 挤在同一张 GPU 上 OOM
        if device is None:
            device = _alloc_next_gpu()
        print(f"embedding device：{device}")
        
        self.device = device
        self.model_path = model_path
        
        print(f"🔧 加载模型：{model_path}")
        print(f"💻 使用设备：{device}")
        
        # 加载模型
        self.model = SentenceTransformer(model_path,device=device,cache_folder=cache_folder)
        
        # 获取嵌入维度
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"📏 嵌入维度：{self.dimension}")
        print(f"✅ 模型加载成功")
    
    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        单个文本向量化
        
        :param text: 输入文本
        :param normalize: 是否归一化（余弦相似度推荐 True）
        :return: 一维 numpy 数组 (dimension,)
        """
        if not text or not isinstance(text, str):
            raise ValueError("输入文本不能为空")
        
        embedding = self.model.encode(text,normalize_embeddings=normalize,convert_to_numpy=True)
        
        return embedding
    
    def encode_batch(self, texts: List[str], normalize: bool = True,batch_size: int = 32,show_progress: bool = False) -> np.ndarray:
        """
        批量文本向量化
        
        :param texts: 文本列表
        :param normalize: 是否归一化
        :param batch_size: 批处理大小
        :param show_progress: 显示进度条
        :return: 二维 numpy 数组 (n_texts, dimension)
        """
        if not texts or not isinstance(texts, list):
            raise ValueError("输入必须是文本列表")
        
        if len(texts) == 0:
            return np.array([]).reshape(0, self.dimension)
        
        # 过滤空文本
        valid_texts = [t for t in texts if t and isinstance(t, str)]
        if len(valid_texts) == 0:
            return np.array([]).reshape(0, self.dimension)
        
        embeddings = self.model.encode(
            valid_texts,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        print(f"embeddings:批量完成！")
        return embeddings
    
    def encode(self, texts: Union[str, List[str]], normalize: bool = True,batch_size: int = 32,show_progress: bool = False) -> np.ndarray:
        """
        智能编码：自动判断单文本或批量
        
        :param texts: 单个文本或文本列表
        :param normalize: 是否归一化
        :param batch_size: 批处理大小
        :param show_progress: 显示进度条
        :return: numpy 数组
        """
        if isinstance(texts, str):
            return self.encode_single(texts, normalize)
        elif isinstance(texts, list):
            return self.encode_batch(texts, normalize, batch_size, show_progress)
        else:
            raise TypeError("输入必须是字符串或字符串列表")
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度
        
        :param embedding1: 向量 1
        :param embedding2: 向量 2
        :return: 相似度分数 (0-1)
        """
        # 如果未归一化，手动计算余弦相似度
        if np.linalg.norm(embedding1) != 1 or np.linalg.norm(embedding2) != 1:
            from sklearn.metrics.pairwise import cosine_similarity
            sim = cosine_similarity([embedding1], [embedding2])[0][0]
        else:
            # 已归一化，直接点积
            sim = np.dot(embedding1, embedding2)
        
        return float(sim)
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """保存向量到文件"""
        np.save(filepath, embeddings)
        print(f"💾 向量已保存：{filepath}")
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """从文件加载向量"""
        return np.load(filepath)
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "dimension": self.dimension,
            "max_seq_length": self.model.max_seq_length
        }


# 全局单例（懒加载，进程内只初始化一次）
_embedder_instance = None

def get_embedder() -> BGEEmbedder:
    """获取全局 embedder 单例（懒加载）"""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = BGEEmbedder(model_path)
    return _embedder_instance

def get_embed(query):
    embedder = get_embedder()
    embeddings = embedder.encode(query,normalize=True,batch_size=32)
    # 转换为列表
    try:
        if isinstance(query, str):
            emb_list = embeddings.tolist()
            count = 1
        else: # LIST,批量处理
            emb_list = embeddings.tolist()
            count = len(embeddings)
        print("dimension",embedder.dimension,embeddings.shape[-1])
        print("embeddings count",count)
        return {"success": True, "embeddings":emb_list}
    except Exception as e:
        print(f"向量化出错：{e}")
        return {"success": False,"data":f"向量化出错:{e}"}

if __name__ == "__main__":
    # ========== 测试代码 ==========
    
    # 配置
    MODEL_PATH = "./models/bge-large-zh-v1.5"  # 修改为你的本地模型路径
    
    # 1. 初始化模型
    embedder = BGEEmbedder(MODEL_PATH)
    
    # 2. 单文本向量化
    print("\n" + "="*60)
    print("【测试 1】单文本向量化")
    print("="*60)
    text = "北京是中国的首都"
    embedding = embedder.encode_single(text)
    print(f"文本：{text}")
    print(f"向量形状：{embedding.shape}")
    print(f"向量前 5 维：{embedding[:5]}")
    
    # 3. 批量向量化
    print("\n" + "="*60)
    print("【测试 2】批量向量化")
    print("="*60)
    texts = [
        "北京是中国的首都",
        "上海是中国的经济中心",
        "Python 是一种编程语言",
        "机器学习是人工智能的分支",
        "深度学习需要大量数据"
    ]
    embeddings = embedder.encode_batch(texts, show_progress=True)
    print(f"文本数量：{len(texts)}")
    print(f"向量形状：{embeddings.shape}")
    print(f"向量前 5 维（第一条）：{embeddings[0][:5]}")
    
    # 4. 智能编码（自动判断）
    print("\n" + "="*60)
    print("【测试 3】智能编码")
    print("="*60)
    # 单文本
    emb1 = embedder.encode("单个文本")
    print(f"单文本向量形状：{emb1.shape}")
    # 批量
    emb2 = embedder.encode(["文本 1", "文本 2", "文本 3"])
    print(f"批量向量形状：{emb2.shape}")
    
    # 5. 计算相似度
    print("\n" + "="*60)
    print("【测试 4】相似度计算")
    print("="*60)
    sim = embedder.compute_similarity(embeddings[0], embeddings[1])
    print(f"文本 1 和文本 2 的相似度：{sim:.4f}")
    
    # 6. 保存和加载
    print("\n" + "="*60)
    print("【测试 5】保存和加载向量")
    print("="*60)
    embedder.save_embeddings(embeddings, "test_embeddings.npy")
    loaded = embedder.load_embeddings("test_embeddings.npy")
    print(f"加载的向量形状：{loaded.shape}")
    
    
    print("\n✅ 所有测试完成！")
