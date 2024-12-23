from sentence_transformers import SentenceTransformer
import numpy as np

def generate_embeddings(questions, model_name="all-MiniLM-L6-v2"):
    # 加載預訓練 SentenceTransformer 模型
    model = SentenceTransformer(model_name)

    # 生成嵌入
    embeddings = model.encode(questions, show_progress_bar=True)

    # 查看嵌入維度
    print("Embedding dimension:", embeddings.shape[0])
    # 保存嵌入數據到檔案
    np.save("data/embeddings/counseling_embeddings.npy", embeddings)
    
    return embeddings