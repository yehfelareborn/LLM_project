import numpy as np
from vector_database_faiss import FaissVectorDatabase
from embedding_generator import generate_embeddings
from config import FAISS_INDEX_FILE, EMBEDDING_MODEL

class Retrieval:
    def __init__(self):
        """
        初始化 FAISS 數據庫和嵌入模型。
        """
        self.db = FaissVectorDatabase(dimension=384)  # 嵌入維度
        self.db.load_index(FAISS_INDEX_FILE)
        self.embedding_model = EMBEDDING_MODEL
         # 加載文本數據
        self.text_data = np.load("data/embeddings/counseling_embeddings.npy", allow_pickle=True)

    def retrieve_contexts(self, question, top_k=5):
        """
        檢索與輸入問題相關的上下文。
        """
        query_embedding = generate_embeddings([question], self.embedding_model)
        distances, indices = self.db.search(query_embedding, top_k=top_k)
        print(self.text_data[0][1])
        # 通過索引獲取對應文本上下文
        # contexts = [self.text_data[idx] for idx in indices[0]]
        return distances, indices
