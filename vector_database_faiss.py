import faiss
import numpy as np

class FaissVectorDatabase:
    def __init__(self, dimension):
        """
        初始化 FAISS 向量數據庫。
        """
        self.index = faiss.IndexFlatL2(dimension)  # L2 距離索引

    def add_embeddings(self, embeddings):
        """
        將嵌入數據添加到 FAISS 索引中。
        """
        self.index.add(embeddings)

    def search(self, query_embedding, top_k=5):
        """
        在索引中查詢最相似的嵌入。
        """
        distances, indices = self.index.search(query_embedding, top_k)
        return distances, indices

    def save_index(self, filepath="data/faiss_index.index"):
        """
        將索引保存到文件。
        """
        faiss.write_index(self.index, filepath)

    def load_index(self, filepath="data/faiss_index.index"):
        """
        從文件加載索引。
        """
        self.index = faiss.read_index(filepath)
