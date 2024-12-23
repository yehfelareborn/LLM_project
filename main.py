from data_preparation import load_and_prepare_data, get_matched_contexts
from embedding_generator import generate_embeddings
from vector_database_faiss import FaissVectorDatabase
from config import EMBEDDING_MODEL, EMBEDDING_FILE, FAISS_INDEX_FILE
from retrieval import Retrieval
from generation import Generation


def main():
    # 1. 資料集準備
    questions, answers = load_and_prepare_data()
    print(f"Loaded {len(questions)} questions and {len(answers)} answers.")

    # 2. 生成嵌入
    embeddings = generate_embeddings(questions)
    print(embeddings.shape)
    print(f"Generated embeddings for {len(questions)} questions.")

    # 3. 向量數據庫操作
    db = FaissVectorDatabase(dimension=embeddings.shape[1])
    db.add_embeddings(embeddings)
    db.save_index(FAISS_INDEX_FILE)
    print("FAISS index saved.")

    # 測試查詢
    # test_query = "How do I deal with negative thoughts that keep coming back?"
    # query_embedding = generate_embeddings([test_query], EMBEDDING_MODEL)
    # distances, indices = db.search(query_embedding)
    # print(f"Query results: {indices}, Distances: {distances}")

    # 第二部分
    # 初始化模塊
    retriever = Retrieval()
    generator = Generation()
    # multimodal = MultimodalSupport()

    # 輸入問題和圖片
    question = "How can I rebuild trust in my relationships?"
    # image_path = "data/images/orange_cat.jpg"
    # image = Image.open(image_path)

    # 檢索文本上下文
    distances, indices = retriever.retrieve_contexts(question)
    print(f"distances: {distances}")
    print(f"contexts: {indices}")
    result = get_matched_contexts(distances, indices, questions, answers)
    print(f"Answer: {result[0]['answer']}")

    # 生成答案
    answer = generator.generate_answer(question, result[0]['answer'])
    print(f"Answer: {answer}")

    # # 處理圖像模態
    # image_embedding = multimodal.get_image_embedding(image, question)
    # print(f"Image embedding: {image_embedding}")


if __name__ == "__main__":
    main()
