from fastapi import FastAPI
from pydantic import BaseModel
from data_preparation import load_and_prepare_data, get_matched_contexts
from embedding_generator import generate_embeddings
from vector_database_faiss import FaissVectorDatabase
from config import EMBEDDING_FILE, FAISS_INDEX_FILE
from retrieval import Retrieval
from generation import Generation
from fastapi.middleware.cors import CORSMiddleware

# 初始化 FastAPI 應用
app = FastAPI(debug=True)

# 啟用 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有來源。可根據需要設置特定的前端 URL。
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化檢索與生成模塊
retriever = Retrieval()
generator = Generation()
questions, answers = load_and_prepare_data()
# 請求模型
class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask_question(request: QueryRequest):
    """
    問答接口
    """
    query = request.query

    # 檢索相關上下文
    distances, indices = retriever.retrieve_contexts(query)
    matched_contexts = get_matched_contexts(distances, indices, questions, answers)
    
    if not matched_contexts:
        return {"query": query, "answer": "No relevant context found."}

    # # 使用生成模塊生成答案
    context = matched_contexts[0]['answer']  # 取第一個匹配的上下文
    answer = generator.generate_answer(query, context)

    return {
        "query": query,
        "context": context,
        "answer": answer
    }


def main():
    # 資料集準備
    questions, answers = load_and_prepare_data()
    print(f"Loaded {len(questions)} questions and {len(answers)} answers.")

    # 生成嵌入
    embeddings = generate_embeddings(questions)
    print(f"Generated embeddings for {len(questions)} questions.")

    # 向量數據庫操作
    db = FaissVectorDatabase(dimension=embeddings.shape[1])
    db.add_embeddings(embeddings)
    db.save_index(FAISS_INDEX_FILE)
    print("FAISS index saved.")


if __name__ == "__main__":
    # 如果是命令列運行，執行初始化部分
    main()

    # 如果以 API 模式運行，啟動服務
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
