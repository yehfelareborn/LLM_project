from datasets import load_dataset

def load_and_prepare_data():
    # 加載資料集
    dataset = load_dataset("Amod/mental_health_counseling_conversations")

    questions = [item['Context'] for item in dataset["train"]]
    answers = [item['Response'] for item in dataset["train"]]

    # 清理數據
    questions = [q.strip() for q in questions if q.strip()]
    answers = [a.strip() for a in answers if a.strip()]

    # 移除重複條目
    unique_data = list(set(zip(questions, answers)))
    questions, answers = zip(*unique_data)
    
    return questions, answers

def get_matched_contexts(distances, indices, questions, answers):
    """
    根據 FAISS 返回的 distances 和 indices，查找對應的提問和答案
    """
    results = []
    for dist, idx in zip(distances[0], indices[0]):  # 遍歷所有檢索結果
        question = questions[idx]  # 提問
        answer = answers[idx]    # 答案
        results.append({
            "distance": dist,
            "question": question,
            "answer": answer
        })
    return results