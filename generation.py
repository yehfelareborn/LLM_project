from transformers import pipeline

class Generation:
    def __init__(self, model_name="distilbert-base-uncased-distilled-squad"):
        """
        初始化 QA 生成模型。
        """
        self.qa_pipeline = pipeline("question-answering", model=model_name)

    def generate_answer(self, question, context):
        """
        使用生成模型返回答案。
        """
        result = self.qa_pipeline(question=question, context=context)
        return result
