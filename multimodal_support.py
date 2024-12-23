from transformers import CLIPProcessor, CLIPModel

class MultimodalSupport:
    def __init__(self):
        """
        初始化 CLIP 模型和處理器。
        """
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def get_image_embedding(self, image, question):
        """
        將圖片和文本嵌入到向量空間。
        """
        inputs = self.clip_processor(text=[question], images=[image], return_tensors="pt")
        outputs = self.clip_model(**inputs)
        return outputs.pooler_output.detach().numpy()

    def combine_results(self, text_results, image_results):
        """
        將文本和圖像的檢索結果整合。
        """
        # 此處可使用權重或其他融合方式
        return text_results + image_results
