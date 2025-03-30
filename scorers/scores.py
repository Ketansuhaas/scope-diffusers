import clip

class BaseScorer:
    def __init__(self, device="cuda"):
        self.device = device

    def name(self):
        raise NotImplementedError

    def compute(self, image_path: str, text: str) -> float:
        raise NotImplementedError

class CLIPScorer(BaseScorer):
    def __init__(self, model_name="ViT-L/14", device="cuda"):
        super().__init__(device)
        import clip
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

    def name(self):
        return "clip"

    def compute(self, image_path, text):
        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        text_tokens = clip.tokenize([text], truncate=True).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return (image_features @ text_features.T).item()

class VQAScorer(BaseScorer):
    def __init__(self, device="cuda"):
        super().__init__(device)
        import t2v_metrics
        self.model = t2v_metrics.VQAScore(model="clip-flant5-xxl")

    def name(self):
        return "vqa"

    def compute(self, image_path, text):
        return self.model(images=[image_path], texts=[text])[0]