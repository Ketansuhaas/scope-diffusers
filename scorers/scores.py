import clip
from openai import OpenAI
import os
from dotenv import load_dotenv

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
        from scorers.t2v_metrics.t2v_metrics import VQAScore 
        self.model = VQAScore(model="clip-flant5-xxl")

    def name(self):
        return "vqa"

    def compute(self, image_path, text):
        return self.model(images=[image_path], texts=[text])[0].item()

class VQACompositeScorer(BaseScorer):
    def __init__(self, device="cuda"):
        super().__init__(device)
        from scorers.t2v_metrics.t2v_metrics import VQAScore

        self.model = VQAScore(model="clip-flant5-xxl")
        self.SYSTEM_PROMPT = """Given an image description, break it down into multiple sub-descriptions, each focusing on a different concept type (object, human, animal, food, activity, attribute, counting, color, material, spatial, location, shape, other). 
Return a list of these sub-descriptions separated by newlines without spaces.

Example:
Description: A serene yoga studio, 'Peace & Harmony' inscribed on the wall, mats laid out in neat rows on the wooden floor, gentle sunlight filtering through large windows, potted plants in the corners, light-colored walls enhancing the calm atmosphere, a soft breeze slightly moving the sheer curtains.

Generated sub-descriptions:
Yoga mats laid out in neat rows on the floor\nLarge windows allowing sunlight to filter in\nPotted plants placed in the corners of the room\nSheer curtains hanging by the windows\nYoga practice implied by the studio setting and mats\nThe atmosphere of the studio is serene and calm\nWalls are light-colored, enhancing the peaceful feeling\nCurtains are sheer and move slightly in the breeze\nMats are arranged in multiple rows\nWalls are light-colored\nThe floor is made of wood\nMats are laid out in neat rows across the floor\nPlants are positioned in the corners of the room\nSunlight enters through the large window\nInterior of a yoga studio named 'Peace & Harmony'\n'Peace & Harmony' is inscribed on the wall
"""     
        load_dotenv()
        env = os.environ
        self.client = OpenAI(api_key=env.get("OPENAI_API_KEY"))

    def name(self):
        return "vqa_composite"

    def get_subdescriptions(self, final_prompt):
        
        try:
            messages = [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Description: {final_prompt}\n\nGenerated sub-descriptions:\n"}
                ]

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
        
            response_text = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API Error: {e}")
            return None
        
        try:
            generated_subdescriptions = response_text.split('\n')
            # Clean up the sub-descriptions by stripping whitespace and removing empty strings
            generated_subdescriptions = [desc.strip() for desc in generated_subdescriptions if desc.strip()]
            # Ensure the list is not empty
            if not generated_subdescriptions:
                print("VQAComposite Scorer: No valid sub-descriptions generated. Returning original final prompt.")
                return [final_prompt]
        except Exception as e:
            print(f"Error parsing generated sub-descriptions: {e}")
            return None
        
        return generated_subdescriptions

    def compute(self, image_path, texts):

        if isinstance(texts, str): 
            texts = self.get_subdescriptions(texts)

        scores = []
        for t in texts:
            score = self.model(images=[image_path], texts=[t])[0].item()
            scores.append(score)
        score = sum(scores) / len(scores)
        return score

# === VQA-style scorers from scorers.t2v_metrics ===

class LlavaScorer(BaseScorer):
    def __init__(self, device="cuda"):
        super().__init__(device)
        from scorers.t2v_metrics.t2v_metrics import VQAScore

    def compute(self, image_path, text):
        return float(self.model(images=[image_path], texts=[text])[0])


class InstructBLIPScorer(BaseScorer):
    def __init__(self, device="cuda"):
        super().__init__(device)
        from scorers.t2v_metrics.t2v_metrics import VQAScore
        self.model = VQAScore(model="instructblip-flant5-xxl")

    def name(self):
        return "instructblip"

    def compute(self, image_path, text):
        return float(self.model(images=[image_path], texts=[text])[0])


# === CLIP-style scorers ===

class PickScoreScorer(BaseScorer):
    def __init__(self, device="cuda"):
        super().__init__(device)
        from scorers.t2v_metrics.t2v_metrics import CLIPScore
        self.model = CLIPScore(model="pickscore-v1")

    def name(self):
        return "pickscore"

    def compute(self, image_path, text):
        return float(self.model(images=[image_path], texts=[text])[0])


class HPSv2Scorer(BaseScorer):
    def __init__(self, device="cuda"):
        super().__init__(device)
        from scorers.t2v_metrics.t2v_metrics import CLIPScore
        self.model = CLIPScore(model="hpsv2")

    def name(self):
        return "hpsv2"

    def compute(self, image_path, text):
        return float(self.model(images=[image_path], texts=[text])[0])


class CLIPLarge336Scorer(BaseScorer):
    def __init__(self, device="cuda"):
        super().__init__(device)
        from scorers.t2v_metrics.t2v_metrics import CLIPScore
        self.model = CLIPScore(model="openai:ViT-L-14-336")

    def name(self):
        return "clip_336"

    def compute(self, image_path, text):
        return float(self.model(images=[image_path], texts=[text])[0])


# === ITM-style scorers ===

class BLIPITMScorer(BaseScorer):
    def __init__(self, device="cuda"):
        super().__init__(device)
        from scorers.t2v_metrics.t2v_metrics import ITMScore
        self.model = ITMScore(model="blip2-itm")

    def name(self):
        return "blip_itm"

    def compute(self, image_path, text):
        return float(self.model(images=[image_path], texts=[text])[0])


class ImageRewardScorer(BaseScorer):
    def __init__(self, device="cuda"):
        super().__init__(device)
        from scorers.t2v_metrics.t2v_metrics import ITMScore
        self.model = ITMScore(model="image-reward-v1")

    def name(self):
        return "image_reward"

    def compute(self, image_path, text):
        return float(self.model(images=[image_path], texts=[text])[0])