from dataclasses import dataclass, field

@dataclass
class Config:
    SYSTEM_PROMPT: str = "V1"
    FILTER_TAGS: dict = field(default_factory=lambda: {'advanced': [], 'basic': ["Scene"]})
    NUM_FILTER: int = 10
    FILTER_BY: str = 'Num_Tokens'
    PROVIDE_PROMPTS: bool = True
    GENAI_CSV_PATH: str = "eval/genai_prompts/genai_dataset.csv"
    INTERPOLATION_TECHNIQUE: str = "nlerp"
    NUM_INFERENCE_STEPS: int = 50
    TEMPERATURE: float = 1.0
    MODEL_ID: str = "stabilityai/stable-diffusion-2-1-base"
    STEP_SIZE: int = 1
    SEED: int = 42

    def create_exp_name(self) -> str:
        model_id = self.MODEL_ID.replace("/", "-")
        if self.PROVIDE_PROMPTS:
            # Create the filter tags part
            advanced_tags = "_".join(self.FILTER_TAGS.get('advanced', [])) or "none"
            basic_tags = "_".join(self.FILTER_TAGS.get('basic', [])) or "none"
            
            filter_tags = f"advanced_{advanced_tags}_basic_{basic_tags}"
            filter_tags = filter_tags.replace(" ", "_")

            model_id = model_id.replace("/", "-")
            return f"exp_dump/{self.INTERPOLATION_TECHNIQUE}_model_{model_id}/num_inference_{self.NUM_INFERENCE_STEPS}_TEMP_{self.TEMPERATURE}_STEP_SIZE_{self.STEP_SIZE}_SEED_{self.SEED}/prompt_exp_{self.SYSTEM_PROMPT}_filter_{filter_tags}_num_prompts_{self.NUM_FILTER}_filter_{self.FILTER_BY}"
        else:
            return f"exp_dump/{self.INTERPOLATION_TECHNIQUE}_model_{model_id}/num_inference_{self.NUM_INFERENCE_STEPS}_TEMP_{self.TEMPERATURE}_STEP_SIZE_{self.STEP_SIZE}_SEED_{self.SEED}/prompt_exp_{self.SYSTEM_PROMPT}_gen_prompts_from_scratch"
