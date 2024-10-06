import torch
from scope_diffuser import SCoPEDiffusionPipeline as sdp_scope
from diffusers import StableDiffusionPipeline as sdp
import matplotlib.pyplot as plt
import numpy as np
from ezcolorlog import root_logger as logger
import os
import argparse

class SCoPE_Exp_Base:
    def __init__(self, config, exp_name, exp_id):
        self.exp_name = exp_name
        self.exp_id = exp_id
        self.exp_dir = os.path.join("experiments", exp_name, exp_id)
        os.makedirs(self.exp_dir, exist_ok=True)
        self.config = config

    def run(self):
        raise NotImplementedError("Subclasses should implement this method")

class SCoPE_Exp_Seed(SCoPE_Exp_Base):
    def __init__(self, config, exp_name, exp_id):
        super().__init__(config, exp_name, exp_id)

    def run(self):
        # Manually define the list of seeds from config
        seed_list = self.config['seed_list']
        for seed in seed_list:
            logger.info(f"Running experiment with seed: {seed}")
            torch.manual_seed(seed)
            # Load the SCoPE Diffusion model
            pipe = sdp_scope.from_pretrained(
                self.config["MODEL_ID"], torch_dtype=torch.float16, low_cpu_mem_usage=True
            )
            pipe = pipe.to(self.config["DEVICE"])
            image_scope_list = []

            for step_size in self.config["step_sizes"]:
                logger.info(f"Running with step size: {step_size}")
                prompt_schedule = self.config['prompt_schedule']
                logger.info(f"Running SCoPE Diffusion on the prompt schedule: {prompt_schedule}")
                image = pipe(
                    prompt_schedule,
                    num_inference_steps=self.config["num_inference_steps"],
                    callback=None,
                    callback_steps=1,
                ).images[0]
                image_scope_list.append(np.array(image))

            logger.info("Running normal Stable Diffusion")
            # Load the normal Stable Diffusion model
            pipe = sdp.from_pretrained(
                self.config["MODEL_ID"], torch_dtype=torch.float16, low_cpu_mem_usage=True
            )
            pipe = pipe.to(self.config["DEVICE"])
            torch.manual_seed(seed)
            image = pipe(
                prompt_schedule[-1][1],
                num_inference_steps=self.config["num_inference_steps"],
                callback=None,
                callback_steps=1,
            ).images[0]
            image_normal = np.array(image)

            # Plotting
            plt.figure(figsize=(20, 8))
            for idx, image_scope in enumerate(image_scope_list):
                plt.subplot(1, len(self.config["step_sizes"]) + 1, idx + 1)
                plt.axis("off")
                plt.imshow(image_scope)
                plt.title(f"Step size = {self.config['step_sizes'][idx]}")

            plt.subplot(1, len(self.config["step_sizes"]) + 1, len(self.config["step_sizes"]) + 1)
            plt.axis("off")
            plt.imshow(image_normal)
            plt.title("Normal image")

            plt.tight_layout()
            plt.figtext(0.5, 0.1, prompt_schedule[-1][1], ha="center", fontsize=15, wrap=True)
            output_path = os.path.join(self.exp_dir, f"output_seed_{seed}.png")
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Saved output to {output_path}")

class SCoPE_Exp_Model(SCoPE_Exp_Base):
    def __init__(self, config, exp_name, exp_id):
        super().__init__(config, exp_name, exp_id)

    def run(self):
        # Manually define the list of model IDs from config
        model_ids = self.config['model_ids']
        for model_id in model_ids:
            logger.info(f"Running experiment with model_id: {model_id}")
            self.config["MODEL_ID"] = model_id
            # Load the SCoPE Diffusion model
            pipe = sdp_scope.from_pretrained(
                self.config["MODEL_ID"], torch_dtype=torch.float16, low_cpu_mem_usage=True
            )
            pipe = pipe.to(self.config["DEVICE"])
            image_scope_list = []

            for step_size in self.config["step_sizes"]:
                logger.info(f"Running with step size: {step_size}")
                prompt_schedule = self.config['prompt_schedule']
                logger.info(f"Running SCoPE Diffusion on the prompt schedule: {prompt_schedule}")
                torch.manual_seed(self.config["seed"])
                image = pipe(
                    prompt_schedule,
                    num_inference_steps=self.config["num_inference_steps"],
                    callback=None,
                    callback_steps=1,
                ).images[0]
                image_scope_list.append(np.array(image))

            logger.info("Running normal Stable Diffusion")
            # Load the normal Stable Diffusion model
            pipe = sdp.from_pretrained(
                self.config["MODEL_ID"], torch_dtype=torch.float16, low_cpu_mem_usage=True
            )
            pipe = pipe.to(self.config["DEVICE"])
            torch.manual_seed(self.config["seed"])
            image = pipe(
                prompt_schedule[-1][1],
                num_inference_steps=self.config["num_inference_steps"],
                callback=None,
                callback_steps=1,
            ).images[0]
            image_normal = np.array(image)

            # Plotting
            plt.figure(figsize=(20, 8))
            for idx, image_scope in enumerate(image_scope_list):
                plt.subplot(1, len(self.config["step_sizes"]) + 1, idx + 1)
                plt.axis("off")
                plt.imshow(image_scope)
                plt.title(f"Step size = {self.config['step_sizes'][idx]}")

            plt.subplot(1, len(self.config["step_sizes"]) + 1, len(self.config["step_sizes"]) + 1)
            plt.axis("off")
            plt.imshow(image_normal)
            plt.title("Normal image")

            plt.tight_layout()
            plt.figtext(0.5, 0.1, prompt_schedule[-1][1], ha="center", fontsize=15, wrap=True)
            model_name = model_id.replace('/', '_')
            output_path = os.path.join(self.exp_dir, f"output_model_{model_name}.png")
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Saved output to {output_path}")