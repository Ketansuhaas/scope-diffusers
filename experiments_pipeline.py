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
        num_seeds = len(seed_list)
        num_steps = len(self.config["step_sizes"]) + 1  # +1 for the normal image

        # Create a figure that holds all results for all seeds with adjusted figure size
        plt.figure(figsize=(5 * num_steps, 5 * num_seeds), dpi=300)  # Adjust size and DPI

        for seed_idx, seed in enumerate(seed_list):
            logger.info(f"Running experiment with seed: {seed}")
            # Load the SCoPE Diffusion model
            pipe = sdp_scope.from_pretrained(
                self.config["MODEL_ID"], torch_dtype=torch.float16, low_cpu_mem_usage=True,
                cache_dir = '/projectnb/vkolagrp/ketanss/scope-diffusers/sdpcache'
            )
            pipe = pipe.to(self.config["DEVICE"])
            image_scope_list = []

            for step_size in self.config["step_sizes"]:
                logger.info(f"Running with step size: {step_size}")
                prompt_schedule_list = self.config['prompt_schedule']

                prompt_schedule = []
                 
                for stage_id, p in enumerate(prompt_schedule_list):       # change step size in the prompt schedule
                    prompt_schedule.append((stage_id*step_size,p))
                
                logger.info(f"Running SCoPE Diffusion on the prompt schedule: {prompt_schedule}")
                torch.manual_seed(seed)
                image = pipe(
                    prompt_schedule,
                    num_inference_steps=self.config["num_inference_steps"],
                    callback=None,
                    callback_steps=1,
                    temperature=self.config["temperature"],
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
                prompt_schedule[-1][1],  # Only the final prompt for normal Stable Diffusion
                num_inference_steps=self.config["num_inference_steps"],
                cache_dir = '/projectnb/vkolagrp/ketanss/scope-diffusers/sdpcache',
                callback=None,
                callback_steps=1,
                temperature=self.config["temperature"],
            ).images[0]
            image_normal = np.array(image)

            # Plot results for this seed
            for idx, image_scope in enumerate(image_scope_list):
                ax = plt.subplot(num_seeds, num_steps, seed_idx * num_steps + idx + 1)
                ax.axis("off")
                ax.imshow(image_scope)
                if seed_idx == 0:  # Only display titles for the first row
                    ax.set_title(f"Step size = {self.config['step_sizes'][idx]}", fontsize=12)

            # Plot normal image for this seed
            ax = plt.subplot(num_seeds, num_steps, seed_idx * num_steps + num_steps)
            ax.axis("off")
            ax.imshow(image_normal)
            if seed_idx == 0:  # Only display title for the normal image in the first row
                ax.set_title("Normal image", fontsize=12)

        # Set overall title for the figure and adjust layout for better spacing
        plt.suptitle(f"Prompt schedule: {prompt_schedule}", fontsize=16, wrap=True)
        plt.subplots_adjust(top=0.6,wspace=0.1, hspace=0.1)  # Adjust the space between subplots

        # Save the figure with proper DPI and output path
        output_path = os.path.join(self.exp_dir, f"output_all_seeds.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved output for all seeds to {output_path}")


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
                self.config["MODEL_ID"], torch_dtype=torch.float16, low_cpu_mem_usage=True,
                cache_dir = '/projectnb/vkolagrp/ketanss/scope-diffusers/sdpcache'
            )
            pipe = pipe.to(self.config["DEVICE"])
            image_scope_list = []

            for step_size in self.config["step_sizes"]:
                logger.info(f"Running with step size: {step_size}")
                prompt_schedule_list = self.config['prompt_schedule']

                prompt_schedule = []
                 
                for stage_id, p in enumerate(prompt_schedule_list):       # change step size in the prompt schedule
                    prompt_schedule.append((stage_id*step_size,p))
                
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
                self.config["MODEL_ID"], torch_dtype=torch.float16, low_cpu_mem_usage=True,
                cache_dir = '/projectnb/vkolagrp/ketanss/scope-diffusers/sdpcache'
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


class SCoPE_Exp_Temperature(SCoPE_Exp_Base):
    def __init__(self, config, exp_name, exp_id):
        super().__init__(config, exp_name, exp_id)

    def run(self):
        # Define the list of temperatures from config
        temperature_list = self.config['temperature_list']
        num_temps = len(temperature_list)
        num_steps = len(self.config["step_sizes"]) + 1  # +1 for the normal image

        # Create a figure that holds all results for all temperatures with adjusted figure size
        plt.figure(figsize=(5 * num_steps, 5 * num_temps), dpi=150)  # Adjust size and DPI

        for temp_idx, temperature in enumerate(temperature_list):
            logger.info(f"Running experiment with temperature: {temperature}")

            # Load the SCoPE Diffusion model
            pipe = sdp_scope.from_pretrained(
                self.config["MODEL_ID"], torch_dtype=torch.float16, low_cpu_mem_usage=True,
                cache_dir = '/projectnb/vkolagrp/ketanss/scope-diffusers/sdpcache'
            )
            pipe = pipe.to(self.config["DEVICE"])
            image_scope_list = []

            for step_size in self.config["step_sizes"]:
                logger.info(f"Running with step size: {step_size}")
                prompt_schedule_list = self.config['prompt_schedule']

                prompt_schedule = []
                 
                for stage_id, p in enumerate(prompt_schedule_list):       # change step size in the prompt schedule
                    prompt_schedule.append((stage_id*step_size,p))
                
                logger.info(f"Running SCoPE Diffusion on the prompt schedule: {prompt_schedule}")
                torch.manual_seed(42)
                image = pipe(
                    prompt_schedule,
                    temperature=temperature,
                    num_inference_steps=self.config["num_inference_steps"],
                    callback=None,
                    callback_steps=1,
                ).images[0]
                image_scope_list.append(np.array(image))

            logger.info("Running normal Stable Diffusion")
            # Load the normal Stable Diffusion model
            pipe = sdp.from_pretrained(
                self.config["MODEL_ID"], torch_dtype=torch.float16, low_cpu_mem_usage=True,
                cache_dir = '/projectnb/vkolagrp/ketanss/scope-diffusers/sdpcache'
            )
            pipe = pipe.to(self.config["DEVICE"])
            torch.manual_seed(42)
            image = pipe(
                prompt_schedule[-1][1],  # Only the final prompt for normal Stable Diffusion
                temperature=temperature,  # Add temperature variation
                num_inference_steps=self.config["num_inference_steps"],
                callback=None,
                callback_steps=1,
            ).images[0]
            image_normal = np.array(image)

            # Plot results for this temperature
            for idx, image_scope in enumerate(image_scope_list):
                ax = plt.subplot(num_temps, num_steps, temp_idx * num_steps + idx + 1)
                ax.axis("off")
                ax.imshow(image_scope)
                if temp_idx == 0:  # Only display titles for the first row
                    ax.set_title(f"Step size = {self.config['step_sizes'][idx]}", fontsize=12)

            # Plot normal image for this temperature
            ax = plt.subplot(num_temps, num_steps, temp_idx * num_steps + num_steps)
            ax.axis("off")
            ax.imshow(image_normal)
            if temp_idx == 0:  # Only display title for the normal image in the first row
                ax.set_title("Normal image", fontsize=12)

        # Set overall title for the figure and adjust layout for better spacing
        plt.suptitle(f"Prompt: {prompt_schedule[-1][1]}", fontsize=16, y=0.95, wrap=True)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust the space between subplots

        # Save the figure with proper DPI and output path
        output_path = os.path.join(self.exp_dir, f"output_all_temperatures.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved output for all temperatures to {output_path}")


class SCoPE_Exp_overall(SCoPE_Exp_Base):
    def __init__(self, config, exp_name, exp_id):
        super().__init__(config, exp_name, exp_id)

    def run(self):
        # Manually define the list of seeds from config
        seed = self.config['seed']
        num_steps = len(self.config["step_sizes"]) + 1  # +1 for the normal image

        # Create a figure that holds all results for all seeds with adjusted figure size
        plt.figure(figsize=(5 * num_steps, 5 * len(self.config["temperatures"])), dpi=500)  # Adjust size and DPI

        for temp_idx, temp in enumerate(self.config["temperatures"]):
            logger.info(f"Running experiment with temperature: {temp}")
            # Load the SCoPE Diffusion model
            pipe = sdp_scope.from_pretrained(
                self.config["MODEL_ID"], torch_dtype=torch.float16, low_cpu_mem_usage=True,
                cache_dir = '/projectnb/vkolagrp/ketanss/scope-diffusers/sdpcache'
            )
            pipe = pipe.to(self.config["DEVICE"])
            image_scope_list = []

            for step_size in self.config["step_sizes"]:
                logger.info(f"Running with step size: {step_size}")
                prompt_schedule_list = self.config['prompt_schedule']

                prompt_schedule = []
                 
                for stage_id, p in enumerate(prompt_schedule_list):       # change step size in the prompt schedule
                    prompt_schedule.append((stage_id*step_size,p))
                
                logger.info(f"Running SCoPE Diffusion on the prompt schedule: {prompt_schedule}")
                torch.manual_seed(seed)
                image = pipe(
                    interpolation_technique = "nlerp",
                    prompt_schedule = prompt_schedule,
                    num_inference_steps=self.config["num_inference_steps"],
                    callback=None,
                    callback_steps=1,
                    temperature=temp,
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
                prompt_schedule[-1][1],  # Only the final prompt for normal Stable Diffusion
                num_inference_steps=self.config["num_inference_steps"],
                cache_dir = '/projectnb/vkolagrp/ketanss/scope-diffusers/sdpcache',
                callback=None,
                callback_steps=1,
                temperature=temp,
            ).images[0]
            image_normal = np.array(image)

            # Plot results for this seed
            for idx, image_scope in enumerate(image_scope_list):
                ax = plt.subplot(len(self.config["temperatures"]), num_steps, temp_idx * num_steps + idx + 1)
                ax.axis("off")
                ax.imshow(image_scope)
                if temp_idx == 0:  # Only display titles for the first row
                    ax.set_title(f"Step size = {self.config['step_sizes'][idx]}", fontsize=12)

            # Plot normal image for this seed
            ax = plt.subplot(len(self.config["temperatures"]), num_steps, temp_idx * num_steps + num_steps)
            ax.axis("off")
            ax.imshow(image_normal)
            if temp_idx == 0:  # Only display title for the normal image in the first row
                ax.set_title("Normal image", fontsize=12)

        # Set overall title for the figure and adjust layout for better spacing
        plt.suptitle(f"Prompt schedule: {prompt_schedule}", fontsize=16, wrap=True)
        plt.subplots_adjust()  # Adjust the space between subplots

        # Save the figure with proper DPI and output path
        output_path = os.path.join(self.exp_dir, f"output_all_seeds.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved output for all seeds to {output_path}")

