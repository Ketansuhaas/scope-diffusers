import numpy as np
import torch
import math

# === Base Interpolator Classes ===

class BasePromptInterpolator:
    def __init__(self, embeddings, device="cuda"):
        """
        Base prompt interpolator.

        Args:
            embeddings (torch.Tensor): (num_stages, batch, seq_len, embed_dim)
            device (str): Device to move the interpolated embeddings to.
        """
        self.embeddings = embeddings
        self.device = device
        self.config = {
            "device": device,
            "num_stages": embeddings.shape[0],
            "interpolator": self.__class__.__name__
        }

    def interpolate(self, time_i):
        raise NotImplementedError("Subclasses must implement this method.")

    def __call__(self, time_i):
        return self.interpolate(time_i).to(self.device)


# === Original NLerpInterpolator (Euclidean + time-dependent tau) ===

class NLerpInterpolatorOG(BasePromptInterpolator):
    def __init__(self, embeddings, interpolation_period=1, device="cuda", **kwargs):
        super().__init__(embeddings, device)
        self.period = interpolation_period
        self.stdev = kwargs.get("std_dev", 5)  # Pull out only what's relevant

        self.config.update({
            "std_dev": self.stdev,
            "interpolation_period": self.period
        })
        self.initialize_spacing()
    
    def initialize_spacing(self):
        q = self.embeddings.shape[0] - 1
        distances = np.zeros((77, q))
        for idx in range(self.embeddings.shape[0] - 1):
            e1 = self.embeddings[idx].detach().cpu().numpy()
            e2 = self.embeddings[idx + 1].detach().cpu().numpy()
            for i in range(e1.shape[1]):
                euclidean_distance = np.linalg.norm(e1[-1, i, :] - e2[-1, i, :])
                distances[i][idx] = euclidean_distance

        times = np.arange(self.embeddings.shape[0], dtype=float)
        self.row_sums = distances.sum(axis=1, keepdims=True)
        distances_normalized = distances / (self.row_sums + 1e-5)
        distances = distances_normalized * self.period
        distances = np.cumsum(distances, axis=1)
        zero_column = np.zeros((distances.shape[0], 1))
        self.times = np.hstack((zero_column, distances))

    def interpolate(self, time_i):
        if time_i >= self.period:
            return self.embeddings[-1]
        tau = self.stdev * (1 - (time_i / self.period)) + 0.1
        interpolated_embedding = self.embeddings[0].clone().detach()
        for i in range(self.embeddings[0].shape[1]):
            if self.row_sums[i] == 0:
                continue
            weights = torch.tensor(
                [np.exp(-((t - time_i) / tau) ** 2 / 2) for t in self.times[i]],
                device=self.device
            )
            weights /= weights.sum()
            weights = weights.unsqueeze(1)

            token_feature_values = torch.stack([
                self.embeddings[k, -1, i, :] for k in range(self.embeddings.shape[0])
            ])
            interpolated_value = torch.sum(token_feature_values * weights, dim=0)
            interpolated_embedding[-1, i, :] = interpolated_value

            original_magnitude = torch.norm(self.embeddings[0][-1, i, :])
            current_magnitude = torch.norm(interpolated_embedding[-1, i, :])
            if current_magnitude > 0:
                interpolated_embedding[-1, i, :] *= (original_magnitude / current_magnitude)

        return interpolated_embedding.to(self.device)

    @staticmethod
    def hparam_grid():
        return {
            "interpolation_period": [4, 12, 20, 28],
            "std_dev": [3, 5],
        }

    @classmethod
    def from_config(cls, embeddings, interpolation_period, device="cuda", **kwargs):
        return cls(
            embeddings=embeddings,
            interpolation_period=interpolation_period,
            stdev=kwargs.get("std_dev", 3),
            device=device
        )

# === Stagewise Prompt Switcher (No interpolation, hard swap at fixed intervals) ===

class StagewisePromptSwitcher(BasePromptInterpolator):
    def __init__(self, embeddings, interpolation_period=1, device="cuda", **kwargs):
        """
        A non-interpolating baseline: just swap in the corresponding prompt embedding
        at evenly spaced steps during the interpolation period.

        Args:
            embeddings (torch.Tensor): Shape (num_stages, batch, seq_len, embed_dim)
            interpolation_period (int): Number of total diffusion steps
        """
        super().__init__(embeddings, device)
        self.period = interpolation_period
        self.num_stages = embeddings.shape[0]

        # Compute step indices at which to switch stages
        self.stage_boundaries = np.linspace(0, self.period, self.num_stages + 1, dtype=int)
        self.config.update({"interpolation_period": self.period})

    def interpolate(self, time_i):
        # Find which stage to use based on time_i
        for i in range(self.num_stages):
            if self.stage_boundaries[i] <= time_i < self.stage_boundaries[i + 1]:
                return self.embeddings[i]
        return self.embeddings[-1]  # If time_i >= period

    @staticmethod
    def hparam_grid():
        return {
            "interpolation_period": [4, 12, 20, 28],
        }

    @classmethod
    def from_config(cls, embeddings, interpolation_period, device="cuda", **kwargs):
        return cls(
            embeddings=embeddings,
            interpolation_period=interpolation_period,
            device=device
        )

# === Stagewise Prompt Switcher Respaced (No interpolation, hard swap at respaced intervals) ===

# UNDER CONSTRUCTION
class StagewisePromptSwitcherRespaced(BasePromptInterpolator):
    def __init__(self, embeddings, interpolation_period=1, device="cuda", **kwargs):
        super().__init__(embeddings, device)
        self.period = interpolation_period
        self.stdev = kwargs.get("std_dev", 5)  # Pull out only what's relevant

        self.config.update({
            "std_dev": self.stdev,
            "interpolation_period": self.period
        })

    def interpolate(self, time_i):
        if time_i >= self.period:
            return self.embeddings[-1]

        q = self.embeddings.shape[0] - 1
        distances = np.zeros((77, q))
        for idx in range(self.embeddings.shape[0] - 1):
            e1 = self.embeddings[idx].detach().cpu().numpy()
            e2 = self.embeddings[idx + 1].detach().cpu().numpy()
            for i in range(e1.shape[1]):
                euclidean_distance = np.linalg.norm(e1[-1, i, :] - e2[-1, i, :])
                distances[i][idx] = euclidean_distance

        tau = self.stdev * (1 - (time_i / self.period)) + 0.1

        times = np.arange(self.embeddings.shape[0], dtype=float)
        row_sums = distances.sum(axis=1, keepdims=True)
        distances_normalized = distances / (row_sums + 1e-5)
        distances = distances_normalized * self.period
        distances = np.cumsum(distances, axis=1)
        zero_column = np.zeros((distances.shape[0], 1))
        times = np.hstack((zero_column, distances))

        interpolated_embedding = self.embeddings[0].clone().detach()
        for i in range(self.embeddings[0].shape[1]):
            if row_sums[i] == 0:
                continue
            weights = torch.tensor(
                [np.exp(-((t - time_i) / tau) ** 2 / 2) for t in times[i]],
                device=self.device
            )
            weights /= weights.sum()
            weights = weights.unsqueeze(1)

            token_feature_values = torch.stack([
                self.embeddings[k, -1, i, :] for k in range(self.embeddings.shape[0])
            ])
            interpolated_value = torch.sum(token_feature_values * weights, dim=0)
            interpolated_embedding[-1, i, :] = interpolated_value

            original_magnitude = torch.norm(self.embeddings[0][-1, i, :])
            current_magnitude = torch.norm(interpolated_embedding[-1, i, :])
            if current_magnitude > 0:
                interpolated_embedding[-1, i, :] *= (original_magnitude / current_magnitude)

        return interpolated_embedding.to(self.device)

    @staticmethod
    def hparam_grid():
        return {
            "interpolation_period": [4, 12, 20, 28],
            "std_dev": [3, 5],
        }

    @classmethod
    def from_config(cls, embeddings, interpolation_period, device="cuda", **kwargs):
        return cls(
            embeddings=embeddings,
            interpolation_period=interpolation_period,
            stdev=kwargs.get("std_dev", 3),
            device=device
        )





# === Interpolator Factory ===

def get_interpolator(method="nlerp"):
    if method == "nlerp_og":
        return NLerpInterpolatorOG
    elif method == "stagewise_switcher":
        """
        StagewisePromptSwitcher does not interpolate but switches embeddings at fixed intervals.
        """
        return StagewisePromptSwitcher
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
