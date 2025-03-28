import numpy as np
import torch
import math

# === Base Interpolator Classes ===

class BasePromptInterpolator:
    def __init__(self, embeddings, times, device="cuda"):
        """
        Base prompt interpolator.

        Args:
            embeddings (torch.Tensor): (num_stages, batch, seq_len, embed_dim)
            times (list): A list of time indices corresponding to each stage, e.g. [0, 25, 50].
            device (str): Device to move the interpolated embeddings to.
        """
        self.embeddings = embeddings
        self.times = times
        self.device = device

    def interpolate(self, time_i):
        raise NotImplementedError("Subclasses must implement this method.")

    def __call__(self, time_i):
        return self.interpolate(time_i).to(self.device)


# === Original NLerpInterpolator (Euclidean + time-dependent tau) ===

class NLerpInterpolatorOG(BasePromptInterpolator):
    def __init__(self, embeddings, times, stdev=5, device="cuda"):
        """
        Args:
            embeddings (torch.Tensor): (num_stages, batch, seq_len, embed_dim)
            times (list): e.g. [0, 25, 50]
            stdev (float): Used to compute time-dependent tau
            device (str): Device
        """
        super().__init__(embeddings, times, device)
        self.stdev = stdev

    def interpolate(self, time_i):
        # If time_i exceeds the last stage, return the final embeddings
        if time_i >= self.times[-1]:
            return self.embeddings[-1]

        # 1) Compute pairwise Euclidean distances between adjacent stages (per token)
        q = len(self.times) - 1  # number of gaps
        distances = np.zeros((77, q))  # assume 77 tokens
        for idx in range(self.embeddings.shape[0] - 1):
            e1 = self.embeddings[idx].detach().cpu().numpy()
            e2 = self.embeddings[idx + 1].detach().cpu().numpy()
            for i in range(e1.shape[1]):  # for each token
                euclidean_distance = np.linalg.norm(e1[-1, i, :] - e2[-1, i, :])
                distances[i][idx] = euclidean_distance

        # 2) Compute time-dependent tau
        tau = self.stdev * (1 - (time_i / self.times[-1])) + 0.1

        # 3) ======== The requested snippet (unchanged) ========
        # Put self.times into a local var "times" so we can apply the snippet exactly
        times = np.array(self.times, dtype=float)

        # Normalize each row by dividing by its sum
        row_sums = distances.sum(axis=1, keepdims=True)
        distances_normalized = distances / (row_sums + 1e-5)
        # Multiply all elements by times[-1]
        distances = distances_normalized * times[-1]
        distances = np.cumsum(distances, axis=1)
        zero_column = np.zeros((distances.shape[0], 1))
        # Concatenate the zero column with the cumulative sum
        times = np.hstack((zero_column, distances))
        # ============ABOVE IS OPTIONAL============================================ (times[i]->times)

        # Initialize the interpolated embedding with the same shape as input embeddings
        interpolated_embedding = self.embeddings[0].clone().detach()
        # Perform weighted sum token-wise across prompts
        for i in range(self.embeddings[0].shape[1]):  # 77 tokens
            if row_sums[i] == 0:
                continue
            # Calculate weights using the exponential function for all points
            weights = torch.tensor([np.exp(-((t - time_i) / tau) ** 2 / 2) for t in times[i]], device=self.device)
            # Normalize weights
            weights /= weights.sum()
            weights = weights.unsqueeze(1)

            token_feature_values = torch.stack([self.embeddings[k, -1, i, :] for k in range(self.embeddings.shape[0])])
            interpolated_value = torch.sum(token_feature_values * weights, dim=0)
            interpolated_embedding[-1, i, :] = interpolated_value

            original_magnitude = torch.norm(self.embeddings[0][-1, i, :])
            current_magnitude = torch.norm(interpolated_embedding[-1, i, :])
            if current_magnitude > 0:
                interpolated_embedding[-1, i, :] *= (original_magnitude / current_magnitude)

        return interpolated_embedding.to(self.device)



def get_interpolator(embeddings, times, method="nlerp", std_dev=3, device="cuda"):
    """
    Factory function to create an interpolator instance based on the chosen method.
    """
    if method == "nlerp_og":
        return NLerpInterpolatorOG(embeddings, times, std_dev, device)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")