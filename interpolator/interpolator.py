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
                # print(e1.shape, e2.shape)
                euclidean_distance = np.linalg.norm(e1[-1, i, :] - e2[-1, i, :])
                distances[i][idx] = euclidean_distance

        times = np.arange(self.embeddings.shape[0], dtype=float)
        self.row_sums = distances.sum(axis=1, keepdims=True)

        # print(distances.shape, self.row_sums.shape)

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


# === NLerpInterpolator (Euclidean + No tau decay) ===
class NLerpInterpolatorOGNoStdDecay(BasePromptInterpolator):
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
                # print(e1.shape, e2.shape)
                euclidean_distance = np.linalg.norm(e1[-1, i, :] - e2[-1, i, :])
                distances[i][idx] = euclidean_distance

        times = np.arange(self.embeddings.shape[0], dtype=float)
        self.row_sums = distances.sum(axis=1, keepdims=True)

        # print(distances.shape, self.row_sums.shape)

        distances_normalized = distances / (self.row_sums + 1e-5)
        distances = distances_normalized * self.period
        distances = np.cumsum(distances, axis=1)
        zero_column = np.zeros((distances.shape[0], 1))
        self.times = np.hstack((zero_column, distances))

    def interpolate(self, time_i):
        if time_i >= self.period:
            return self.embeddings[-1]
        tau = self.stdev
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


# === NLerpInterpolator (Cosine + No tau decay) ===
class NLerpInterpolatorCosineRespacingNoDecay(BasePromptInterpolator):
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
                # print(e1.shape, e2.shape)
                token_emb1 = e1[-1, i, :]
                token_emb2 = e2[-1, i, :]
                cosine_distance = 1 - np.dot(token_emb1, token_emb2) / (np.linalg.norm(token_emb1) * np.linalg.norm(token_emb2) + 1e-8)
                distances[i][idx] = cosine_distance

        times = np.arange(self.embeddings.shape[0], dtype=float)
        self.row_sums = distances.sum(axis=1, keepdims=True)

        # print(distances.shape, self.row_sums.shape)

        distances_normalized = distances / (self.row_sums + 1e-5)
        distances = distances_normalized * self.period
        distances = np.cumsum(distances, axis=1)
        zero_column = np.zeros((distances.shape[0], 1))
        self.times = np.hstack((zero_column, distances))

    def interpolate(self, time_i):
        if time_i >= self.period:
            return self.embeddings[-1]
        tau = self.stdev
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



# === NLerpInterpolator for flux===
class NLerpInterpolatorOG_Flux(BasePromptInterpolator):
    """
    An NLerp (normalized linear interpolation) interpolator for Flux embeddings.
    
    This version is adapted for Flux where each timestep's embedding has shape 
    [1, 512, D] (i.e. no separate negative/positive embeddings). Interpolation is 
    performed token-wise over the 512 tokens.
    """
    def __init__(self, embeddings, interpolation_period=1, device="cuda", **kwargs):
        super().__init__(embeddings, device)
        self.period = interpolation_period
        self.stdev = kwargs.get("std_dev", 5)
        
        self.config.update({
            "std_dev": self.stdev,
            "interpolation_period": self.period
        })
        self.initialize_spacing()
    
    def initialize_spacing(self):
        """
        Computes a per-token cumulative distance (or 'time') schedule along the diffusion steps.
        
        Assumes self.embeddings has shape [T, 1, 512, D] where T is the number of timesteps.
        """
        q = self.embeddings.shape[0] - 1  # number of segments (T-1)
        distances = np.zeros((512, q))     # for each of 512 tokens

        # For each consecutive pair of timesteps, compute the Euclidean distance token-wise.
        for idx in range(q):
            # Get embeddings for timestep idx and idx+1; shape: [512, D]
            e1 = self.embeddings[idx].detach().cpu().numpy()[0]
            e2 = self.embeddings[idx + 1].detach().cpu().numpy()[0]
            for i in range(e1.shape[0]):  # for each token (512)
                euclidean_distance = np.linalg.norm(e1[i] - e2[i])
                distances[i, idx] = euclidean_distance

        # Sum distances per token (to normalize later)
        self.row_sums = distances.sum(axis=1, keepdims=True)
        # Normalize distances per token; add a small epsilon to avoid division by zero.
        distances_normalized = distances / (self.row_sums + 1e-5)
        # Scale by the interpolation period
        distances_scaled = distances_normalized * self.period
        # Compute cumulative distances per token (the 'time' at each step)
        distances_cumsum = np.cumsum(distances_scaled, axis=1)
        zero_column = np.zeros((distances_cumsum.shape[0], 1))
        # Concatenate the zero at the beginning so that each token has T values (for T timesteps)
        self.times = np.hstack((zero_column, distances_cumsum))  # shape [512, T]

    def interpolate(self, time_i):
        """
        Interpolates the embeddings at a given time step (time_i).
        
        If time_i >= period, returns the final embedding.
        Otherwise, computes token-wise weights based on a Gaussian function over the 
        precomputed times and returns a weighted sum of embeddings.
        """
        if time_i >= self.period:
            return self.embeddings[-1]
        
        tau = self.stdev * (1 - (time_i / self.period)) + 0.1
        # Start from the initial embedding; shape: [1, 512, D]
        interpolated_embedding = self.embeddings[0].clone().detach()

        # For each of the 512 tokens
        for i in range(interpolated_embedding.shape[1]):
            if self.row_sums[i] == 0:
                continue

            # Compute weights using a Gaussian function on the precomputed times
            weights = torch.tensor(
                [np.exp(-((t - time_i) / tau) ** 2 / 2) for t in self.times[i]],
                device=self.device
            )
            weights /= weights.sum()
            weights = weights.unsqueeze(1)  # shape: [T, 1]

            # Collect the embeddings from each timestep for token i;
            # each has shape [D]. The result has shape [T, D].
            token_feature_values = torch.stack([
                self.embeddings[k, 0, i, :] for k in range(self.embeddings.shape[0])
            ])
            # Compute weighted sum over timesteps; result shape: [D]
            interpolated_value = torch.sum(token_feature_values * weights, dim=0)

            # Normalize magnitude to match the original token's scale
            original_magnitude = torch.norm(self.embeddings[0, 0, i, :])
            current_magnitude = torch.norm(interpolated_value)
            if current_magnitude > 0:
                interpolated_value *= (original_magnitude / current_magnitude)

            # Write back the interpolated value for token i
            interpolated_embedding[0, i, :] = interpolated_value

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

class StagewisePromptSwitcherRespaced(BasePromptInterpolator):
    def __init__(self, embeddings, interpolation_period=1, device="cuda", **kwargs):
        super().__init__(embeddings, device)
        self.period = interpolation_period
        self.num_stages = embeddings.shape[0]

        # Compute step indices at which to switch stages
        self.stage_boundaries = np.linspace(0, self.period, self.num_stages + 1, dtype=int)
        self.config.update({"interpolation_period": self.period})
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

        final_embedding = self.embeddings[0].clone().detach()
        for i in range(self.embeddings[0].shape[1]):
            if self.row_sums[i] == 0:
                continue
            
            weights = []
            for t_id, t in enumerate(self.times[i][:-1]):
                if time_i>=self.times[i][t_id] and time_i <self.times[i][t_id+1]:
                    weights.append(1)
                else:
                    weights.append(0)
            weights.append(0)
     
            weights = torch.tensor(
                weights,
                device=self.device
            )
            weights = weights.unsqueeze(1)

            token_feature_values = torch.stack([
                self.embeddings[k, -1, i, :] for k in range(self.embeddings.shape[0])
            ])
            interpolated_value = torch.sum(token_feature_values * weights, dim=0)
            final_embedding[-1, i, :] = interpolated_value

        return final_embedding.to(self.device)

    @staticmethod
    def hparam_grid():
        return {
            "interpolation_period": [4, 12, 20, 28]
        }

    @classmethod
    def from_config(cls, embeddings, interpolation_period, device="cuda", **kwargs):
        return cls(
            embeddings=embeddings,
            interpolation_period=interpolation_period,
            device=device
        )


    def singular_reweight(self, prompt_embeds1, prompt_embeds2, alpha):
        # return prompt_embeds2
        from sklearn.decomposition import PCA
        # Convert to numpy for PCA (ensure tensors are on CPU)
        matrix1 = prompt_embeds1[0].cpu().numpy()
        matrix2 = prompt_embeds2[0].cpu().numpy()

        # Perform PCA on matrix1
        pca1 = PCA(n_components=77)
        pca1.fit(matrix1)

        # Transform matrix2 into the subspace of matrix1
        matrix2_transformed = pca1.transform(matrix2)

        # Reconstruct matrix2 in the subspace of matrix1
        matrix2_reconstructed = pca1.inverse_transform(matrix2_transformed)

        # Convert back to PyTorch tensor with correct dtype and device
        tensor2_reconstructed = torch.from_numpy(matrix2_reconstructed).to(prompt_embeds1.dtype)
        
        # Ensure batch dimension is retained
        tensor2_reconstructed = tensor2_reconstructed.unsqueeze(0).to(prompt_embeds2.device)

        return tensor2_reconstructed

    # def singular_reweight(self, p1, p2, alpha):
    #     # return p2
    #     """
    #     Projects B onto the subspace spanned by the columns of A.
    #     A: (batch, 77, 1024) tensor
    #     B: (batch, 77, 1024) tensor
    #     Returns: (batch, 77, 1024) projected tensor
    #     """
    #     A = p1[0]  # Remove batch dimension
    #     B = p2[0]

    #     # Compute the projection matrix P = A (A^T A)^{-1} A^T
    #     AtA = (A.T @ A).to(torch.float32)  # Convert before computation
    #     AtA_inv = torch.linalg.pinv(AtA).to(A.dtype)  # Convert back to original dtype
    #     P = A @ AtA_inv @ A.T

    #     # Project B onto A's subspace
    #     B_projected = P @ B
            
    #     # Restore batch dimension
    #     return B_projected.unsqueeze(0)



# Under construction

class SLerpInterpolator(BasePromptInterpolator):
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
                # print(e1.shape, e2.shape)
                euclidean_distance = np.linalg.norm(e1[-1, i, :] - e2[-1, i, :])
                distances[i][idx] = euclidean_distance

        times = np.arange(self.embeddings.shape[0], dtype=float)
        self.row_sums = distances.sum(axis=1, keepdims=True)

        # print(distances.shape, self.row_sums.shape)

        distances_normalized = distances / (self.row_sums + 1e-5)
        distances = distances_normalized * self.period
        distances = np.cumsum(distances, axis=1)
        zero_column = np.zeros((distances.shape[0], 1))
        self.times = np.hstack((zero_column, distances))
    
    def slerp(embedding1, embedding2, val):
        embedding1 = embedding1[0]
        embedding2 = embedding2[0]
        low_norm = embedding1 / torch.norm(embedding1, dim=1, keepdim=True)
        high_norm = embedding2 / torch.norm(embedding2, dim=1, keepdim=True)
        dot = (low_norm * high_norm).sum(1)
        omega = torch.acos(dot)
        so = torch.sin(omega)
        faktor1 = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1).unsqueeze(0)
        mask = torch.isnan(faktor1)
        mean = torch.mean(faktor1[~mask])
        faktor1[mask] = mean
        faktor2 = (torch.sin(val * omega) / so).unsqueeze(1).unsqueeze(0)
        mask = torch.isnan(faktor2)
        mean = torch.mean(faktor2[~mask])
        faktor2[mask] = mean
        res = faktor1 * embedding1 + faktor2 * embedding2
        return res

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









# === Interpolator Factory ===

def get_interpolator(method="nlerp"):
    if method == "nlerp_og":
        return NLerpInterpolatorOG
    elif method == "nlerp_og_no_std_decay":
        return NLerpInterpolatorOGNoStdDecay
    elif method == "nlerp_cosine_respacing_no_std_decay":
        return NLerpInterpolatorCosineRespacingNoDecay
    elif method == "nlerp_og_flux":
        return NLerpInterpolatorOG_Flux
    elif method == "stagewise_switcher":
        """
        StagewisePromptSwitcher does not interpolate but switches embeddings at fixed intervals.
        """
        return StagewisePromptSwitcher
    elif method == "stagewise_switcher_respaced":
        """
        StagewisePromptSwitcher does not interpolate but switches embeddings at fixed intervals.
        """
        return StagewisePromptSwitcherRespaced
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
