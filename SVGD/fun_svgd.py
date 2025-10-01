import torch
from torch.linalg import cholesky
from scipy.spatial.distance import cdist
from scipy.special import kv, gamma
import numpy as np
from typing import Union, Literal
import time




def _default_seed():
    return int(time.time() * 1000000) % (2**32)

def matern_kernel(x, lengthscale=1.0, nu=1.5, output_std=1.0):
    x = np.atleast_2d(x)
    dists = cdist(x, x, metric='euclidean')
    if nu == 0.5:
        K = np.exp(-dists / lengthscale)
    else:
        sqrt_2nu_d = np.sqrt(2 * nu) * dists / lengthscale
        factor = (2 ** (1. - nu)) / gamma(nu)
        K = factor * (sqrt_2nu_d ** nu) * kv(nu, sqrt_2nu_d)
        K[np.isnan(K)] = 1.0
    return output_std* K  # For 2D particles, we only multiply it with output_std
                            # For 1D particles, we would multiply it with output_std ** 2

def generate_matern_2d_particles(
    num_pop,
    n1, n2,
    lengthscale_1=1.0, nu_1=1.5,
    lengthscale_2=1.0, nu_2=1.5,
    output_std=1.0,
    device="cpu",
    seed = None,
    dtype=torch.float32,
):
    """
    Generate 2D particles with separable Matérn covariance.
    """
    # Grid in each dimension
    x1 = np.linspace(0, 1, n1).reshape(-1, 1)
    x2 = np.linspace(0, 1, n2).reshape(-1, 1)

    # Separate Matérn kernels for each dimension
    K1 = matern_kernel(x1, lengthscale_1, nu_1, output_std)
    K2 = matern_kernel(x2, lengthscale_2, nu_2, output_std)
    K1 = torch.tensor(K1, device=device, dtype=dtype)
    K2 = torch.tensor(K2, device=device, dtype=dtype)

    # Kronecker product for 2D covariance
    K = torch.kron(K1, K2)  # shape: [n1*n2, n1*n2]

    # Add jitter for stability
    K = K + 1e-6 * torch.eye(n1 * n2, device=device, dtype=dtype)

    # Cholesky decomposition
    L = cholesky(K)

    if seed is None:
        seed = _default_seed()
    torch.manual_seed(seed)

    # Sample
    Z = torch.randn((num_pop, n1 * n2), device=device, dtype=dtype)
    particles = Z @ L.T

    return particles.reshape(num_pop, n1, n2)





# Generate a velocity model constrained to be within a desired range
# How to use this:
#   constrained_vel = Vel_Constraint(vel_model, min_vel, max_vel)
#   vel = constrained_vel() will return the velocity model with the constraints applied
#   vel.parameters() is a learnable parameter that will be optimized during training
# class Vel_Constraint(torch.nn.Module):
#     def __init__(self, vel_model, min_vel, max_vel):
#         super().__init__()
#         self.min_vel = min_vel
#         self.max_vel = max_vel
#         self.vel_model_parameter = torch.nn.Parameter(
#             torch.logit((vel_model - min_vel) /
#                         (max_vel - min_vel))
#         )

#     def forward(self):
#         return (torch.sigmoid(self.vel_model_parameter) *
#                 (self.max_vel - self.min_vel) +
#                 self.min_vel)

# constrained_vel = Vel_Constraint(v_population, vel_min, vel_max)



def rbf_kernel(x, h=-1, h_scaling = 1.0):
    pairwise_dists = torch.cdist(x, x, p=2)
    if h <= 0:
        h = torch.median(pairwise_dists).pow(2) / torch.log( torch.tensor(x.shape[0], dtype=torch.float32) + 1)
        h = h * h_scaling
        # print(f"Calculated RBF kernel bandwidth sqrt(h): {h.sqrt():.4f}")
    K = torch.exp( -pairwise_dists.pow(2) / h)

    # print(f"RBF kernel shape: {K.shape}, h: {h:.4f}")
    return K, h




def f_water_mask_population(n_population, n1, n2, water_n):
    """
    Create a mask for the water layer in the population velocity models.
    
    Args:
        n_population (int): Number of population members.
        n1 (int): First dimension size (e.g., depth).
        n2 (int): Second dimension size (e.g., horizontal).
        water_n (int): Number of layers in the water column.
        
    Returns:
        torch.Tensor: A mask tensor of shape (n_population, n1 * n2) with 0 in the water layer and 1 elsewhere.
    """
    nn = n1 * n2
    mask = torch.ones((n_population, nn), dtype=torch.float32)

    rows = torch.arange(n1) * n2  
    offsets = torch.arange(water_n)
    indices = (rows[:, None] + offsets).flatten()  # Shape [n1 * water_n]
    mask[:, indices] = 0.0  # Set water layer to 0

    return mask





def get_diffusion_schedule(
    num_iterations: int,
    schedule_type: Literal['linear', 'exponential', 'polynomial', 'sqrt', 'quadratic', 'cosine_annealing'] = 'linear',
    current_iteration: int = 0,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    power: float = 2.0,
    cosine_s: float = 0.008,
    return_tensor: bool = False
) -> Union[float, torch.Tensor, np.ndarray]:
    """
    Generate diffusion noise schedule for various commonly used scheduling strategies.
    
    Parameters:
    -----------
    num_iterations : int
        Total number of diffusion iterations/timesteps
    schedule_type : str
        Type of schedule ('linear', 'exponential', 'polynomial', 'sqrt', 'quadratic', 'cosine_annealing')
    current_iteration : int, default 0
        Current iteration/timestep (0 to num_iterations-1)
    beta_start : float, default 0.0001
        Starting beta value (noise level)
    beta_end : float, default 0.02
        Ending beta value (noise level)
    power : float, default 2.0
        Power for polynomial schedule
    cosine_s : float, default 0.008
        Small offset for cosine schedule to prevent beta from being too small
    return_tensor : bool, default False
        If True, returns torch.Tensor; if False, returns float
    
    Returns:
    --------
    float, torch.Tensor, or np.ndarray
        The diffusion coefficient (beta) for the current iteration
    """
    
    if current_iteration >= num_iterations:
        raise ValueError(f"current_iteration ({current_iteration}) must be less than num_iterations ({num_iterations})")
    
    if current_iteration < 0:
        raise ValueError(f"current_iteration ({current_iteration}) must be non-negative")
    
    # Normalize current iteration to [0, 1]
    t = current_iteration / (num_iterations - 1) if num_iterations > 1 else 0.0

    t = 1.0 - t  # Reverse time for diffusion process
    
    if schedule_type == 'linear':
        # Linear interpolation between beta_start and beta_end
        beta = beta_start + (beta_end - beta_start) * t
                
    elif schedule_type == 'exponential':
        # Exponential decay schedule
        beta = beta_start * (beta_end / beta_start) ** t
        
    elif schedule_type == 'polynomial':
        # Polynomial schedule
        beta = beta_start + (beta_end - beta_start) * (t ** power)
        
    elif schedule_type == 'sqrt':
        # Square root schedule
        beta = beta_start + (beta_end - beta_start) * np.sqrt(t)
        
    elif schedule_type == 'quadratic':
        # Quadratic schedule
        beta = beta_start + (beta_end - beta_start) * (t ** 2)
        
    elif schedule_type == 'cosine_annealing':
        # Cosine annealing schedule (different from cosine diffusion schedule)
        beta = beta_end + (beta_start - beta_end) * (1 + np.cos(np.pi * t)) / 2
        
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}. "
                        f"Choose from: 'linear', 'cosine', 'exponential', 'polynomial', 'sqrt', 'quadratic', 'cosine_annealing'")
    
    if return_tensor:
        return torch.tensor(beta, dtype=torch.float32)
    else:
        return float(beta)


def get_full_diffusion_schedule(
    num_iterations: int,
    schedule_type: str = 'linear',
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    power: float = 2.0,
    return_tensor: bool = False
) -> Union[np.ndarray, torch.Tensor]:
    """
    Generate the full diffusion schedule for all iterations at once.
    
    Returns:
    --------
    np.ndarray or torch.Tensor
        Array of beta values for all iterations
    """
    
    betas = []
    for i in range(num_iterations):
        beta = get_diffusion_schedule(
            num_iterations=num_iterations,
            schedule_type=schedule_type,
            current_iteration=i,
            beta_start=beta_start,
            beta_end=beta_end,
            power=power,
            return_tensor=False
        )
        betas.append(beta)
    
    betas = np.array(betas)
    
    if return_tensor:
        return torch.tensor(betas, dtype=torch.float32)
    else:
        return betas

