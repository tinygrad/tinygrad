import torch
#  Define a simple model for demonstration purposes (replace with actual model)
def noise_schedule(timesteps):
    return torch.linspace(1e-4, 0.02, timesteps)

# Define a simple model for demonstration purposes (replace with actual model)
def diffuse(image, timesteps):
    noise = torch.randn_like(image)
    noisy_image = image + noise_schedule(timesteps) * noise
    return noisy_image, noise