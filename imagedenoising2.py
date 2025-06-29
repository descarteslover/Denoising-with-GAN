import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor, ToPILImage
from skimage import data, color, metrics
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from PIL import Image 

# Set the environment variable to avoid the OpenMP warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'



# GENERATIVE ADVERSERIAL NETWORK MODEL
class DenoisingGenerator(nn.Module):
    def __init__(self):
        super(DenoisingGenerator, self).__init__()

        self.model = nn.Sequential(
            #This model is not a true generative adversarial network.
            #There is only a generative model which minimizes a loss function, as opposed to minimizing
            #the discriminator as is the case for the GAN.
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Load and prepare data
def load_and_preprocess_image(sigma=25):
    """Load a sample image, convert it to grayscale, and add Gaussian noise.
    Return the original and noisy images as PIL images.
    """
    original_image = data.camera()  # Grayscale image (512x512)
    
    # Convert to RGB (3 channels)
    original_image = np.stack([original_image]*3, axis=-1)
    original_image = Image.fromarray(original_image)
    
    # Create noisy version
    np_image = np.array(original_image)
    noisy_image = np_image + sigma * np.random.randn(*np_image.shape)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    noisy_image = Image.fromarray(noisy_image)
    
    return original_image, noisy_image

def calculate_metrics(original, denoised):
    """Calculate PSNR, SSIM, MSE, and NRMSE between original and denoised images."""
    original_np = np.array(original)
    denoised_np = np.array(denoised)

    original_gray =color.rgb2gray(original_np)
    denoised_gray = color.rgb2gray(denoised_np)

    metrics_dict = {
        'PSNR': metrics.peak_signal_noise_ratio(original_np, denoised_np),
        'SSIM': metrics.structural_similarity(original_gray, denoised_gray,data_range=1.0, channel_axis=None),
        'MSE': metrics.mean_squared_error(original_np, denoised_np),
        'NRMSE': metrics.normalized_root_mse(original_np, denoised_np)
        }
    return metrics_dict


sigma = 25  # Standard deviation of Gaussian noise
epochs = 1000
batch_size = 1
    
# Load images
original_image, noisy_image = load_and_preprocess_image(sigma)
    
# Convert images to tensors
to_tensor = ToTensor()
original_tensor = to_tensor(original_image).unsqueeze(0)
noisy_tensor = to_tensor(noisy_image).unsqueeze(0)
    
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_tensor = original_tensor.to(device)
noisy_tensor = noisy_tensor.to(device)
    
# Initialize model, loss and optimizer
generator = DenoisingGenerator().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(generator.parameters(), lr=0.001)
    
# Training loop
for epoch in tqdm(range(epochs), desc="Training"):
    optimizer.zero_grad()
    denoised = generator(noisy_tensor)
    loss = criterion(denoised, original_tensor)
    loss.backward()
    optimizer.step()
        
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.4f}")
   
# Load previously trained model
#generator.load_state_dict(torch.load(f'generator_model_n{sigma}_e{epochs}.pth'))
# Generate denoised image
with torch.no_grad():
    denoised_tensor = generator(noisy_tensor)
    
# Convert results to PIL images
to_pil = ToPILImage()
denoised_image = to_pil(denoised_tensor.squeeze(0).cpu())
    
#Calculate metrics
noisy_metrics = calculate_metrics(original_image, noisy_image)
denoised_metrics = calculate_metrics(original_image, denoised_image)

print("\nNoisy Image Metrics:" )
for name, value in noisy_metrics.items():
    print(f"{name}: {value:.4f}")

print("\nDenoised Image Metrics:")
for name, value in denoised_metrics.items():
    print(f"{name}: {value:.4f}")

# Plot results
plt.figure(figsize=(15, 5))
    
plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title("Original Image")
plt.axis('off')
    
plt.subplot(1, 3, 2)
plt.imshow(noisy_image)
plt.title(f"Noisy Image (sigma={sigma})")
plt.axis('off')
    
plt.subplot(1, 3, 3)
plt.imshow(denoised_image)
plt.title("Denoised Image")
plt.axis('off')
    
plt.tight_layout()
plt.show()
    
# Save results
denoised_image.save(f'denoised_image_n{sigma}_e{epochs} second run.jpg')
torch.save(generator.state_dict(), f'generator_model_n{sigma}_e{epochs}.pth')

