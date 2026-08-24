import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

class DepthEstimator:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.model_type = "MiDaS_small"

        self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.model.to(self.device)
        self.model.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = transforms.dpt_transform 

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image_pil = Image.fromarray(image_rgb)
        input_image_np = np.array(input_image_pil)
        input_tensor = self.transform(input_image_np).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # Replace invalid depth values (e.g., <= 0) with a small positive value
        depth_map[depth_map <= 0] = 1e-6

        # Debug: Print depth statistics
        print(f"Depth Map Stats - Min: {np.min(depth_map)}, Max: {np.max(depth_map)}, Mean: {np.mean(depth_map)}")

        return depth_map
    def save_depth_visualization(self, depth_map: np.ndarray, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(10, 8))
        plt.imshow(depth_map, cmap='inferno')  # You can also try 'plasma', 'magma', 'viridis'
        plt.colorbar(label='Relative Depth')
        plt.axis('off')
        plt.title('Estimated Depth Map')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()