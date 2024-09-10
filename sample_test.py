import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from models.vit import ViT
from models.cnn import CNN
from tqdm import tqdm
from dataset import KadidDataset
import torch.nn.functional as F
from skimage import io
from utils import rgb_to_grayscale
from models.trcnn import TrCNN
from utils import rgb_to_grayscale

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAMPLE_IMAGE_PATH = "sample.png"

def load_sample_image(image_path):
  transform = ToTensor()  
  sample_image = io.imread(image_path)
  sample_image = transform(sample_image)
  sample_image = sample_image.reshape(1, sample_image.shape[0], sample_image.shape[1], sample_image.shape[2])
  sample_image = rgb_to_grayscale(sample_image).unsqueeze(0)
  sample_image = sample_image.to(device)
  return sample_image

cnn = CNN(
  diffusion_x=100, 
  diffusion_y=100
).to(device)

vit = ViT(
  channel=1, 
  height=100, 
  width=100, 
  n_patches=10, 
  n_blocks=2, 
  hidden_d=8, 
  n_heads=2, 
  out_d=1
).to(device)

cnn.load_state_dict(torch.load("cnn.pth"))
vit.load_state_dict(torch.load("vit.pth"))

trcnn = TrCNN(cnn, vit).to(device)

sample_image = load_sample_image(SAMPLE_IMAGE_PATH)
sample_output = trcnn(sample_image)

print(f"Accuracy: {sample_output.item()}")

