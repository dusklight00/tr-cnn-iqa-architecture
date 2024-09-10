import torch
from dataset import KadidDataset
from torchvision.transforms import ToTensor
from models.vit import ViT
from models.cnn import CNN
from models.trcnn import TrCNN
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import rgb_to_grayscale
import torch.optim as optim
import torch.nn as nn

def train(trcnn, cnn, vit, loader, learning_rate=0.01, device='cpu'):

  cnn_optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
  vit_optimizer = optim.Adam(vit.parameters(), lr=learning_rate)
  
  criterion = nn.MSELoss()

  total_loss = 0
  for batch in tqdm(loader, desc='Training'):
    x, y = batch

    x = rgb_to_grayscale(x).unsqueeze(1) 
    y = y.reshape(-1, 1).float()
    x, y = x.to(device), y.to(device)

    y_hat = trcnn(x)

    loss = criterion(y_hat, y)
    total_loss += loss.item()

    cnn_optimizer.zero_grad()
    vit_optimizer.zero_grad()

    loss.backward()

    cnn_optimizer.step()
    vit_optimizer.step()
  
  mean_loss = total_loss / len(loader)
  return mean_loss


VIT_MODEL_PATH = "vit.pth"
CNN_MODEL_PATH = "cnn.pth"
EPOOCHS = 10
SAVE_EACH_BATCH = False
SAVE_EACH_EPOCH = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = KadidDataset(
  csv_file="data/kadid10k/dmos.csv",
  root_dir="data/kadid10k/images",
  transform=ToTensor()
)

loader = DataLoader(data, batch_size=100, shuffle=True, num_workers=1)

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

cnn.load_state_dict(torch.load(CNN_MODEL_PATH))
vit.load_state_dict(torch.load(VIT_MODEL_PATH))

trcnn = TrCNN(cnn, vit).to(device)

if __name__ == '__main__':
  for epoch in range(EPOOCHS):
    loss = train(trcnn, cnn, vit, loader, device=device)
    print(f"Epoch: {epoch}, Loss: {loss}")
    if SAVE_EACH_BATCH:
      torch.save(cnn.state_dict(), CNN_MODEL_PATH)
      torch.save(vit.state_dict(), VIT_MODEL_PATH)
  if SAVE_EACH_EPOCH:
    torch.save(cnn.state_dict(), CNN_MODEL_PATH)
    torch.save(vit.state_dict(), VIT_MODEL_PATH)


