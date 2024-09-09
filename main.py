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

def rgb_to_grayscale(tensor):
    return 0.2989 * tensor[:, 0, :, :] + 0.5870 * tensor[:, 1, :, :] + 0.1140 * tensor[:, 2, :, :]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

# train_data = datasets.MNIST(
#     root = 'data',
#     train = True,
#     download = True,
#     transform = ToTensor()
# )

# test_data = datasets.MNIST(
#     root = 'data',
#     train = False,
#     download = True,
#     transform = ToTensor()
# )

train_data = KadidDataset(
  csv_file="data/kadid10k/dmos.csv",
  root_dir="data/kadid10k/images",
  transform=ToTensor()
)

train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1)
# test_loader = DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1)

criterion = nn.MSELoss()

cnn = CNN((100, 100)).to(device)
vit = ViT(
        (1, 100, 100), n_patches=10, n_blocks=2, hidden_d=8, n_heads=2, out_d=1
    ).to(device)

cnn_optimizer = optim.Adam(cnn.parameters(), lr=0.01)
vit_optimizer = optim.Adam(vit.parameters(), lr=0.01)

if __name__ == '__main__':
    
  for batch in tqdm(train_loader, desc='Training'):
    x, y = batch

    x = rgb_to_grayscale(x).unsqueeze(1) 
    y = y.reshape(-1, 1).float()
    x, y = x.to(device), y.to(device)
    y_hat = cnn(x)
    y_hat = vit(y_hat)
    loss = criterion(y_hat, y)
    cnn_optimizer.zero_grad()
    vit_optimizer.zero_grad()
    loss.backward()
    cnn_optimizer.step()
    vit_optimizer.step()

    print(f"Loss: {loss.item()}")

  # with torch.no_grad():
  #   correct = 0
  #   total = 0

  #   for images, labels in tqdm(test_loader, desc='Testing'):

  #     images = images.to(device)
  #     labels = labels.to(device)

  #     test_output = cnn(images)
  #     test_output = vit(test_output)

  #     pred_y = torch.max(test_output, 1)[1].data.squeeze()
  #     total += labels.size(0)

  #     correct += (pred_y == labels).sum().item()

  #   accuracy = correct / total
  #   print(f"Test Accuracy of the model on the {total} test images: {accuracy}")