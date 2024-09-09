import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from models.vit import ViT
from models.cnn import CNN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = datasets.MNIST(
    root = 'data',
    train = True,
    download = True,
    transform = ToTensor()
)

test_data = datasets.MNIST(
    root = 'data',
    train = False,
    download = True,
    transform = ToTensor()
)

train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1)
test_loader = DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1)

criterion = nn.CrossEntropyLoss()

cnn = CNN((12, 12)).to(device)
vit = ViT(
        (1, 12, 12), n_patches=3, n_blocks=2, hidden_d=8, n_heads=2, out_d=10
    ).to(device)

cnn_optimizer = optim.Adam(cnn.parameters(), lr=0.01)
vit_optimizer = optim.Adam(vit.parameters(), lr=0.01)

if __name__ == '__main__':
    
  for batch in train_loader:
      x, y = batch
      x, y = x.to(device), y.to(device)
      y_hat = cnn(x)
      y_hat = vit(y_hat)
      loss = criterion(y_hat, y)
      cnn_optimizer.zero_grad()
      vit_optimizer.zero_grad()
      loss.backward()
      cnn_optimizer.step()
      vit_optimizer.step()
      print(loss)