import torch
from dataset import KadidDataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from models.cnn import CNN
from models.dnn import DNN
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

data = KadidDataset(
  csv_file="data/kadid10k/dmos.csv",  
  root_dir="data/kadid10k/images",
  transform=ToTensor()
)

loader = DataLoader(data, batch_size=100, shuffle=True, num_workers=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cnn = CNN(
  diffusion_x=100, 
  diffusion_y=100
).to(device)

dnn = DNN(
  input_dim=(100, 100),
  hidden_dim=100,
  output_dim=1
).to(device)

class TrCNN(nn.Module):
    def __init__(self, cnn, dnn):
      super(TrCNN, self).__init__()
      self.cnn = cnn
      self.dnn = dnn

    def forward(self, x):
      x = self.cnn(x)
      x = self.dnn(x)
      return x

cnn_optimizer = optim.Adam(cnn.parameters(), lr=0.01)
dnn_optimizer = optim.Adam(dnn.parameters(), lr=0.01)

criterion = nn.MSELoss()

total_loss = 0
loss_history = []

if __name__ == '__main__':

  for batch in tqdm(loader):
    x, y = batch

    x = x.float()
    y = y.reshape(-1, 1).float()
    x, y = x.to(device), y.to(device)

    print(x.shape)

    trcnn = TrCNN(cnn, dnn)

    y_hat = trcnn(x)

    loss = criterion(y_hat, y)
    loss_history.append(loss.item())

    total_loss += loss.item()

    cnn_optimizer.zero_grad()
    dnn_optimizer.zero_grad()

    loss.backward()

    cnn_optimizer.step()
    dnn_optimizer.step()

    print(loss)

  mean_loss = total_loss / len(loader)
  print(mean_loss)