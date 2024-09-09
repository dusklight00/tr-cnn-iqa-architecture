import torch.nn as nn

class TrCNN(nn.Module):
    def __init__(self, cnn, vit):
      super(TrCNN, self).__init__()
      self.cnn = cnn
      self.vit = vit

    def forward(self, x):
      x = self.cnn(x)
      x = self.vit(x)
      return x